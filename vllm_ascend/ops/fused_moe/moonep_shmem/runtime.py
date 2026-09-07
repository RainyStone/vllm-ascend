#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
"""MoonEP 对称内存运行时：Buffer 池 / 共享权重槽 / 每层状态 / 两段式初始化。

移植自 MindSpeed-MM 训练侧 shmem_ep_dispatcher.py（commit 204a1311），
推理侧裁剪说明：

  * **裁掉全部梯度侧设施**：home_g / reduce_b / full_g / rb 梯度表、
    reduce_grad 调用、GmmSinkFunction、跨 rank 反向 barrier 均不迁移
    （推理无反向）。
  * **权重 publish 时机**：训练版每次 forward 重新发布 home 权重（权重随
    优化器更新）；推理权重静态，改为加载完成后一次性 publish
    （``MoonEPLayerState.publish_weights``）。
  * **B 预取槽全模型共享一份**：复用契约与训练版一致——同一 rank 任意
    时刻至多一个 MoE 层使用 B 槽（fwd 逐层串行，prefetch→GEMM→combine
    完整包含在单层 forward 内）。因此 v1 与 DBO / 跨层多流交叠互斥。

内存布局（与训练版一致）：
  * 权重表为**全局 [E+B] 行布局**的 VMM 投影：home 段（本 rank 属主专家）
    在 ``rank*epn`` 行，slot 段在 ``E`` 行起；供 prefetch_weight 寻址。
  * GEMM 需要 [epn+B]（masters→slots 组序）的权重，由
    ``_create_packed_projection`` 构造零拷贝 VA 别名（home 段在 0 行、
    slot 段在 epn 行）。
  * **down 投影转置存储**：ascend-moonep 的 prefetch VMM 路径以 ctx['H']
    （dispatch payload 的 hidden）为权重行数，要求三个投影行数都等于 H；
    down 逻辑形状 [H_f, H]（rows=H_f≠H），故 down 的 VMM 表一律按转置后
    [H, H_f] 存储，边界处转置（publish 时 ``w2.transpose(1,2)`` 写入、
    GEMM 时传 ``compute_w["down"].transpose(1,2)`` 视图）。内核只按行字节
    寻址，对内容朝向无感。

运行时为两段式：Buffer/VMM chunk 构造只登记 descriptor，全部层构造完成后
统一 ``ShmemRuntime.init_with_buffer``（UID bootstrap，descriptor 快照须
覆盖全部分配），之后才能 dispatch。
"""

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from vllm.logger import logger


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def physical_tokens_per_expert(cu_seqlens: torch.Tensor, rank: int, epn: int,
                               E: int) -> torch.Tensor:
    """diff(cu_seqlens) → [masters(epn) | slots(B)] 的本 rank 物理组计数 [epn+B]。

    cu_seqlens 的组 0..E-1 按全局专家号排列，本 rank 的本地专家段对应下标
    [rank*epn, (rank+1)*epn)；组 E..E+B-1 为副本槽段。diff 的首项即
    cu_seqlens[0]（前一项视为 0）。
    """
    counts = cu_seqlens - torch.cat((cu_seqlens.new_zeros(1), cu_seqlens[:-1]))
    return torch.cat((counts[rank * epn:(rank + 1) * epn], counts[E:]))


def storage_shape(name: str, rows: int, cols: int) -> Tuple[int, int]:
    """VMM 存储形状：down 投影转置存储（逻辑 [H_f,H] → 物理 [H,H_f]）。

    ascend-moonep 的 prefetch VMM 路径以 ctx['H']（dispatch payload 的
    hidden）为权重行数，down 逻辑行数 H_f≠H 会寻址错位；转置后行数=H 吻合，
    内核只按行字节寻址、对内容朝向无感（上游无需改动）。
    """
    return (cols, rows) if name == "down" else (rows, cols)


def _create_packed_projection(home: torch.Tensor, slot: torch.Tensor,
                              device: int) -> torch.Tensor:
    """构造 [epn+B, H, H'] 打包 VA 别名（home 段在 0 行、slot 段在 epn 行）。

    GEMM 组序权重用的零拷贝视图：与 [E+B] 全局投影共享同一物理内存。
    要求 epn*row 按 VMM 分配粒度对齐（否则 slot 无法落在第 epn 行边界）。
    """
    from ascend_moonep.buffer_c import (
        VmmProjection,
        acl_vmm_granularity,
        acl_vmm_map,
        acl_vmm_reserve,
        acl_vmm_set_access,
        create_sym_tensor_from_ptr,
    )

    epn = int(home.shape[0])
    B = int(slot.shape[0])
    H, Hp = int(home.shape[1]), int(home.shape[2])
    row = H * Hp * int(home.element_size())
    gran = int(acl_vmm_granularity(int(device)))
    home_off = 0
    slot_off = epn * row
    if slot_off % gran != 0:
        raise RuntimeError(
            f"packed [epn+B] projection requires epn*row aligned to VMM gran: "
            f"epn*row={slot_off} gran={gran} (epn={epn}, row={row}). "
            f"请调整 EP 尺寸（epn=E/R）或投影形状使其满足对齐。"
        )
    home_alloc = getattr(home, "_vmm_allocation", None)
    slot_alloc = getattr(slot, "_vmm_allocation", None)
    if home_alloc is None or slot_alloc is None:
        raise ValueError("home/slot must be VMM chunks (create_vmm_physical)")
    dev = int(device)
    total = slot_off + int(slot_alloc.size)
    base = int(acl_vmm_reserve(total))
    segments = []
    try:
        for alloc, off in ((home_alloc, home_off), (slot_alloc, slot_off)):
            va = base + off
            acl_vmm_map(va, int(alloc.size), alloc.mem_handle, 0)
            acl_vmm_set_access(va, int(alloc.size), dev)
            segments.append((va, int(alloc.size), alloc.mem_handle))
    except Exception:
        from ascend_moonep.buffer_c import acl_vmm_release, acl_vmm_unmap
        for va, _, _ in segments:
            try:
                acl_vmm_unmap(va)
            except Exception:
                pass
        try:
            acl_vmm_release(base)
        except Exception:
            pass
        raise
    tensor = create_sym_tensor_from_ptr(
        base, [epn + B, H, Hp], home.dtype, torch.device(f"npu:{dev}"))
    tensor._projection = VmmProjection(base, segments)
    return tensor


# ---------------------------------------------------------------------------
# Buffer 池（同一 (S,H,K,E,R,group) 配置的 Buffer 全模型单例）
# ---------------------------------------------------------------------------
class MoonEPBufferPool:
    """Per-(S,H,K,E,R,group) ascend-moonep Buffer singleton.

    注意：Buffer 构造只登记 VMM descriptor，不初始化 shmem 运行时——运行时在
    init_moonep_shmem_states 收尾处统一拉起（descriptor 快照须覆盖全部分配）。
    """

    _lock = threading.Lock()
    _buffers: Dict[Tuple, Any] = {}

    @classmethod
    def get(
        cls,
        S: int, H: int, K: int, E: int, R: int, B: int,
        group: dist.ProcessGroup, token_padding: int,
    ) -> Any:
        from ascend_moonep import Buffer

        rank = dist.get_rank(group)
        # urma_dev 起 Buffer 支持 use_udma kwarg（Ascend950 默认开 UDMA）。
        # MOONEP_DISABLE_UDMA=1 时强制回退 MTE（排查/兼容用）。
        use_udma = os.environ.get("MOONEP_DISABLE_UDMA", "0") != "1"
        key = (S, H, K, E, R, rank, id(group), use_udma)
        with cls._lock:
            buf = cls._buffers.get(key)
            if buf is None:
                logger.info(
                    "[moonep_shmem] 创建 ascend-moonep Buffer: S=%d H=%d K=%d "
                    "E=%d R=%d B=%d token_padding=%d use_udma=%s",
                    S, H, K, E, R, B, token_padding, use_udma,
                )
                buf = Buffer(
                    S, H, K, E, R,
                    token_padding=token_padding, B=B, group=group,
                    explicitly_destroy=True,
                    **({"use_udma": use_udma} if not use_udma else {}),
                )
                cls._buffers[key] = buf
        return buf

    @classmethod
    def destroy_all(cls) -> None:
        with cls._lock:
            for buf in cls._buffers.values():
                try:
                    buf.destroy()
                except Exception:
                    pass
            cls._buffers.clear()


# ---------------------------------------------------------------------------
# 共享权重槽（B 个预取槽，全 MoE 层共享一份）
# ---------------------------------------------------------------------------
class SharedSlotPool:
    """全层共享的 B 个权重预取槽（bf16）。

    复用安全性（训练版论证的推理侧简化）：权重槽的使用完整包含在单层
    forward 内（prefetch → GEMM → combine，同 rank 单流串行）；推理无反向，
    不存在梯度缓冲的跨层复用问题。full_w 的 home 段不可共享（权重值各层
    不同）。若未来引入跨层 MoE 交叠（DBO/多流），须改为按流多槽或加占用锁。
    """

    _lock = threading.Lock()
    _pools: Dict[Tuple, Dict[str, torch.Tensor]] = {}

    @classmethod
    def get(
        cls,
        epn: int,
        B: int,
        E: int,
        rank: int,
        proj_shapes: Dict[str, Tuple[int, int]],
        dev: int,
    ) -> Dict[str, torch.Tensor]:
        from ascend_moonep.buffer_c import create_vmm_physical

        key = (epn, B, E, rank, tuple(sorted(proj_shapes.items())), int(dev))
        with cls._lock:
            pool = cls._pools.get(key)
            if pool is None:
                slot_w: Dict[str, torch.Tensor] = {}
                total_bytes = 0
                for name, (rows, cols) in proj_shapes.items():
                    rs, cs = storage_shape(name, rows, cols)
                    # B 个权重预取槽（bf16，全层共享）
                    slot_w[name] = create_vmm_physical(
                        [B, rs, cs], torch.bfloat16, dev)[0]
                    slot_w[name].zero_()
                    total_bytes += slot_w[name].numel() * slot_w[name].element_size()
                pool = slot_w
                cls._pools[key] = pool
                logger.info(
                    "[moonep_shmem] 创建全层共享权重预取槽: B=%d epn=%d E=%d "
                    "rank=%d dev=%d 共 %.1f MiB",
                    B, epn, E, rank, dev, total_bytes / 1024 / 1024,
                )
        return pool

    @classmethod
    def destroy_all(cls) -> None:
        with cls._lock:
            cls._pools.clear()


# ---------------------------------------------------------------------------
# 每层状态（初始化时构建一次，权重加载后 publish 一次）
# ---------------------------------------------------------------------------
class MoonEPLayerState:
    """单层 MoE 路由专家的对称张量与计算视图。

    全局行布局（VMM 投影）：
      full_w[name]    [E+B, rows, cols] bf16 投影：home 段（本 rank 属主专家）
                      在 [rank*epn, (rank+1)*epn) 行，slot 段在 [E, E+B) 行；
                      供 prefetch_weight（算子经挂载的 source VA 寻址）。
      home_w[name]    [epn, rows, cols] bf16 本层 home chunk（权重发布目标）。
      compute_w[name] [epn+B, rows, cols] bf16 打包别名（home 在 0 行、slot
                      在 epn 行）——GEMM 组序权重（masters → slots），零拷贝。
    """

    def __init__(
        self,
        buffer: Any,
        rank: int,
        epn: int,
        E: int,
        B: int,
        H: int,
        K: int,
        H_f: int,
        proj_shapes: Dict[str, Tuple[int, int]],
        layer_name: str,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        from ascend_moonep.buffer_c import (
            attach_local_source_vas,
            create_full_projection_tensor,
            create_vmm_physical,
        )

        self.buffer = buffer
        self.group = ep_group
        self.rank, self.epn, self.E, self.B = rank, epn, E, B
        self.H, self.K, self.H_f = H, K, H_f
        self.proj_shapes = proj_shapes
        self.layer_name = layer_name

        dev = int(torch.npu.current_device())
        self.dev = dev

        # B 个预取槽全模型共享（同一 (epn,B,E,shapes) 配置下只建一份）
        self.slot_w = SharedSlotPool.get(epn, B, E, rank, proj_shapes, dev)

        # 每层独立的 home 权重 chunk + [E+B] 投影 + [epn+B] 打包计算视图
        self.home_w: Dict[str, torch.Tensor] = {}
        self.full_w: Dict[str, torch.Tensor] = {}
        self.compute_w: Dict[str, torch.Tensor] = {}
        for name, (rows, cols) in proj_shapes.items():
            rs, cs = storage_shape(name, rows, cols)
            hw = create_vmm_physical([epn, rs, cs], torch.bfloat16, dev)[0]
            hw.zero_()
            self.home_w[name] = hw
            fw = create_full_projection_tensor(
                hw, self.slot_w[name], E, B, rank, dev)
            attach_local_source_vas(
                fw, hw.data_ptr(), self.slot_w[name].data_ptr())
            self.full_w[name] = fw
            self.compute_w[name] = _create_packed_projection(
                hw, self.slot_w[name], dev)
        logger.info(
            "[moonep_shmem] 层 %s 状态就绪: epn=%d E=%d B=%d H=%d H_f=%d dev=%d",
            layer_name, epn, E, B, H, H_f, dev,
        )

    @torch.no_grad()
    def publish_weights(self, w13_weight: torch.Tensor,
                        w2_weight: torch.Tensor) -> None:
        """把本 rank 属主专家权重一次性发布进 home chunk（加载完成后调用）。

        vllm 权重加载完成后（AscendUnquantizedFusedMoEMethod.
        process_weights_after_loading 已做 transpose）的逻辑布局：
          w13_weight [epn, H, 2*H_f]（gate 在前、up 在后）
          w2_weight  [epn, H_f, H]
        MoonEP home chunk 约定：
          gate/up  [epn, H, H_f]（原样）
          down     [epn, H, H_f]（转置存储，逻辑 [epn, H_f, H] → 物理 [epn, H, H_f]，
                    见模块 docstring 的 storage_shape 说明）
        """
        if w13_weight.shape[0] != self.epn:
            raise ValueError(
                f"[moonep_shmem] 层 {self.layer_name} 本地专家数 "
                f"{w13_weight.shape[0]} 与 epn={self.epn} 不一致"
            )
        if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
            raise ValueError(
                f"[moonep_shmem] v1 仅支持 bf16 权重，层 {self.layer_name} "
                f"实际 dtype: w13={w13_weight.dtype} w2={w2_weight.dtype}"
            )
        self.home_w["gate"].copy_(w13_weight[:, :, : self.H_f])
        self.home_w["up"].copy_(w13_weight[:, :, self.H_f:])
        self.home_w["down"].copy_(w2_weight.transpose(1, 2))
        logger.info(
            "[moonep_shmem] 层 %s 权重已发布到 home chunk（epn=%d）",
            self.layer_name, self.epn,
        )


# ---------------------------------------------------------------------------
# 运行时引导（两段式：先构造全部 Buffer/VMM chunk，再统一起 shmem 运行时）
# ---------------------------------------------------------------------------
def shmem_runtime_initialized() -> bool:
    from ascend_moonep import ShmemRuntime

    return ShmemRuntime.is_initialized()


def _ensure_shmem_runtime(ep_group) -> None:
    """全部 Buffer/VMM chunk 构造完成后统一拉起 shmem 运行时（UID bootstrap）。"""
    from ascend_moonep import ShmemRuntime

    if not ShmemRuntime.is_initialized():
        logger.info("[moonep_shmem] 拉起 ShmemRuntime（UID bootstrap）...")
        ShmemRuntime.init_with_buffer(group=ep_group)
        logger.info("[moonep_shmem] ShmemRuntime 初始化完成")


def init_moonep_shmem_states(
    moe_runners: List[Any],
    ep_group: dist.ProcessGroup,
    max_tokens_per_rank: int,
    num_slots_B: Optional[int] = None,
    token_padding: int = 1,
) -> None:
    """为每个 MoE 层构建对称内存状态并发布权重（模型加载完成后调用一次）。

    调用点：NPUModelRunner.load_model 的 DeviceMemoryProfiler 窗口内
    （VMM 分配不走 torch caching allocator，必须计入显存 profiling，
    否则 KV cache 会超分配导致 OOM）。

    Args:
        moe_runners: AscendMoERunner 实例列表（每个对应一层 MoE）。
        ep_group: EP 进程组。
        max_tokens_per_rank: 每 rank 最大 token 数 S（MoonEP Buffer 静态形状，
            推理侧取 ceil(max_num_batched_tokens / tp_size)）。
        num_slots_B: 每 rank 权重预取槽数，None=epn（每属主专家一个槽）。
        token_padding: 每个 VM group 段向上对齐的 token 数（昇腾默认 1）。
    """
    from vllm_ascend.ops.fused_moe.moonep_shmem import check_ascend_moonep_available

    check_ascend_moonep_available()

    if not moe_runners:
        raise ValueError("[moonep_shmem] 未找到任何 MoE 层，无法初始化 shmem 状态")
    if max_tokens_per_rank <= 0:
        raise ValueError(
            f"[moonep_shmem] max_tokens_per_rank 必须为正，got {max_tokens_per_rank}"
        )

    R = dist.get_world_size(ep_group)
    rank = dist.get_rank(ep_group)

    for layer_idx, runner in enumerate(moe_runners):
        moe_config = runner.moe_config
        # num_logical_experts：全局逻辑专家数（shmem 与冗余专家互斥，
        # 配置守卫已保证 num_experts == num_logical_experts）
        E = int(moe_config.num_logical_experts)
        K = int(moe_config.experts_per_token)
        if E % R != 0:
            raise ValueError(
                f"[moonep_shmem] 层 {runner.layer_name}: E({E}) 必须整除 R({R})"
            )
        epn = E // R
        B = epn if num_slots_B is None else int(num_slots_B)

        # 从加载后的实际权重推导投影形状（w13 [epn, H, 2H_f]，w2 [epn, H_f, H]）
        w13 = runner.routed_experts.w13_weight
        w2 = runner.routed_experts.w2_weight
        if w13.shape[0] != epn:
            raise ValueError(
                f"[moonep_shmem] 层 {runner.layer_name}: 本地专家数 "
                f"{w13.shape[0]} != epn({epn})，请检查 EP 切分"
            )
        H = int(w13.shape[1])
        H_f = int(w13.shape[2] // 2)
        if w2.shape[1] != H_f or w2.shape[2] != H:
            raise ValueError(
                f"[moonep_shmem] 层 {runner.layer_name}: w2 形状 "
                f"{tuple(w2.shape)} 与期望 [epn, {H_f}, {H}] 不符"
            )
        proj_shapes = {"gate": (H, H_f), "up": (H, H_f), "down": (H_f, H)}

        buffer = MoonEPBufferPool.get(
            max_tokens_per_rank, H, K, E, R, B, ep_group, token_padding)
        state = MoonEPLayerState(
            buffer, rank, epn, E, B, H, K, H_f, proj_shapes,
            layer_name=runner.layer_name, ep_group=ep_group,
        )
        # 推理权重静态：构建状态后立即发布，之后每次 forward 无需再拷贝
        state.publish_weights(w13.data, w2.data)
        runner.routed_experts._moonep_state = state
        logger.info(
            "[moonep_shmem] 层 %d/%d (%s) 初始化完成: S=%d E=%d epn=%d B=%d K=%d",
            layer_idx + 1, len(moe_runners), runner.layer_name,
            max_tokens_per_rank, E, epn, B, K,
        )

    # descriptor 快照须覆盖全部 VMM 分配（Buffer + 各层 home + 共享 slot），
    # 因此必须在所有层构造完成后统一拉起运行时
    _ensure_shmem_runtime(ep_group)
    logger.info(
        "[moonep_shmem] 全部 %d 层 MoE 状态初始化完成，shmem 运行时就绪",
        len(moe_runners),
    )


def destroy_all_moonep_states() -> None:
    """进程退出前清理：销毁 Buffer 池并 finalize shmem 运行时。"""
    from ascend_moonep import ShmemRuntime

    MoonEPBufferPool.destroy_all()
    SharedSlotPool.destroy_all()
    if ShmemRuntime.is_initialized():
        ShmemRuntime.finalize()
    logger.info("[moonep_shmem] shmem 运行时已 finalize")
