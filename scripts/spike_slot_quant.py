"""MoonEP slot 模式 × W4A8MXFP 量化专家（DSv4-Flash）：卡上只读诊断 spike。

配套文档：docs/moonep_slot_quant.md。本脚本**不改产品代码**、不 init shmem、
不加载模型，只做取证；所有判据在打印里给出 PASS/FAIL。

  S1  产品代码路径：直接在 moonep_balance._LayerWeightState 上建一层量化
      state（E=2 就够，~30 MiB、无需 shmem/8 卡），验 _infer_F / publish /
      prefetch_views / compute_weights，并要求 alias 上的 gmm1 与真实权重
      bit-identical —— M0-M6 验前提，S1 验真代码

  S2  引擎真正调的那条链：build_fused_experts_input → run_slot_forward（buffer
      用空替身）→ unified_apply_mlp，覆盖融合 gmm1+swiglu 与 gmm2 两个封装算子，
      判据是 alias 与真实权重**逐比特一致**（S1 用的是裸 gmm1，不覆盖这两个）

  M0  NZ(29) 元数据零拷贝构造探针 —— 自建 NZ 张量是否与 npu_format_cast 等价
  M1  四个 payload 处理后元数据快照（w13 / w13_scale / w2 / w2_scale）
  M2  裸字节视图可读（flat [nbytes] uint8 + clone）
  M3  逐字节镜像一致性（home 载体 ← 权重 storage）
  M4  alias 数值透明性 —— **生死判据**：自建 NZ alias 上的 gmm1 与真实权重
      bit-identical（走 A5 的 W4A8MXFP antiquant 调用，见 device_op.py:1362-1377）
  M6  VMM 账：真实几何下单层四件套能否建起来（gran/lcm 约束）+ 全模型 VA 总量
      （M6 需要 ascend_moonep 的 VMM 绑定，缺依赖时自动 SKIP）

用法：
  # 单卡（选一张有余量的空卡）
  ASCEND_RT_VISIBLE_DEVICES=0 python scripts/spike_slot_quant.py
  # 指定 epn（默认按 DSv4/R=8 的 32 跑单层 VMM 原型）
  ASCEND_RT_VISIBLE_DEVICES=0 python scripts/spike_slot_quant.py --epn 4
  # 只跑某几项
  ASCEND_RT_VISIBLE_DEVICES=0 python scripts/spike_slot_quant.py --only M0 M1 M4
  # 引擎侧打点自检（MOONEP_SLOT_STATS=1 的日志）
  python scripts/spike_slot_quant.py --check-stats /path/to/prefill.log
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ASCEND_CUSTOM_OPP_PATH 必须在 import torch_npu / CANN 初始化**之前**设好：
# aclnn 的 vendor 解析在 CANN init 时读一次就缓存了，之后再 setenv 已经晚了。
# 实测（2026-09-17，A5/950PR）：只在 main() 里调 bootstrap_custom_op_env()，
# aclnnSwigluGroupQuant 会落到**内置**算子并报
# "y dtype must be hifloat8, got 36"（swiglu_group_quant_hifp8_tiling.cpp:103）；
# 进程启动时就设好，才走仓内 vendor 算子并正常返回
# (y=(T,2F) fp8_e4m3fn, scale=(T,F/64,2) e8m0)。引擎侧
# AscendPlatform.import_kernels → bootstrap_custom_op_env 发生在引擎 init
# （早于设备初始化），所以引擎那条路本来就是对的；spike 得自己补这一段。
_VENDOR_DIR = (Path(__file__).resolve().parents[1] / "vllm_ascend"
               / "_cann_ops_custom" / "vendors" / "custom_transformer")
if _VENDOR_DIR.is_dir():
    _prev = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    os.environ["ASCEND_CUSTOM_OPP_PATH"] = (
        f"{_VENDOR_DIR}:{_prev}" if _prev else str(_VENDOR_DIR))

import torch  # noqa: E402
import torch_npu  # noqa: E402, F401

FP8 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
F4X2 = torch_npu.float4_e2m1fn_x2

# DSv4-Flash-0731 的 MoE 几何（config.json：hidden_size=4096,
# moe_intermediate_size=2048, n_routed_experts=256, num_hidden_layers=43）
DSV4_H = 4096
DSV4_F = 2048
DSV4_E = 256
DSV4_LAYERS = 43
# config.json 里 swiglu_limit=10.0：融合 gmm1 会对 gate 做 clamp，S2 用真实值
DSV4_SWIGLU_LIMIT = 10.0

_RESULTS: list[tuple[str, bool | None, str]] = []


def report(tag: str, ok: bool | None, msg: str) -> None:
    state = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    _RESULTS.append((tag, ok, msg))
    print(f"[{tag}] {state}  {msg}", flush=True)


# ---------------------------------------------------------------------------
# 本地工具：元数据构造（create_sym_tensor_from_ptr 的替代，多一个 npu_format 键）
# ---------------------------------------------------------------------------
def _prod(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _contiguous_strides(shape):
    if len(shape) == 0:
        return ()
    strides = [1]
    for d in reversed(shape[1:]):
        strides.insert(0, strides[0] * d)
    return tuple(strides)


def sym_tensor_from_ptr(ptr, shape, dtype, device, stride=None,
                        storage_offset=0, nbytes=None, npu_format=None):
    """在已有设备地址上造非 owning 张量。

    对照组：third_party/ascend_moonep/ascend_moonep/buffer_c.py:396-434
    （``create_sym_tensor_from_ptr``，纯 Python，无 C++ 实现）。这里多传一个
    ``npu_format`` 键 —— Phase 2 若要把它收进库里，就是这个改法（零编译）。
    """
    shape = tuple(int(d) for d in shape)
    if stride is None:
        stride = _contiguous_strides(shape)
    if nbytes is None:
        nbytes = _prod(shape) * torch.empty((), dtype=dtype).element_size()
    metadata = {
        "data_ptr": int(ptr),
        "device": torch.device(device),
        "nbytes": int(nbytes),
        "dtype": dtype,
        "size": list(shape),
        "stride": [int(s) for s in stride],
        "storage_offset": int(storage_offset),
    }
    if npu_format is not None:
        metadata["npu_format"] = int(npu_format)
    storage = torch_npu._C._construct_storage_from_data_pointer(
        metadata["data_ptr"], metadata["device"], metadata["nbytes"])
    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(
        metadata, storage)


def flat_bytes(t: torch.Tensor) -> torch.Tensor:
    """把张量的**底层 storage** 当平坦 uint8 看（绕开非连续视图不能 reshape）。

    起点用 ``untyped_storage().data_ptr()`` 而非 ``data_ptr()``，所以
    storage_offset 已包含在内 —— 与镜像拷贝的语义一致。
    """
    st = t.untyped_storage()
    return sym_tensor_from_ptr(st.data_ptr(), (st.nbytes(),), torch.uint8,
                               t.device, nbytes=st.nbytes())


# ---------------------------------------------------------------------------
# 复刻 AscendW4A8MXFPDSDynamicFusedMoEMethod.process_weights_after_loading
# （vllm_ascend/quantization/methods/fp8.py:108-130）
# ---------------------------------------------------------------------------
def process_weight_like_ds(w: torch.Tensor) -> torch.Tensor:
    """w13/w2：npu_format_cast(NZ, 29, customize=fp8, input=fp4x2) 再 transpose(1,2)。"""
    cast = torch_npu.npu_format_cast(
        w.view(torch.uint8), 29, customize_dtype=FP8, input_dtype=F4X2)
    return cast.transpose(1, 2)


def process_scale_like_ds(s: torch.Tensor) -> torch.Tensor:
    """两个 scale：reshape(g, n, k//2, 2).view(uint8).transpose(-3, -2)。"""
    g, n, k = s.shape
    return s.reshape(g, n, k // 2, 2).view(torch.uint8).transpose(-3, -2)


def payload_sizes(H: int, F: int):
    """四个 payload 的 (名字, 打包后形状(不含 E 维), 逐专家字节数)。

    形状来自 w4a8_mxfp4.py:117-141 的分配；逐专家字节数即处理后的
    ``stride(0) * element_size()``（M1 会实测校验）。
    """
    return [
        ("w13", (2 * F, H // 2), (2 * F) * (H // 2)),
        ("w13_scale", (2 * F, H // 32), (2 * F) * (H // 32)),
        ("w2", (H, F // 2), H * (F // 2)),
        ("w2_scale", (H, F // 32), H * (F // 32)),
    ]


def carrier_shape(row_bytes: int, max_row_elems: int = 32768):
    """把 row_bytes 分解成 bf16「字节载体」形状 (R, C)：R*C*2 == row_bytes。

    约束来自 prefetch kernel（ascend-moonep/kernels/prefetch.cpp:97-105）：
    ``elements_per_row = Hp`` 且要求 ``ub_bytes/2/Hp >= 1``（UB 内部暂存区
    < 188 KiB），所以 Hp 取 32768（32 KiB 元素 = 64 KiB 字节）封顶。
    """
    elems = row_bytes // 2
    if elems * 2 != row_bytes or elems <= 0:
        raise ValueError(f"row_bytes={row_bytes} 不是 2 的正倍数")
    c = min(elems, max_row_elems)
    while c > 1 and elems % c:
        c //= 2
    return (elems // c, c)


# ---------------------------------------------------------------------------
# M0：NZ 元数据探针
# ---------------------------------------------------------------------------
def check_M0(dev: str, args) -> None:
    src = torch.arange(64, dtype=torch.bfloat16, device=dev)
    md = dict(data_ptr=src.data_ptr(), device=torch.device(dev), nbytes=128,
              dtype=torch.bfloat16, size=[2, 4, 8], stride=[32, 8, 1],
              storage_offset=0)
    storage = torch_npu._C._construct_storage_from_data_pointer(
        md["data_ptr"], md["device"], md["nbytes"])
    plain = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(md, storage)
    nz = sym_tensor_from_ptr(src.data_ptr(), (2, 4, 8), torch.bfloat16, dev,
                             stride=(32, 8, 1), nbytes=128, npu_format=29)
    f_plain = int(torch_npu.get_npu_format(plain))
    f_nz = int(torch_npu.get_npu_format(nz))
    same = (tuple(plain.shape) == tuple(nz.shape)
            and plain.stride() == nz.stride()
            and plain.untyped_storage().nbytes() == nz.untyped_storage().nbytes()
            and plain.data_ptr() == nz.data_ptr()
            and plain.dtype == nz.dtype)
    print(f"       plain: format={f_plain} shape={tuple(plain.shape)} "
          f"stride={plain.stride()} storage={plain.untyped_storage().nbytes()}")
    print(f"       npu_format=29: format={f_nz} shape={tuple(nz.shape)} "
          f"stride={nz.stride()} storage={nz.untyped_storage().nbytes()}")
    report("M0", f_nz == 29 and same,
           f"format {f_plain}->{f_nz}，shape/stride/storage/dtype 逐字段不变={same}")


# ---------------------------------------------------------------------------
# M1/M2/M3：payload 元数据 + 裸字节视图 + 逐字节镜像
# ---------------------------------------------------------------------------
def make_payloads(dev: str, E: int, H: int, F: int, seed: int = 0):
    """造 E 个专家的四个 payload（真实几何但 E 很小），返回处理后的张量。"""
    g = torch.Generator(device=dev).manual_seed(seed)
    w13 = torch.randint(0, 256, (E, 2 * F, H // 2), dtype=torch.uint8,
                        device=dev, generator=g)
    w2 = torch.randint(0, 256, (E, H, F // 2), dtype=torch.uint8,
                       device=dev, generator=g)
    # scale 的 e8m0 位型取 [120,136) 区间（约 2^-7..2^8），避开 0xFF(NaN)
    s13 = torch.randint(120, 136, (E, 2 * F, H // 32), dtype=torch.uint8,
                        device=dev, generator=g).view(E8M0)
    s2 = torch.randint(120, 136, (E, H, F // 32), dtype=torch.uint8,
                       device=dev, generator=g).view(E8M0)
    return {
        "w13": process_weight_like_ds(w13),
        "w2": process_weight_like_ds(w2),
        "w13_scale": process_scale_like_ds(s13),
        "w2_scale": process_scale_like_ds(s2),
    }


def _meta_row(t: torch.Tensor) -> dict:
    st = t.untyped_storage()
    return {
        "shape": tuple(t.shape),
        "stride": tuple(t.stride()),
        "dtype": str(t.dtype).replace("torch.", ""),
        "esize": t.element_size(),
        "format": int(torch_npu.get_npu_format(t)),
        "contig": bool(t.is_contiguous()),
        "soff": int(t.storage_offset()),
        "nbytes": int(st.nbytes()),
        "numel_bytes": t.numel() * t.element_size(),
        "stride0_bytes": int(t.stride(0)) * t.element_size(),
    }


def check_M1(dev: str, args) -> None:
    E = 2  # 两个专家就够把「逐专家步长」测出来，显存 ~26 MiB
    H, F = DSV4_H, DSV4_F
    ps = make_payloads(dev, E, H, F)
    expect = {name: nbytes for name, _, nbytes in payload_sizes(H, F)}
    ok_all = True
    for name, t in ps.items():
        m = _meta_row(t)
        row = m["stride0_bytes"]
        per_expert = m["numel_bytes"] // E
        unpadded = (row == per_expert)
        covered = m["nbytes"] >= E * row + m["soff"]
        ok = (m["esize"] == 1 and unpadded and covered)
        ok_all &= ok
        print(f"       {name:<10} shape={m['shape']} stride={m['stride']} "
              f"dtype={m['dtype']} esize={m['esize']} fmt={m['format']} "
              f"contig={m['contig']} soff={m['soff']}")
        print(f"                 逐专家步长={row} B (期望 {expect[name]}) "
              f"numel/E={per_expert} storage={m['nbytes']} B "
              f"无padding={unpadded} 覆盖={covered}")
        # t[1] - t[0] 的实测步长（跨专家指针差）
        step = t[1:].data_ptr() - t[0:1].data_ptr()
        print(f"                 t[1].data_ptr()-t[0].data_ptr()={step} B")
        ok_all &= (step == row)
    report("M1", ok_all,
           f"四个 payload：element_size==1、逐专家无 padding、"
           f"t[1]-t[0]==stride(0)*esize 全通过（E={E}, H={H}, F={F}）")


def check_M2(dev: str, args) -> None:
    ps = make_payloads(dev, 2, 512, 256)  # 小几何，够验证「能读」
    ok = True
    for name, t in ps.items():
        try:
            fb = flat_bytes(t)
            clone = fb.clone()
            torch.npu.synchronize()
            same = bool(torch.equal(fb, clone))
            ok &= same
            print(f"       {name:<10} flat=[{fb.numel()}] uint8 可读，clone 一致={same}")
        except Exception as exc:  # noqa: BLE001 - 诊断脚本要打印原始异常
            ok = False
            print(f"       {name:<10} 失败：{type(exc).__name__}: {exc}")
    report("M2", ok, "非连续镜像视图 → flat [nbytes] uint8 可读且可 clone")


def check_M3(dev: str, args) -> None:
    ps = make_payloads(dev, 2, 512, 256)
    ok = True
    for name, t in ps.items():
        fb = flat_bytes(t)
        home = torch.empty(fb.numel(), dtype=torch.uint8, device=dev)
        home.copy_(fb)  # 生产里就是 home_xx.copy_(w13_src) 的逐字节版
        torch.npu.synchronize()
        diff = int((home != fb).sum().item())
        ok &= diff == 0
        print(f"       {name:<10} 镜像 {fb.numel()} B，不等字节数={diff}")
    report("M3", ok, "home 载体 ← 权重 storage 的逐字节拷贝无损（绕开 NZ 互拷）")


# ---------------------------------------------------------------------------
# M4：alias 数值透明性（生死判据）
# ---------------------------------------------------------------------------
def _gmm1_w4a8mxp(x_fp8, weight, weight_scale, pertoken_scale, group_list_cumsum):
    """A5 的 W4A8MXFP gmm1 —— 逐字复刻 device_op.py:1362-1377 的调用。"""
    return torch_npu.npu_grouped_matmul(
        x=[x_fp8],
        weight=[weight],
        scale=None,
        antiquant_scale=[weight_scale],
        scale_dtype=None,
        per_token_scale=[pertoken_scale],
        per_token_scale_dtype=E8M0,
        split_item=2,
        group_type=0,
        group_list=group_list_cumsum,
        x_dtype=FP8,
        weight_dtype=F4X2,
        output_dtype=torch.bfloat16,
    )[0]


def build_alias_payload(base_va: int, epn: int, B: int, ref: torch.Tensor,
                        dev: str, is_nz: bool, npu_format: int) -> torch.Tensor:
    """在 ``base_va`` 上把 [epn] home 行 + [B] slot 行拼成与 ref 同形/同 stride 的
    权重视图（Phase 1 ``_packed_alias`` 的原型）。

    - NZ payload（w13/w2）：按 **cast 时的形状**（ref 的 transpose 前身）建
      带 format 的张量再 ``transpose(1,2)``，与生产路径逐字段一致；
    - ND payload（两个 scale）：照抄 ref 的 shape/stride/storage_offset + format。

    ``npu_format`` 由调用方给（实测值）—— format 值的具体语义是 A5 内部枚举，
    不要硬编码 29（M1 实测真实 payload 是 50）。
    """
    row_bytes = ref.stride(0) * ref.element_size()
    n = epn + B
    if is_nz:
        cast_shape = (n, ref.shape[2], ref.shape[1])
        cast = sym_tensor_from_ptr(base_va, cast_shape, ref.dtype, dev,
                                   nbytes=n * row_bytes, npu_format=npu_format)
        alias = cast.transpose(1, 2)
    else:
        stride = (ref.stride(0),) + tuple(ref.stride()[1:])
        alias = sym_tensor_from_ptr(
            base_va, (n,) + tuple(ref.shape[1:]), ref.dtype, dev,
            stride=stride, storage_offset=ref.storage_offset(),
            nbytes=n * row_bytes, npu_format=npu_format)
    assert alias.shape == (n,) + tuple(ref.shape[1:]), alias.shape
    assert alias.stride() == (ref.stride(0),) + tuple(ref.stride()[1:]), alias.stride()
    return alias


def check_M4(dev: str, args) -> None:
    # 小几何（非方形，便于暴露 transpose/stride 错误）：H=384=12*32, F=64 -> 2F=128
    E, H, F, B, T = 8, 384, 64, 2, 96
    ps = make_payloads(dev, E, H, F, seed=7)
    ref_w13, ref_s13 = ps["w13"], ps["w13_scale"]
    row_w = ref_w13.stride(0) * ref_w13.element_size()
    row_s = ref_s13.stride(0) * ref_s13.element_size()
    fmt_w = int(torch_npu.get_npu_format(ref_w13))
    fmt_s = int(torch_npu.get_npu_format(ref_s13))

    # 打包 VA：home(epn=E 行) + slot(B 行)，两块各自的 scale 同理
    w_buf = torch.zeros((E + B) * row_w, dtype=torch.uint8, device=dev)
    s_buf = torch.zeros((E + B) * row_s, dtype=torch.uint8, device=dev)

    def pack(buf, ref):
        fb = flat_bytes(ref)
        view = sym_tensor_from_ptr(buf.data_ptr(), (fb.numel(),), torch.uint8, dev)
        view.copy_(fb)  # home 段
        torch.npu.synchronize()
        return fb

    fb_w = pack(w_buf, ref_w13)
    pack(s_buf, ref_s13)

    # 自检：alias 的 home 段字节必须与真实权重 storage 一致
    fb_alias = sym_tensor_from_ptr(w_buf.data_ptr(), (fb_w.numel(),),
                                   torch.uint8, dev)
    bytes_eq = bool(torch.equal(fb_w, fb_alias))

    # x 激活：A5 动态 MX 量化（device_op.py:1317-1320 的等价）
    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev) * 0.5
    x_fp8, xs = torch_npu.npu_dynamic_mx_quant(x, dst_type=FP8)
    # A5DeviceAdaptor.maybe_normalize_mxfp_scale_layout（device_op.py:1229-1234）：
    # 只在 2 维时把 [T, k] 变 [T, k/2, 2]；实测 npu_dynamic_mx_quant 直接给 3 维
    if xs.ndim == 2:
        xs = xs.reshape(xs.shape[0], xs.shape[1] // 2, 2)
    print(f"       x_fp8={tuple(x_fp8.shape)} {x_fp8.dtype} "
          f"pertoken_scale={tuple(xs.shape)} {xs.dtype}")

    counts = torch.tensor([13, 11, 12, 10, 9, 14, 13, 14], dtype=torch.int64, device=dev)
    gl_ref = counts.cumsum(0)
    gl_alias = torch.cat([counts, torch.zeros(B, dtype=torch.int64, device=dev)]).cumsum(0)

    out_ref = _gmm1_w4a8mxp(x_fp8, ref_w13, ref_s13, xs, gl_ref)

    # 逐个 format 变体试：实测 format（期望 PASS）与「教科书 NZ=29」（预期 FAIL）
    variants = [(f"实测 fmt={fmt_w}", fmt_w), ("NZ 常量 29", 29)]
    ok_any = False
    for label, fmt in variants:
        try:
            aw = build_alias_payload(w_buf.data_ptr(), E, B, ref_w13, dev, True, fmt)
            as_ = build_alias_payload(s_buf.data_ptr(), E, B, ref_s13, dev, False, fmt_s)
            out_alias = _gmm1_w4a8mxp(x_fp8, aw, as_, xs, gl_alias)
            torch.npu.synchronize()
            eq = bool(torch.equal(out_ref, out_alias))
            maxdiff = float((out_ref.float() - out_alias.float()).abs().max())
            fin = bool(torch.isfinite(out_alias.float()).all())
            ok_any |= eq
            print(f"       {label:<20} alias fmt={int(torch_npu.get_npu_format(aw))} "
                  f"shape={tuple(aw.shape)} 有限={fin} bit-identical={eq} "
                  f"max|diff|={maxdiff:.4g}")
        except Exception as exc:  # noqa: BLE001 - 诊断脚本要打印原始异常
            print(f"       {label:<20} 失败：{type(exc).__name__}: {exc}")

    finite = bool(torch.isfinite(out_ref.float()).all())
    print(f"       out_ref shape={tuple(out_ref.shape)} 有限={finite} "
          f"home 段字节一致={bytes_eq}（w13 fmt={fmt_w}, scale fmt={fmt_s}）")
    report("M4", ok_any and finite and bytes_eq,
           f"自建 format={fmt_w} alias + 照抄 stride 的 scale alias 上跑 W4A8MXFP "
           f"gmm1，与真实权重 bit-identical")


# ---------------------------------------------------------------------------
# S1：产品代码路径（moonep_balance._LayerWeightState）
# ---------------------------------------------------------------------------
def check_S1(dev: str, args) -> None:
    """在**真实产品代码**上建一层量化 state，并验证镜像与 alias。

    M0-M6 验的是前提（VMM/NZ/gmm 的语义）；这一项验的是真代码：_infer_F /
    _payload_specs / publish / _packed_alias / compute_weights / prefetch_views。
    显存 ~130 MiB、不需要 SHMEM 运行时、不需要 8 卡 —— 走的是与引擎里逐字节
    相同的构造路径。

    E=8（rank0 持全部专家）是**能让四个 payload 的 epn*row 都对齐 VMM gran 的
    最小专家数**：最小的 payload（w2_scale，256 KiB）在 E=8 时是 2 MiB = gran。
    E 更小就会落到 buffer_c 的 lcm 回退布局（slot_start > E 直接抛）—— 真实配置
    的 epn（R=8 时 32、R=32 时 8）都满足这条。
    """
    import types

    from vllm_ascend.ops.fused_moe import moonep_balance as mb

    E, H, F, B, T = 8, DSV4_H, DSV4_F, 2, 96
    ps = make_payloads(dev, E, H, F, seed=11)
    layer = types.SimpleNamespace(
        w13_weight=ps["w13"], w2_weight=ps["w2"],
        w13_weight_scale=ps["w13_scale"], w2_weight_scale=ps["w2_scale"])

    dev_id = int(torch.npu.current_device())
    state = mb._LayerWeightState(layer, E, H, rank=0, dev=dev_id)
    # state.B 默认取 epn（库的 plan 要求 B == E/R），这里跟着它走
    B = state.B
    inferred_ok = (state.F == F and state.epn == E and B == E)
    print(f"       推断层几何：F={state.F}（期望 {F}）epn={state.epn} "
          f"B={state.B}  一致={inferred_ok}")
    spec_names = [s.name for s in state._specs]
    print(f"       payload 顺序={spec_names}")
    for spec in state._specs:
        rows, cols = spec.carrier
        print(f"       {spec.name:<10} row={spec.row_bytes:>9} B "
              f"carrier=({rows}, {cols}) fmt={spec.npu_format} "
              f"NZ={spec.is_nz}")

    # 1) 镜像：home 的字节必须与参考权重逐字节相同
    state.publish()
    mirror_bad = []
    for spec in state._specs:
        src = flat_bytes(spec.ref)[: spec.nbytes]
        dst = flat_bytes(state.homes[spec.name])[: spec.nbytes]
        if not bool(torch.equal(src, dst)):
            mirror_bad.append(spec.name)
    torch.npu.synchronize()
    print(f"       publish 后逐字节不一致的 payload：{mirror_bad or '无'}")

    # 2) prefetch 视图：四个 [E+B, R, C] bf16，且挂的 source VA 指向 home/slot
    views = state.prefetch_views()
    views_ok = len(views) == len(state._specs) and all(
        v.dtype == torch.bfloat16 and v.ndim == 3 and int(v.shape[0]) == E + B
        and v.is_contiguous() for v in views)
    va_ok = all(
        int(getattr(v, "_moonep_owner_source_va"))
        == state.homes[spec.name].data_ptr()
        and int(getattr(v, "_moonep_slot_source_va"))
        == state.slots[spec.name].data_ptr()
        for v, spec in zip(views, state._specs))
    print(f"       投影描述符 {len(views)} 个：形状/类型/连续={views_ok} "
          f"source VA 指向 home/slot={va_ok}")

    # 3) alias 数值透明：走产品 compute_weights()，与真实权重比 bit-identity
    weights = state.compute_weights()
    row_w = int(ps["w13"].stride(0)) * int(ps["w13"].element_size())
    row_s = int(ps["w13_scale"].stride(0)) * int(ps["w13_scale"].element_size())
    alias_shape_ok = (
        tuple(weights.w1.shape) == (E + B,) + tuple(ps["w13"].shape[1:])
        and tuple(weights.w2.shape) == (E + B,) + tuple(ps["w2"].shape[1:])
        and int(weights.w1.stride(0)) == row_w
        and int(weights.w2.stride(0)) == int(ps["w2"].stride(0)) * int(ps["w2"].element_size())
        and tuple(weights.w1_scale.shape) == (E + B,) + tuple(ps["w13_scale"].shape[1:])
        and int(weights.w1_scale.stride(0)) == row_s)
    print(f"       alias 形状/逐专家步长对齐参考权重={alias_shape_ok} "
          f"w1={tuple(weights.w1.shape)} stride0={int(weights.w1.stride(0))} "
          f"w1_scale={tuple(weights.w1_scale.shape)}")

    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev) * 0.5
    x_fp8, xs = torch_npu.npu_dynamic_mx_quant(x, dst_type=FP8)
    if xs.ndim == 2:
        xs = xs.reshape(xs.shape[0], xs.shape[1] // 2, 2)
    counts = torch.zeros(E, dtype=torch.int64, device=dev)
    counts[0] = T - 1
    counts[1] = 1
    gl_ref = counts.cumsum(0)
    gl_alias = torch.cat([counts, torch.zeros(B, dtype=torch.int64,
                                              device=dev)]).cumsum(0)
    out_ref = _gmm1_w4a8mxp(x_fp8, ps["w13"], ps["w13_scale"], xs, gl_ref)
    out_alias = _gmm1_w4a8mxp(x_fp8, weights.w1, weights.w1_scale, xs, gl_alias)
    torch.npu.synchronize()
    bit_eq = bool(torch.equal(out_ref, out_alias))
    maxdiff = float((out_ref.float() - out_alias.float()).abs().max())
    finite = bool(torch.isfinite(out_alias.float()).all())
    print(f"       产品 alias 上 gmm1：有限={finite} bit-identical={bit_eq} "
          f"max|diff|={maxdiff:.4g}")
    report("S1", inferred_ok and not mirror_bad and views_ok and va_ok
           and alias_shape_ok and bit_eq and finite,
           "产品代码 _LayerWeightState：量化下四 payload 的镜像/投影/alias 全对，"
           "alias 上 gmm1 与真实权重 bit-identical")


# ---------------------------------------------------------------------------
# S2：产品入口链上的两个**封装算子**（融合 gmm1+swiglu / gmm2）
# ---------------------------------------------------------------------------
class _NoPrefetchBuffer:
    """``run_slot_forward`` 的 buffer 替身：不 init shmem、不搬字节。

    S2 手填进 slot 段的字节就是 prefetch 本会搬来的内容（单 rank，专家索引即
    身份），所以这里只需要一个空实现 —— S2 要覆的是 ``run_slot_forward`` 的
    后三步（build_mlp_compute_input → 换 weights → unified_apply_mlp）。
    搬字节那一环是纯拷贝，由 M3 与库侧负责（见文档 §关键结论 1）。
    """

    def prefetch_projections(self, **_kwargs) -> None:
        return None


def _patch_extra_ctx():
    """把各模块里 import 走的 ``_EXTRA_CTX`` 换成 fake 上下文。

    spike 里没有真实 forward context：真代理的 ``__getattr__`` 会落到
    ``vllm.forward_context.get_forward_context()`` 上抛异常，而
    ``quant_apply_mlp`` 一进来就读 ``_EXTRA_CTX.moe_comm_type``（:154，判断 MC2）。
    MXFP 路径的分支条件是 ``use_mxfp_quant``（:156 的 ``and not use_mxfp_quant``
    让 MC2 分支对 MXFP 不可达），所以 comm type 取 None 不改变走哪条路；
    ``capturing=False`` 是 harness 惯例。整体扫一遍而不是逐个点名 —— 少漏一个
    模块就少一次 AttributeError。
    """
    import types

    from vllm_ascend.ascend_forward_context import _EXTRA_CTX as real
    fake = types.SimpleNamespace(capturing=False, moe_comm_type=None)
    patched = []
    for mod in list(sys.modules.values()):
        if getattr(mod, "_EXTRA_CTX", None) is real:
            mod._EXTRA_CTX = fake  # noqa: SLF001
            patched.append(getattr(mod, "__name__", "?"))
    return patched


def _bit_equal(a, b) -> bool:
    """逐比特相等（1 字节浮点走 uint8 视图，绕开比较算子对 e8m0 的支持差异）。"""
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return a is b
    if tuple(a.shape) != tuple(b.shape) or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    try:
        return bool(torch.equal(a.contiguous().view(torch.uint8),
                                b.contiguous().view(torch.uint8)))
    except RuntimeError:
        return bool(torch.equal(a, b))


def _max_diff(a, b) -> float:
    if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
        return float("nan")
    if a.numel() == 0 or tuple(a.shape) != tuple(b.shape):
        return float("nan")
    try:
        d = (a.float() - b.float()).abs()
    except (RuntimeError, TypeError):
        d = (a.contiguous().view(torch.uint8).float()
             - b.contiguous().view(torch.uint8).float()).abs()
    return float(d.max())


def _instrument_device_op(recs: dict, cur: dict):
    """钩住 ``quant_apply_mlp`` 真正调的两个封装算子，记录 kwargs 与中间张量。

    为什么不自己抄 kwargs 去调裸算子：**要验的就是产品代码怎么调**（kwargs 抄错
    就变成验另一样东西）。挂钩只观测、不改行为：既证明走的是 MXFP 融合分支，
    又能把 alias 与真实权重的差异定位到具体算子（swiglu 出参 / gmm2 入参），
    而不是只看最终输出。两个方法在 DeviceOperator 上都是 staticmethod，
    取出来是普通函数，包一层再塞回 staticmethod 即可。
    """
    from vllm_ascend.device.device_op import DeviceOperator

    orig1 = DeviceOperator.npu_grouped_matmul_swiglu_quant
    orig2 = DeviceOperator.npu_grouped_matmul_gmm2

    def gmm1(**kwargs):
        out = orig1(**kwargs)
        recs[cur["k"]].update(swiglu=out, gmm1_kwargs=kwargs)
        return out

    def gmm2(**kwargs):
        out = orig2(**kwargs)
        recs[cur["k"]].update(gmm2=out, gmm2_kwargs=kwargs,
                              gmm2_x=kwargs.get("hidden_states"))
        return out

    DeviceOperator.npu_grouped_matmul_swiglu_quant = staticmethod(gmm1)
    DeviceOperator.npu_grouped_matmul_gmm2 = staticmethod(gmm2)
    return orig1, orig2


def check_S2(dev: str, args) -> None:
    """引擎真正调的那条链上，alias 与真实权重是否**逐比特**同结果。

    与 S1 的分工：S1 证明「自建 NZ 元数据 == CANN npu_format_cast」（元数据层，
    用裸 gmm1）；S2 证明「产品入口链 + 产品 compute_weights 的 alias」在
    ``run_slot_forward`` 本体所调的两个封装算子上不引入任何位差（算子层）：
    激活的 MX 动态量化 → 融合 gmm1+swiglu
    （``DeviceOperator.npu_grouped_matmul_swiglu_quant`` → ``..._v2``）→ gmm2
    （``DeviceOperator.npu_grouped_matmul_gmm2`` → ``npu_grouped_matmul``
    + ``antiquant_scale``），这两个才是引擎实际发的算子。

    行序技巧：单专家布局下，把 T 行切成 home 段（T_a 行，走真实权重的 0 号专家）
    与 slot 段（T_b 行，走 alias 的槽位 0 组），ref 侧把两段并进 0 号专家。这样
    两次调用的每一行语义相同、顺序相同，可以整张 ``torch.equal`` 而不需要重排序。
    slot 第 j 行填专家 j 的字节 ⇒ 只有「槽位行 ↔ group_list 里 E+j 组」的映射正确，
    slot 段才会与 ref 对得上（映射错位、段基址偏移都会直接暴露）。

    显存 ~260 MiB（E=8 时四个 payload 的 home+slot 各 ~102 MiB），单卡、不需要
    shmem/8 卡。
    """
    import types

    from vllm_ascend.ops.fused_moe import moonep_balance as mb
    from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
    from vllm_ascend.ops.fused_moe.moe_runtime_args import (
        build_fused_experts_input,
        build_mlp_compute_input,
    )
    from vllm_ascend.quantization.quant_type import QuantType

    patched = _patch_extra_ctx()
    print(f"       _EXTRA_CTX 已替换为 fake：{patched}")

    # 引擎里 torch.ops._C_ascend 是 device_allocator/camem.py 顺手 import 进来的
    # （A5 上 utils.enable_custom_op() 反而被 :426 的 A5 门禁挡掉，.so 不由它加载）。
    # W4A8MXFP 的融合 gmm1 后半段正是 torch.ops._C_ascend.npu_swiglu_group_quant
    # （device_op.py:1379，DSv4 的 swiglu_limit 只能走它），spike 显式补上。
    # （aclnn 侧的 vendor 解析靠模块顶部的 ASCEND_CUSTOM_OPP_PATH，见那里注释。）
    from vllm_ascend.utils import bootstrap_custom_op_env
    bootstrap_custom_op_env()
    import vllm_ascend.vllm_ascend_C  # type: ignore  # noqa: F401
    if not hasattr(torch.ops._C_ascend, "npu_swiglu_group_quant"):
        report("S2", None,
               "vllm_ascend_C 已 import 但 _C_ascend.npu_swiglu_group_quant 仍缺失"
               "——A5 上这个算子没编进来，融合 gmm1 走不通")
        return

    # E=8 是最小的「四个 payload 都对齐 VMM gran」的专家数（同 S1），
    # rank=0/R=1 ⇒ epn=E、B=epn。
    E, H, F = 8, DSV4_H, DSV4_F
    T_a, T_b = 48, 16
    T = T_a + T_b
    ps = make_payloads(dev, E, H, F, seed=13)
    layer = types.SimpleNamespace(
        w13_weight=ps["w13"], w2_weight=ps["w2"],
        w13_weight_scale=ps["w13_scale"], w2_weight_scale=ps["w2_scale"])

    dev_id = int(torch.npu.current_device())
    state = mb._LayerWeightState(layer, E, H, rank=0, dev=dev_id)
    B = state.B
    state.publish()

    # slot 段第 j 行 ← 专家 (j % E) 的字节（= prefetch 单 rank 下本会搬来的内容）
    for spec in state._specs:
        slot_flat = mb._flat_bytes(state.slots[spec.name], spec.nbytes)
        ref_flat = mb._flat_bytes(spec.ref, spec.nbytes)
        for j in range(B):
            src = ref_flat[(j % E) * spec.row_bytes:(j % E + 1) * spec.row_bytes]
            slot_flat[j * spec.row_bytes:(j + 1) * spec.row_bytes].copy_(src)
    torch.npu.synchronize()
    print(f"       E={E} epn={state.epn} B={B} payload={[s.name for s in state._specs]}"
          f"  slot 行已填入对应专家字节")

    x = torch.randn(T, H, dtype=torch.bfloat16, device=dev) * 0.5
    topk_weights = torch.full((T, 1), 1.0, dtype=torch.float32, device=dev)
    topk_ids = torch.zeros((T, 1), dtype=torch.int32, device=dev)

    # 与 w4a8_mxfp4.py:207-230（产品侧唯一构造 W4A8MXFP fused_experts_input 的地方）
    # 逐项对齐：quant_type / 四个 mxfp_* 参数 / w1_scale,w2_scale / swiglu_limit
    fei = build_fused_experts_input(
        hidden_states=x,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1=ps["w13"],
        w2=ps["w2"],
        quant_type=QuantType.W4A8MXFP,
        dynamic_eplb=False,
        mxfp_act_quant_type=FP8,
        mxfp_weight_quant_type=F4X2,
        mxfp_scale_dtype=E8M0,
        mxfp_per_token_scale_dtype=E8M0,
        mxfp_use_bf16=True,
        w1_scale=ps["w13_scale"],
        w2_scale=ps["w2_scale"],
        swiglu_limit=DSV4_SWIGLU_LIMIT,
    )

    def _disp(hidden, group_counts):
        # 与 moonep_comm 的 slot dispatch 输出同形：group_list_type=1（计数）
        return types.SimpleNamespace(
            hidden_states=hidden, group_list=group_counts, group_list_type=1,
            dynamic_scale=None, topk_scales=None,
            combine_metadata=types.SimpleNamespace(expanded_row_idx=None))

    cnt_ref = torch.zeros(E, dtype=torch.int64, device=dev)
    cnt_ref[0] = T  # home 段 + slot 段都归 0 号专家
    cnt_alias = torch.zeros(E, dtype=torch.int64, device=dev)
    cnt_alias[0] = T_a
    slot_cnt = torch.zeros(B, dtype=torch.int64, device=dev)
    slot_cnt[0] = T_b  # 槽位 0 组拿到 slot 段
    gl_alias = torch.cat([cnt_alias, slot_cnt])

    mlp_ref = build_mlp_compute_input(
        fused_experts_input=fei,
        token_dispatch_output=_disp(x.clone(), cnt_ref),
        use_fusion_ops=False)

    recs = {"ref": {}, "alias": {}}
    cur = {"k": "ref"}
    orig1, orig2 = _instrument_device_op(recs, cur)
    try:
        # ref：真实权重，走同一条 unified_apply_mlp
        out_ref, _ = unified_apply_mlp(mlp_compute_input=mlp_ref)
        # alias：run_slot_forward 本体（只有 prefetch 换成空实现）
        cur["k"] = "alias"
        out_alias = mb.run_slot_forward(
            state=state, buffer=_NoPrefetchBuffer(), plan=None,
            token_dispatch_output=_disp(x.clone(), gl_alias),
            fused_experts_input=fei, use_fusion_ops=False)
    finally:
        from vllm_ascend.device.device_op import DeviceOperator
        DeviceOperator.npu_grouped_matmul_swiglu_quant = staticmethod(orig1)
        DeviceOperator.npu_grouped_matmul_gmm2 = staticmethod(orig2)
    torch.npu.synchronize()

    kw1, kw2 = recs["alias"].get("gmm1_kwargs"), recs["alias"].get("gmm2_kwargs")
    if kw1 is None or kw2 is None:
        report("S2", False,
               f"两个封装算子没被调到（gmm1={kw1 is not None} gmm2={kw2 is not None}）"
               "—— 说明走的是别的分支，本次没验到要验的东西")
        return
    mxfp_ok = (kw1.get("use_mxfp_quant") is True
               and kw1.get("mxfp_quant_dtype") == QuantType.W4A8MXFP
               and kw1.get("weight_quant_type") == F4X2
               and kw1.get("act_quant_type") == FP8
               and isinstance(kw1.get("weight"), torch.Tensor)
               and kw2.get("use_mxfp_quant") is True
               and isinstance(kw2.get("weight_scale"), torch.Tensor))
    n_ref = int(recs["ref"]["gmm2_kwargs"]["group_list"].numel())
    n_alias = int(recs["alias"]["gmm2_kwargs"]["group_list"].numel())
    gl_ok = (n_ref == E and n_alias == E + B)
    print(f"       gmm1 走 MXFP 融合分支={mxfp_ok}（use_mxfp_quant=True、"
          f"mxfp_quant_dtype=W4A8MXFP、weight_quant_type=fp4x2、单张量权重）")
    print(f"       组数：ref={n_ref}（E={E}）alias={n_alias}（E+B={E + B}）"
          f" 一致={gl_ok}")

    sw_ref, sw_al = recs["ref"]["swiglu"], recs["alias"]["swiglu"]
    checks = [
        ("融合 gmm1 的 swiglu 输出", sw_ref[0], sw_al[0]),
        ("融合 gmm1 的 out_scale", sw_ref[1], sw_al[1]),
        ("gmm2 的输入", recs["ref"]["gmm2_x"], recs["alias"]["gmm2_x"]),
        ("最终输出", out_ref, out_alias),
    ]
    ok = True
    for tag, a, b in checks:
        eq = _bit_equal(a, b)
        ok = ok and eq
        print(f"       {tag:<24} bit-identical={eq}  max|diff|={_max_diff(a, b):.4g}"
              f"  shape={tuple(a.shape) if isinstance(a, torch.Tensor) else a}")
    finite = bool(torch.isfinite(out_alias.float()).all())
    if not ok and isinstance(out_ref, torch.Tensor) and isinstance(out_alias, torch.Tensor):
        # 差异落在哪一段：home 段（[0,T_a)）说明镜像/元数据不对，
        # slot 段（[T_a,T)）说明槽位映射或段基址不对
        diff_rows = (out_ref.float() != out_alias.float()).any(dim=-1).nonzero().flatten()
        home = int((diff_rows < T_a).sum())
        print(f"       不一致的行：{int(diff_rows.numel())} 行"
              f"（home 段 {home} 行 / slot 段 {int(diff_rows.numel()) - home} 行）"
              f" 前几个={diff_rows[:8].tolist()}")
    report("S2", ok and mxfp_ok and gl_ok and finite,
           f"产品入口链（run_slot_forward 本体 + 两个封装算子）：alias 与真实权重"
           f"逐比特一致={ok}，走的确实是 MXFP 融合分支={mxfp_ok}，"
           f"alias 组数含 {B} 个槽位={gl_ok}，输出有限={finite}")


# ---------------------------------------------------------------------------
# M6：VMM 账（gran/lcm 约束 + VA 总量）
# ---------------------------------------------------------------------------
def _m6_import():
    try:
        from ascend_moonep import buffer_c as bc
    except Exception as exc:  # noqa: BLE001 - 缺依赖就 SKIP
        return None, f"{type(exc).__name__}: {exc}"
    return bc, None


def check_M6(dev: str, args) -> None:
    bc, err = _m6_import()
    if bc is None:
        report("M6", None, f"跳过：ascend_moonep.buffer_c 不可导入（{err}）；"
                           f"先 export PYTHONPATH=<repo>/third_party/ascend_moonep "
                           f"与 ASCEND_MOONEP_BUILD_DIR")
        return
    epn = int(args.epn)
    dev_id = int(dev.split(":")[-1])
    H, F, E = DSV4_H, DSV4_F, DSV4_E
    gran = int(bc.acl_vmm_granularity(dev_id))
    print(f"       VMM granularity={gran} B，epn={epn}，E={E}，层数={DSV4_LAYERS}")

    # --- M6a：单层四件套（home+slot+full+alias）真实几何原型 ---
    ok_a, va_layer, phys_layer = True, 0, 0
    for name, _, row_bytes in payload_sizes(H, F):
        try:
            r, c = carrier_shape(row_bytes)
            home, _, _ = bc.create_vmm_physical([epn, r, c], torch.bfloat16, dev_id)
            slot, _, _ = bc.create_vmm_physical([epn, r, c], torch.bfloat16, dev_id)
            full = bc.create_full_projection_tensor(home, slot, E, epn, 0, dev_id)
            bc.attach_local_source_vas(full, home.data_ptr(), slot.data_ptr())
            alias_va = int(bc.acl_vmm_reserve((2 * epn) * row_bytes))
            bc.acl_vmm_release(alias_va)
            phys = int(home._vmm_allocation.size) + int(slot._vmm_allocation.size)
            # full 投影的保留 VA：与 create_full_projection_tensor 同式
            # （buffer_c.py:692-695）——home 段在位、slot 段从 ALIGN_UP(E*row, gran) 起
            slot_off = _align_up(E * row_bytes, gran)
            va_total = _align_up(
                max(int(home._vmm_allocation.size),
                    slot_off + int(slot._vmm_allocation.size)), gran)
            va_layer += va_total + (2 * epn) * row_bytes
            phys_layer += phys
            print(f"       M6a {name:<10} carrier=[{epn},{r},{c}] row={row_bytes} B  "
                  f"gran 对齐: epn*row%gran={(epn * row_bytes) % gran}  "
                  f"lcm/gran 比={_lcm_ratio(gran, row_bytes)}  "
                  f"home+slot 物理={phys / 2**20:.1f} MiB  full VA={va_total / 2**20:.1f} MiB")
            full._projection.destroy()
            home._vmm_allocation.destroy()
            slot._vmm_allocation.destroy()
        except Exception as exc:  # noqa: BLE001
            ok_a = False
            print(f"       M6a {name:<10} 失败：{type(exc).__name__}: {exc}")
    report("M6a", ok_a,
           f"真实几何（row 8MiB/4MiB/512KiB/256KiB）下单层 home+slot+full+alias "
           f"全部建成；单层物理={phys_layer / 2**20:.1f} MiB，VA={va_layer / 2**30:.2f} GiB")

    # --- M6b：全模型 VA 总量（只 reserve，不占物理）---
    alias_total = phys_home_total = full_total = 0
    for name, _, row_bytes in payload_sizes(H, F):
        alias_total += DSV4_LAYERS * (2 * epn) * row_bytes
        phys_home_total += DSV4_LAYERS * epn * row_bytes
        # full 投影：home 段在位 + slot 段从 align_up(E*row, gran) 起
        slot_off = ((E * row_bytes + gran - 1) // gran) * gran
        slot_size = ((epn * row_bytes + gran - 1) // gran) * gran
        full_total += DSV4_LAYERS * _align_up(max(slot_off + slot_size, gran), gran)
    total_va = alias_total + full_total
    print(f"       单卡 R={E // epn}：home 镜像物理={phys_home_total / 2**30:.1f} GiB，"
          f"alias VA={alias_total / 2**30:.1f} GiB，full VA={full_total / 2**30:.1f} GiB，"
          f"合计 VA={total_va / 2**30:.1f} GiB")
    ok_b = True
    try:
        va = int(bc.acl_vmm_reserve(total_va))
        bc.acl_vmm_release(va)
        print(f"       一次性 reserve({total_va / 2**30:.1f} GiB) → OK")
    except Exception as exc:  # noqa: BLE001
        ok_b = False
        print(f"       一次性 reserve({total_va / 2**30:.1f} GiB) 失败："
              f"{type(exc).__name__}: {exc}")
    report("M6b", ok_b, f"全模型（{DSV4_LAYERS} 层）VA 合计 {total_va / 2**30:.1f} GiB "
                        f"可一次性 reserve")


def _align_up(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def _lcm_ratio(gran: int, row_bytes: int) -> int:
    import math
    return math.lcm(gran, row_bytes) // row_bytes


# ---------------------------------------------------------------------------
# 引擎日志自检（M7）
# ---------------------------------------------------------------------------
def check_stats(path: str) -> int:
    """解析 MOONEP_SLOT_STATS=1 的退出打点行。

    判据：``total_slot_rows > 0`` 且 ``calls_with_slot_rows > 0`` —— 否则
    prefetch 根本没被触发，端到端「跑通」是假通过。
    """
    import re
    pat = re.compile(r"\[moonep-stats\]\s+dispatch_calls=(\d+)\s+"
                     r"calls_with_slot_rows=(\d+)\s+total_slot_rows=(\d+)\s+"
                     r"total_prefetched_expert_slots=(\d+)")
    hits = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                hits.append(tuple(int(g) for g in m.groups()))
    if not hits:
        print(f"FAIL  {path} 里没有 [moonep-stats] 行；"
              f"确认启动时带了 MOONEP_SLOT_MODE=1 MOONEP_SLOT_STATS=1")
        return 1
    calls, with_slots, slot_rows, experts = hits[-1]
    ok = slot_rows > 0 and with_slots > 0
    print(f"{'PASS' if ok else 'FAIL'}  {path}: dispatch_calls={calls} "
          f"calls_with_slot_rows={with_slots} total_slot_rows={slot_rows} "
          f"prefetched_expert_slots={experts}（{len(hits)} 个 rank 报了打点）")
    return 0 if ok else 1


CHECKS = {
    "M0": check_M0,
    "M1": check_M1,
    "M2": check_M2,
    "M3": check_M3,
    "M4": check_M4,
    "S1": check_S1,
    "S2": check_S2,
    "M6": check_M6,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help=f"只跑指定项，可选 {sorted(CHECKS)}")
    ap.add_argument("--device", default=None, help="npu 设备号，默认当前设备")
    ap.add_argument("--epn", type=int, default=DSV4_E // 8,
                    help="M6 用的每卡专家数，默认 32（DSv4 E=256 / R=8）")
    ap.add_argument("--check-stats", metavar="LOG", default=None,
                    help="只解析引擎日志里的 [moonep-stats] 打点（M7）")
    args = ap.parse_args()

    if args.check_stats:
        return check_stats(args.check_stats)

    if not torch.npu.is_available():
        print("FAIL  没有可用的 NPU")
        return 2
    dev_id = int(args.device) if args.device is not None else int(torch.npu.current_device())
    dev = f"npu:{dev_id}"
    torch.npu.set_device(dev_id)
    print(f"[spike] device={dev} torch={torch.__version__} "
          f"torch_npu={torch_npu.__version__}", flush=True)

    names = args.only if args.only else list(CHECKS)
    for name in names:
        fn = CHECKS.get(name)
        if fn is None:
            print(f"[{name}] SKIP  未知项")
            continue
        try:
            fn(dev, args)
        except Exception as exc:  # noqa: BLE001 - 诊断脚本打印原始异常最有价值
            import traceback
            traceback.print_exc()
            report(name, False, f"抛异常中断：{type(exc).__name__}: {exc}")

    print("\n==== 汇总 ====")
    for tag, ok, msg in _RESULTS:
        state = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
        print(f"  {tag:<4} {state}  {msg}")
    decisive = [r for r in _RESULTS if r[0] == "M4"]
    if decisive and decisive[0][1] is False:
        print("\nM4 不通过 ⇒ 自建 NZ 元数据与 CANN 内部 npu_format_cast 不等价，"
              "Route A′ 需要重估（见 docs/moonep_slot_quant.md 风险 §2）。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
