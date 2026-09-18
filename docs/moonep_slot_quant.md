# MoonEP slot 模式支持量化专家（W4A8MXFP）：改动分析与卡上验证

对象：`/mnt/data/DeepSeek-V4-Flash-0731`（DeepseekV4ForCausalLM，fp8 块量化主干 + MXFP4 专家）。
配套只读诊断脚本：`scripts/spike_slot_quant.py`（本文所有"实测"均为该脚本在 Ascend950PR 上的输出原文，见附录 A）。

---

## 0. 一句话结论

**只改集成层 4 个文件就能端到端，库（`third_party/ascend_moonep`）一行都不用改，`_C.so` 不用重编。**
关键实测：自建 NZ 元数据的 `[epn+B]` alias 在真实 W4A8MXFP gmm1 上与真实权重 **bit-identical**（M4），
真实几何下的镜像/投影/alias 四件套全部建得起来且 VA 可一次性 reserve（M6）。
**唯一的硬约束是卡数：R 必须 ≥ 8**（R=2 时 home 镜像 + 权重 137 GiB > 128 GiB HBM，方案直接不成立）。

> **状态（2026-09-17）**：Phase 1 已落地（§4），bf16 回归 R=2/R=3 PASS、量化产品路径 S1 **与 S2** PASS（§5）；
> 端到端 M7/M8 待 R≥8 空卡。实现做成 `_PayloadSpec` 参数化的**统一实现**（bf16 与量化同一份代码），
> 量化 payload 受 `MOONEP_SLOT_QUANT` 闸门（默认 1）控制。
> S2 覆盖到引擎真正调的两个封装算子（融合 gmm1+swiglu / gmm2），alias 与真实权重四项中间量全 bit-identical；
> 顺带查清 `ASCEND_CUSTOM_OPP_PATH` 的 vendor 解析时机问题（见 §5 末）。

---

## 1. 问题：为什么必须动 slot

量化模型在 moonep 下现在只能走 owner 模式（`MOONEP_SLOT_MODE=0`），而 owner 模式
**把每一行都发给持有该专家的 rank** —— 热点专家会把负载全压在一个 rank 上。
本仓能用的两个自动均衡手段都被 moonep 明令禁止：

| 手段 | 状态 | 证据 |
|---|---|---|
| dynamic EPLB / mix_placement | 构造期 raise | `fused_moe_0_23_0.py:289-293` |
| slot 模式（B 个预取槽） | 量化下 raise | `fused_moe_0_23_0.py:296-299` |

所以"让 moonep 支持 DSv4"= "让 slot 模式的**权重镜像 / RMA 预取 / `[epn+B]` alias** 三件套吃下
MXFP4 + e8m0"。slot 的语义见 `moonep_balance.py:15-40` 的模块 docstring。

难点在于 payload 不是一个而是**四个**：w13、w13_scale、w2、w2_scale。四个都要镜像、
都要能被远端预取、都要进 alias，否则 slot 里的远端专家算子要么拿不到 scale、要么 scale 是错的。

---

## 2. 关键结论（每条都有实测或代码原文）

### K1 库侧的 bf16 硬墙一道都不用碰 —— home/slot/full 可以声明成 bf16「字节载体」

prefetch 链**从不读这三个张量自己的 `data_ptr`**：它只取 `full.shape[1:]`（当 H/Hp）和挂在张量上的
两个 source VA（`api.py:748-803`：`H = full_weight.shape[1]`、`Hp = full_weight.shape[2]`、
`owner_src = getattr(full_weight, "_moonep_owner_source_va")`），然后 kernel 自己按元素宽度寻址：

```cpp
// ascend-moonep/kernels/prefetch.cpp:97-141（原文）
const int64_t elements_per_row = Hp;
const int64_t rows_per_chunk = buffer_bytes / sizeof(bfloat16_t) / elements_per_row;
const int32_t owner = expert / epn;              // epn = ceil(E / ranks)
const int32_t local_expert = expert % epn;
const int64_t source = (local_expert * H + row) * Hp;
const int64_t destination = (slot * H + row) * Hp;
aclshmemx_mte_get_nbi(prefetch_ptr + destination, remote_expert_ptr + source,
                      ub, buffer_bytes, rows * elements_per_row, owner, event_id);
```

逐专家搬运的字节数是 `H * Hp * 2`，**这是一个纯平坦的字节拷贝**（源/目的偏移都按同一元素宽缩放）。
所以只要让 `H * Hp * 2 == 该 payload 每专家的真实字节数`，把载体声明成 bf16 就是完全透明的。
载体形状取 `row_bytes / 2` 的任意因子分解，脚本里的规则是
`carrier_shape()`：`C = min(row_bytes/2, 32768)` 向下取整除，`R = (row_bytes/2)/C`
（`32768` 这个封顶来自 kernel 的 UB 约束 `ub_bytes/2/Hp >= 1`，`prefetch.cpp:100-105`）。

### K2 NZ 格式是**纯元数据**，可以零拷贝贴到任意 VA 上（M0 实测）

`torch_npu._C._construct_storage_from_data_pointer(ptr, device, nbytes)` +
`_construct_NPU_Tensor_From_Storage_And_Metadata(metadata, storage)`，metadata 里多一个
`npu_format` 键即可。M0 实测：`format 2 → 29`，而 **shape / stride / storage / dtype 逐字段不变**，
不触发 transdata、不碰 device。`create_sym_tensor_from_ptr` 本身是**纯 Python**
（`third_party/ascend_moonep/ascend_moonep/buffer_c.py:396-434`，无 C++ 实现），
所以集成层自己拼 metadata 就行 —— Phase 2 若要把 `npu_format` 收进库里，也是零编译的几行。

### K3 真实 payload 的 format 是 **50**（不是 29），dtype 是 **uint8**（M1 实测）

`npu_format_cast(x, 29, customize_dtype=torch.float8_e4m3fn, input_dtype=float4_e2m1fn_x2)`
的返回值实测是 `dtype=uint8, get_npu_format()=50`。**别硬编码 29**。
好消息是 M4 实测 **format 50 与 29 在这个算子上都 bit-identical**，所以两边都能用；
但抄元数据时应当照抄实测值（`torch_npu.get_npu_format(ref)`），不要写常量。

### K4 四个 payload 逐专家都无 padding（M1 实测）

`s[1] - s[0] == stride(0) * element_size() == numel/E` 对四个 payload 全部精确成立，
`untyped_storage().nbytes() == E * stride(0)` 也成立。所以
**"每专家一块连续字节"这个 `_packed_alias` 的核心假设在量化下依然成立**，
`row_bytes := stride(0) * element_size()`（M1 实测：8 MiB / 4 MiB / 512 KiB / 256 KiB）。

### K5 RMA 只有 **src** 必须是对称内存（代码原文）

```
// ascend-moonep/3rdparty/shmem/include/device/gm2gm/engine/shmem_device_mte.h:31-44
@param dst [in] Pointer on local device of the destination data.
@param src [in] Pointer on Symmetric memory of the source data.
```

我方的 home 镜像必须是 VMM（对端来读它）；**slot chunk 必须是 VMM 只是因为 alias 要求连续 VA**，
不是 RMA 的要求。反过来：权重本体在普通设备内存里，RMA 读不到 ⇒ **home 镜像省不掉**
（详见 §5 的可选优化与它为什么不划算）。

### K6 `prefetch_projections` 天然支持 4 个 payload

`api.py:1026-1051` 接受任意长度的投影序列，只断言 `dtype == bf16 and is_contiguous()`、
`ndim == 3`、`shape[0] == E + B`、`shape[1:]` 全为正。四个 bf16 字节载体全部满足 ⇒
`buffer.prefetch_projections(plan=plan, projections=[full_w13, full_w13s, full_w2, full_w2s])`，
一处都不用改库。

### K7 scale 的 alias 必须**照抄 stride**（M1/M4）

两个 scale 处理后的形状是 4 维且非连续（实测 `w13_scale: shape=(E,64,4096,2) stride=(524288,2,128,1) fmt=2`），
而 w13/w2 是 NZ 转置视图（实测 `shape=(E,2048,4096) stride=(8388608,1,2048) fmt=50`）。
两类 payload 的 alias 构造方式不同：

- **NZ（w13/w2）**：按 `npu_format_cast` 的**产出形状** `[n, 2F, H/2]` + 实测 format 建张量，
  再 `.transpose(1, 2)` —— 与生产路径逐字段一致；
- **ND（scale 与 bf16）**：shape `= (n,) + ref.shape[1:]`、stride `= (ref.stride(0),) + ref.stride()[1:]`、
  format 照抄，但 **`storage_offset` 取 0**（不能照抄：参考张量可能是
  `full[rank*epn:(rank+1)*epn]` 这类带偏移的切片，照抄会把整块 alias 平移进 slot 段）。

### K8 R 必须 ≥ 8（见 §3）

---

## 3. 显存与 VA 账（M6 实测，非推导）

每专家每层四个 payload 合计 **12.75 MiB**（8 + 4 + 0.5 + 0.25 MiB）；
E=256、43 层 ⇒ **MoE 权重合计 137 GiB**，**home 镜像与之等大**（K5：只能复制一份）。

| R | epn | 权重/卡 | home 镜像/卡 | slot(B=epn，层间共享) | 合计/卡 | 128 GiB HBM |
|---|---|---|---|---|---|---|
| 2 | 128 | 68.5 GiB | 68.5 GiB | 1.6 GiB | **138.6 GiB** | ✗ 不可能 |
| 4 | 64 | 34.3 GiB | 34.3 GiB | 0.8 GiB | **69.4 GiB** | △ 余 ~59 GiB（131072 MiB/卡）：**功能验证可行**，产量配置仍推荐 R≥8 |
| 8 | 32 | 17.1 GiB | 17.1 GiB | 0.4 GiB | **34.6 GiB** | ✓ **推荐** |
| 16 | 16 | 8.6 GiB | 8.6 GiB | 0.2 GiB | **17.4 GiB** | ✓ |

> **跑测入口是仓内的** `scripts/bench_long.py` / `moe_par_bench.py` / `moe_ab_test.py`
> （`vllm.LLM(...)` + `ASCEND_RT_VISIBLE_DEVICES` / `TP` 环境变量）；单实例 +
> `enable_expert_parallel` 时 **R（EP 组大小）= TP**，所以 R 是选卡时定下来的，跟脚本无关。
> 仓库外 `../scripts/` 下的服务化脚本（`start_prefill.sh`）**当前跑不起来、不作为入口**
> （用户 2026-09-17 明确）。⇒ **要落地这套方案得先凑齐 8 张空卡（TP=8）**。

单卡（R=8）**VA 预算（M6b 实测）**：alias 34.3 GiB + `full` 投影 154.2 GiB = **188.5 GiB**，
`acl_vmm_reserve(188.5 GiB)` 一次性成功。其中 154 GiB 的 `full` 投影 VA 里只有本 rank 的
home 段和 slot 段是真映射，其余是保留未映射区（`buffer_c.py:692-695`）—— 是死 VA，不是死物理。
真要省可以改用"廉价描述符"（用 `home.data_ptr()` 造一个 `[E+B,R,C]` 的纯描述符张量当载体），
**当前实测不需要**。单层四件套的物理量：816 MiB（home 408 + slot 408）。

### 3.1 packed alias 的 VMM 对齐约束（新增，Phase 1 实现后确认）

`_packed_alias` 把 home 段与 slot 段拼进**一段连续 VA**，所以
`epn * row_bytes % granularity(2 MiB) == 0` 是**硬约束**（`moonep_balance._packed_alias` 显式检查）。
四个 payload 里最紧的是 **w2_scale（row = 256 KiB）**：

| payload | row_bytes | `epn*row % 2MiB == 0` 要求 |
|---|---|---|
| w13 | 8 MiB | 任意 epn |
| w2 | 4 MiB | 任意 epn |
| w13_scale | 512 KiB | epn 是 4 的倍数 |
| **w2_scale** | **256 KiB** | **epn 是 8 的倍数** ← 最紧 |

⇒ DSv4（E=256）下 **epn ≡ 0 (mod 8)**，即 R ∈ {1,2,4,8,16,32}；与显存账相交 ⇒ **可行区间 R ∈ {4,8,16,32}**
（R=2 的 138.6 GiB > 128 GiB 不成立；R=4 是"能跑通功能验证"的下限，**不是 8 卡才可跑**）。
另外 UDMA combine 要求 `R == 1 || (R 偶 且 (R ≤ 16 || R % 16 == 0))`
（`ascend-moonep/csrc/ops/moonep_combine_udma.cpp:66-71`，`kMaxGroupWidth=16`）⇒ R=4 / 8 都满足；
MTE combine（`moonep_combine.cpp`）无此门。bf16 只有一个 payload，不受对齐这条约束，R=2/3 的回归才跑得动。

---

## 4. 改动清单

### Phase 1 — 集成层（✅ 已实现 2026-09-17，不重编、不改库）

按用户要求做成**参数化统一实现**：home / slot / full / alias 四件套由 `_PayloadSpec` 驱动，
bf16 只是"一个 payload、`row_bytes = H*2F*2`、自然载体 `(H, 2F)`"的特例，量化是"四个 payload、
`row_bytes = numel/experts`、载体 C 取到 32768"的特例。**同一份代码路径，无分支拷贝。**

**1) `vllm_ascend/ops/fused_moe/fused_moe_0_23_0.py`** —— 硬墙已删

原来的 `if MOONEP_SLOT_MODE==1 and quant_config is not None: raise NotImplementedError(...)`
整段删除，换成中文注释说明**为什么校验后移**：`__init__` 时权重还没加载，逐 payload 的
dtype / 无 padding / `MOONEP_SLOT_QUANT` 闸门只能在权重加载后做，落在
`moonep_balance._LayerWeightState`。顺带删掉因此不再使用的 `import os`。

> **为什么是这个文件而不是 `fused_moe.py`**：见附录 B。

**2) `vllm_ascend/ops/fused_moe/moonep_balance.py`** —— 主体（整体重写）

- `_PayloadSpec`（frozen dataclass）：`name / ref / is_nz / row_bytes / nbytes / carrier=(R,C) / npu_format`；
  **shape、stride、format、dtype 一律运行期从参考张量读，不硬编码**（K3/K7 的教训）
- `_sym_tensor(ptr, shape, dtype, device, *, stride, storage_offset, nbytes, npu_format)`：
  库内 `create_sym_tensor_from_ptr` 的等价物 + `npu_format` 元数据键（K2）
- `_flat_bytes(t, nbytes)`：从 **`data_ptr()`**（含 `storage_offset`）起造 flat uint8 视图，
  镜像一律裸字节 `copy_`（绕开"两个 NZ 张量互拷"，见 M3）
- `_carrier_shape(row_bytes, natural=None)`：bf16 走 `natural`（保持老路径逐字节不变），
  量化取 `C = min(elems, 32768)` 向下整除（`_MAX_CARRIER_ROW_ELEMS = 32768`，即 prefetch kernel
  的 UB 上限 64 KiB）
- `_make_spec`：算 `row_bytes = stride(0) * element_size()`，校验
  `numel/experts * element_size() == row_bytes`（K4 无 padding），校验 payload 不越过 storage
- `_infer_F`：**`F = w13.shape[2] // 2`**（两种布局都把 `2F` 放最后轴），再按 dtype 分头校验
  `shape[1] == H`（bf16）或 `shape[1]*2 == H` 且 scale 的 `shape[2] == 2F`（量化）
- `_payload_specs`：bf16 → `[w13, w2]`；量化 → `[w13, w13_scale, w2, w2_scale]`（K6 四个都要预取），
  受 `MOONEP_SLOT_QUANT` 闸门（默认 1，置 0 则量化层显式报错 = 回滚开关）
- `_SharedSlots.get(...)`：每个 payload 一块 `[B,R,C]` bf16 VMM chunk，key 含 `(name, *carrier)`
- `_LayerWeightState`：字典 `slots/homes/fulls/aliases` 按 payload 名索引；
  `publish()` 逐 payload 裸字节镜像、`prefetch_views()` 返回 `[full...]`（K6）、
  `compute_weights()` 返回 `MoEWeights(w1/w2/w1_scale/w2_scale = aliases[...])`
- `_packed_alias(home, slot, dev, spec)`：NZ → 建 cast 形状（`[n, ref.shape[2], ref.shape[1]]`）+
  `npu_format` 再 `.transpose(1,2)`；ND → 照抄 `ref.stride()[1:]`。**alias 的 `storage_offset` 恒为 0**
  （行基址是 alias 头，不是参考张量的偏移）
- `run_slot_forward(*, state, buffer, plan, token_dispatch_output, fused_experts_input, use_fusion_ops)`：
  prefetch → `build_mlp_compute_input(...)`（与 owner 路径同源）→
  `dataclasses.replace(mlp_in, weights=state.compute_weights())` → `unified_apply_mlp`

**3) `vllm_ascend/ops/fused_moe/moonep_comm.py`**

- `:363-367`：删掉这里自己算的 `F = int(w13.shape[2]) // 2`，改由
  `_LayerWeightState(layer, E, H, rank, dev)` 内部推导 + 校验（**不是因为它算错**
  —— 量化布局实测是 `[epn, H/2, 2F]`，`shape[2]//2` 恰好仍是 F；是为了把布局知识收在一处：
  w2 是 `[epn, F/2, H]`、两个 scale 又各不相同，散落推导迟早会错）
- `:812-818`：slot 分支改调 `run_slot_forward(..., token_dispatch_output=disp,
  fused_experts_input=fused_experts_input, use_fusion_ops=self.use_fusion_ops)`

**4) `vllm_ascend/envs.py`** —— 登记 `MOONEP_SLOT_MODE`（bool）、`MOONEP_B`（str/None）、
`MOONEP_SLOT_QUANT`（bool，默认 1）。同时把 README 三处（`:18`、`:208`、`:248`）与
`docs/cross_repo_fixes.md:296` 的"slot 仅支持 bf16/不支持量化"口径改成了量化后的说法。

**实现期踩到、值得后人记住的四个坑**（都在 bf16 回归里暴露，量化下被 `element_size()==1` 掩盖）：

1. **`stride(0)` 是元素、`row_bytes` 是字节**——只有 1 字节 payload 时两者才恰好相等。
   用 `stride(0) != row_bytes` 做断言在 bf16 下必然误报（`_packed_alias` 与 `_make_spec` 各踩一次）。
2. **`full[rank*epn:(rank+1)*epn].contiguous()` 不是新张量**——对已连续的切片 `.contiguous()` 是 no-op，
   返回的是 `storage_offset != 0` 的视图。裸字节镜像必须从 **`data_ptr()`** 起算，不能从
   `untyped_storage().data_ptr()` 起算，否则镜像的是整表的头部而不是本 rank 的行。
3. **`unified_apply_mlp` 两个分支都返回元组** `(hidden_states, before_gmm2_evt)`
   （unquant 分支 `return hidden_states, None`），调用方要 `rows, _ = ...`。
4. **alias 的行基址必须是 alias 头**：参考张量自带 `storage_offset` 时不能照抄，否则整块 alias
   会平移进 slot 段（`_packed_alias` 里显式断言 `tensor.data_ptr() == base` 钉住这点）。

### Phase 2 — vendored python（可选，仍不重编）

把 K2 的 `npu_format` 参数收进 `buffer_c.py:396-434`、`_ELEM_SIZE`/`_chunk_nbytes`（`:61-94`）放宽到
`dtype.itemsize`、`api.py:777-796/1046-1051` 解除 bf16 断言、`ops.py`/`launchers.py` 的 dtype 守门。
纯粹为了可读性 —— Phase 1 用 bf16 载体已经绕开全部断言。
**注意两份 `ascend_moonep/` 必须同步改**（`PYTHONPATH` 顺序一变就崩）。

### Phase 3 — csrc + AscendC（可选，需重编）

`csrc/ops/prefetch.cpp:48-79/103-112`（host ABI 加 `element_bytes`、去掉 dtype 白名单）、
`include/ascend_moonep/kernels/launch.h:146-166`、`kernels/prefetch.cpp:58-102`、
`kernels/prefetch_udma.cpp:52/103/139`。
**动手前必须先备份**：`ascend_moonep/_C.cpython-312-x86_64-linux-gnu.so` 目前**没有备份**
（`lib/` 下唯一先例是 `libshmem.so.bak0914`，按同样命名留 `.bakMMDD`）。

---

## 5. 验证阶梯

| 项 | 内容 | 状态 |
|---|---|---|
| M0 | NZ 元数据零拷贝（format 2→29，其余字段不变） | **PASS**（本机 Ascend950PR） |
| M1 | 四个 payload 元数据 + 逐专家步长 | **PASS**（无 padding，step = stride(0)） |
| M2 | 非连续视图 → flat uint8 可读可 clone | **PASS** |
| M3 | home 载体 ← 权重 storage 逐字节无损 | **PASS** |
| M4 | **生死判据**：自建 alias 上的 W4A8MXFP gmm1 与真实权重 bit-identical | **PASS**（max\|diff\|=0） |
| M6a | 真实几何单层 home+slot+full+alias 全部建成（gran/lcm 约束） | **PASS** |
| M6b | 全模型 VA 188.5 GiB 一次性 reserve | **PASS** |
| **S1** | **产品代码路径**（`_LayerWeightState` 真构造，非原型）：量化下四 payload 的镜像逐字节无损 / 投影描述符 source VA 指向 home+slot / alias 形状与逐专家步长对齐 / alias 上 gmm1 与真实权重 bit-identical | **PASS**（`--only S1`，max\|diff\|=0） |
| **S2** | **引擎真正调的那条链**：`build_fused_experts_input` → `run_slot_forward` 本体（buffer 用空替身）→ `unified_apply_mlp`，即融合 gmm1+swiglu（`DeviceOperator.npu_grouped_matmul_swiglu_quant`，A5 上落到 `torch.ops._C_ascend.npu_swiglu_group_quant`，`clamp_value=swiglu_limit=10.0`）与 gmm2（`..._gmm2` → `npu_grouped_matmul(antiquant_scale)`）。判据：swiglu 输出 / out_scale / gmm2 输入 / 最终输出**逐比特相等**，且组数确为 E+B（alias 里 8 个槽位参与 GMM） | **PASS**（`--only S2`，四项 max\|diff\|=0；显存 ~260 MiB、单卡、无需 shmem） |
| **R-bf16** | 改完的产品代码跑既有 bf16 回归：`smoke_slots.py`（R=2）与 `probe_slots_multi.py`（R=3，多热专家，需 `MOONEP_DISABLE_UDMA=1`，UDMA combine 要求 R 为偶数或 16 的倍数） | **PASS**（rel err 0.0052–0.0075） |
| M7 | 引擎端到端：`MOONEP_SLOT_MODE=1 MOONEP_SLOT_STATS=1` 跑一次，`[moonep-stats]` 行必须 `total_slot_rows>0 && calls_with_slot_rows>0`（否则 prefetch 没生效，是**假通过**） | 未跑（需 R≥8 空卡） |
| M8 | 与 owner 模式做 logprobs A/B 对照（`scripts/moe_ab_test.py`） | 未跑 |

复现：`ASCEND_RT_VISIBLE_DEVICES=<空卡> python scripts/spike_slot_quant.py`
（M6 需要 `PYTHONPATH` 带 vendored `third_party/ascend_moonep` + `ASCEND_MOONEP_BUILD_DIR`；
缺依赖时 M6 自动 SKIP）。日志自检：`python scripts/spike_slot_quant.py --check-stats <prefill.log>`。

**判据不通过时的分支**：M4 不过 ⇒ Route A′ 重估（退回 Phase 3 让 kernel 字节化，或放弃 slot+量化）；
M1 出现 `element_size() != 1` 或 `stride(0)*esize != numel/E` ⇒ 整套字节账重算。

**S2 顺带查清的两件事（都会以"算子报错"的形式误导人，写在这里省下一次排查）**：

1. **`ASCEND_CUSTOM_OPP_PATH` 必须在进程启动前设好**。aclnn 的 vendor 解析在 CANN init 时读一次就缓存，
   之后在 `main()` 里调 `bootstrap_custom_op_env()` **已经晚了**：`aclnnSwigluGroupQuant` 会落到 CANN
   内置算子，报 `y dtype must be hifloat8, got 36`（`swiglu_group_quant_hifp8_tiling.cpp:103`）——
   看着像"dtype 不支持"，其实是"vendor 算子没被选中"。spike 因此在 import torch_npu **之前**
   自己设 `ASCEND_CUSTOM_OPP_PATH`（`scripts/spike_slot_quant.py` 顶部）。引擎侧没这个问题：
   `AscendPlatform.import_kernels → bootstrap_custom_op_env` 发生在引擎 init、早于设备初始化。
   选错 vendor 的第二个表现是 `Check yOut != nullptr failed`（clamp 取 0 时的同一个误报）。
2. **A5 上 `torch.ops._C_ascend` 不由 `enable_custom_op()` 加载**：`utils.py:426` 有一道
   "A5 上暂时关闭全部 custom op"的门禁（`VLLM_BATCH_INVARIANT or device_type == A5 ⇒ False`），
   实际把 `.so` 带进来的是 `device_allocator/camem.py:58` 的顶层 import。
   spike 里显式 `import vllm_ascend.vllm_ascend_C` 补齐，否则 `npu_swiglu_group_quant` 直接 AttributeError。

---

## 6. 风险与未知

1. **R ≥ 8 是硬前提**（§3）。当前 2 卡脚本下方案不成立 —— 这是最大的 blocker，且是资源问题不是技术问题。
2. **M4 只在小几何（E=8/H=384/F=64）上验过**，大几何理论上只是同一段元数据逻辑的重复，
   但 `npu_grouped_matmul` 内部对 NZ 描述符的处理路径是否随尺寸变化，没有反例也没有保证。
   落地后 M7/M8 会覆盖。
3. **`get_npu_format()` 返回值是 A5 内部枚举**（实测 NZ 类 payload = 50，ND = 2），
   不同 CANN 版本可能不同 ⇒ 代码里必须照抄实测值，不要写常量。
4. **`npu_dynamic_mx_quant` 返回的 pertoken_scale 是 `uint8` 且已经是 3 维** `[T, H/64, 2]`（M4 实测），
   而 `A5DeviceAdaptor.maybe_normalize_mxfp_scale_layout` 只在 2 维时才 reshape
   （`device_op.py:1229-1234`）—— 手工造张量时容易在这里把 scale 形状搞错。
   **激活的 MX 量化走 `DeviceOperator.npu_dynamic_quant(..., use_mxfp_quant=True)`，不要自己拼。**
5. **UDMA 分支未验证**：本仓运行脚本现在是 `MOONEP_DISABLE_UDMA=1`（走 MTE kernel）。
   若要在 enable_udma 下跑，`kernels/prefetch_udma.cpp` 的字节账要单独复核。
6. `home.copy_` 两个 NZ 张量之间的可靠性未知 —— 这是 M3 存在的理由，Phase 1 一律走裸字节拷贝，
   **不要把 publish 改回 `home_w13.copy_(w13_src)`**。

---

## 7. 不在本次范围

- 改 vendored 库 / 重编 `_C.so`（Phase 1 只动了集成层 4 个文件，Phase 2/3 未做）
- **通信量化**（dispatch/combine 传 fp8、dispatch 字节数减半）—— 用户已明确不纳入。
  现状：moonep 的 dispatch/combine 只发 bf16，而 MC2 在 A5 上走 `dispatch_with_quant`
  （`token_dispatcher.py:214-267`：`quant_mode=4`、`y_dtype=float8_e4m3fn`），
  所以 **moonep 的 dispatch 字节数是 MC2 的 2 倍**。要补得改 ascend-moonep 的 C++/kernel
  （`TORCH_CHECK kBFloat16`、对称 buffer 按 bf16 建、无 scales 参数），集成层单独改不动。
- owner 模式 × W4A8MXFP 的 logprobs A/B（既有遗留项，是任何 slot 工作的前置基线）

---

## 附录 A：spike 实测输出（Ascend950PR，`torch_npu 2.10.0.post4`）

```
[spike] device=npu:0 torch=2.10.0+cpu torch_npu=2.10.0.post4
       plain: format=2 shape=(2, 4, 8) stride=(32, 8, 1) storage=128
       npu_format=29: format=29 shape=(2, 4, 8) stride=(32, 8, 1) storage=128
[M0] PASS  format 2->29，shape/stride/storage/dtype 逐字段不变=True

       w13        shape=(2, 2048, 4096) stride=(8388608, 1, 2048) dtype=uint8 esize=1 fmt=50 contig=False soff=0
                 逐专家步长=8388608 B (期望 8388608) numel/E=8388608 storage=16777216 B 无padding=True 覆盖=True
                 t[1].data_ptr()-t[0].data_ptr()=8388608 B
       w2         shape=(2, 1024, 4096) stride=(4194304, 1, 1024) dtype=uint8 esize=1 fmt=50 contig=False soff=0
                 逐专家步长=4194304 B (期望 4194304) numel/E=4194304 storage=8388608 B 无padding=True 覆盖=True
       w13_scale  shape=(2, 64, 4096, 2) stride=(524288, 2, 128, 1) dtype=uint8 esize=1 fmt=2 contig=False soff=0
                 逐专家步长=524288 B (期望 524288) numel/E=524288 storage=1048576 B 无padding=True 覆盖=True
       w2_scale   shape=(2, 32, 4096, 2) stride=(262144, 2, 64, 1) dtype=uint8 esize=1 fmt=2 contig=False soff=0
                 逐专家步长=262144 B (期望 262144) numel/E=262144 storage=524288 B 无padding=True 覆盖=True
[M1] PASS  四个 payload：element_size==1、逐专家无 padding、t[1]-t[0]==stride(0)*esize 全通过（E=2, H=4096, F=2048）

       w13        flat=[262144] uint8 可读，clone 一致=True
       w2         flat=[131072] uint8 可读，clone 一致=True
       w13_scale  flat=[16384] uint8 可读，clone 一致=True
       w2_scale   flat=[8192] uint8 可读，clone 一致=True
[M2] PASS  非连续镜像视图 → flat [nbytes] uint8 可读且可 clone

       w13        镜像 262144 B，不等字节数=0
       ...（四个 payload 不等字节数均为 0）
[M3] PASS  home 载体 ← 权重 storage 的逐字节拷贝无损（绕开 NZ 互拷）

       x_fp8=(96, 384) torch.float8_e4m3fn pertoken_scale=(96, 6, 2) torch.uint8
       实测 fmt=50            alias fmt=50 shape=(10, 192, 128) 有限=True bit-identical=True max|diff|=0
       NZ 常量 29             alias fmt=29 shape=(10, 192, 128) 有限=True bit-identical=True max|diff|=0
       out_ref shape=(96, 128) 有限=True  home 段字节一致=True（w13 fmt=50, scale fmt=2）
[M4] PASS  自建 format=50 alias + 照抄 stride 的 scale alias 上跑 W4A8MXFP gmm1，与真实权重 bit-identical

       VMM granularity=2097152 B，epn=32，E=256，层数=43
       M6a w13        carrier=[32,128,32768] row=8388608 B  gran 对齐: epn*row%gran=0  lcm/gran 比=1  home+slot 物理=512.0 MiB  full VA=2304.0 MiB
       M6a w13_scale  carrier=[32,8,32768]   row=524288 B   gran 对齐: epn*row%gran=0  lcm/gran 比=4  home+slot 物理=32.0 MiB   full VA=144.0 MiB
       M6a w2         carrier=[32,64,32768]  row=4194304 B  gran 对齐: epn*row%gran=0  lcm/gran 比=1  home+slot 物理=256.0 MiB  full VA=1152.0 MiB
       M6a w2_scale   carrier=[32,4,32768]   row=262144 B   gran 对齐: epn*row%gran=0  lcm/gran 比=8  home+slot 物理=16.0 MiB   full VA=72.0 MiB
[M6a] PASS  真实几何（row 8MiB/4MiB/512KiB/256KiB）下单层 home+slot+full+alias 全部建成；单层物理=816.0 MiB，VA=4.38 GiB
       单卡 R=8：home 镜像物理=17.1 GiB，alias VA=34.3 GiB，full VA=154.2 GiB，合计 VA=188.5 GiB
       一次性 reserve(188.5 GiB) → OK
[M6b] PASS  全模型（43 层）VA 合计 188.5 GiB 可一次性 reserve
```

S1/S2（产品路径，2026-09-17 全量跑，9 项全 PASS；`--only S1 S2` 可单独复现）：

```
       _EXTRA_CTX 已替换为 fake：[...]
       E=8 epn=8 B=8 payload=['w13', 'w13_scale', 'w2', 'w2_scale']  slot 行已填入对应专家字节
       gmm1 走 MXFP 融合分支=True（use_mxfp_quant=True、mxfp_quant_dtype=W4A8MXFP、weight_quant_type=fp4x2、单张量权重）
       组数：ref=8（E=8）alias=16（E+B=16） 一致=True
       融合 gmm1 的 swiglu 输出      bit-identical=True  max|diff|=0  shape=(64, 2048)
       融合 gmm1 的 out_scale       bit-identical=True  max|diff|=0  shape=(64, 32, 2)
       gmm2 的输入                  bit-identical=True  max|diff|=0  shape=(64, 2048)
       最终输出                      bit-identical=True  max|diff|=0  shape=(64, 4096)
[S1] PASS  产品代码 _LayerWeightState：量化下四 payload 的镜像/投影/alias 全对，alias 上 gmm1 与真实权重 bit-identical
[S2] PASS  产品入口链（run_slot_forward 本体 + 两个封装算子）：alias 与真实权重逐比特一致=True，走的确实是 MXFP 融合分支=True，alias 组数含 8 个槽位=True，输出有限=True
```

S2 的 slot 段只给了槽位 0 组 16 行 token，其余 7 个槽位 0 行 —— 与引擎里"只预取被路由到的专家"同形；
`out_scale` 是 e8m0（`[64, 32, 2]`，与 M4 的 pertoken_scale 同布局），也逐比特一致。

DSv4-Flash 几何取自 `config.json`：`hidden_size=4096, moe_intermediate_size=2048,
n_routed_experts=256, num_hidden_layers=43`（⇒ F=2048，2F=4096）。

## 附录 B：为什么改 `fused_moe_0_23_0.py` 而不是 `fused_moe.py`

装的是 vLLM 0.23.0，所以活实现是 **`fused_moe_0_23_0.py`**；`fused_moe.py` 对 0.23.0 只是门面：

- `fused_moe.py:291-295`：`if vllm_version_is("0.23.0"): from ...fused_moe_0_23_0 import AscendFusedMoE, AscendMoERunner`，
  并**改写 `__module__ = __name__`** —— 这就是"看起来像 fused_moe.py"的假象来源：
  `inspect.getsourcefile(AscendFusedMoE)` 会误报 `fused_moe.py`，但
  `inspect.getsourcefile(AscendFusedMoE.__init__)` 是 `fused_moe_0_23_0.py`（`co_filename` 改不动）。
- `fused_moe.py` 自身**没有 `class AscendFusedMoE`**；全仓 `grep enable_moonep` 也只命中
  `fused_moe_0_23_0.py`。

**两个文件为什么同时存在（main2main）**：上游 vLLM 做了 MoE 重构（`FusedMoE`/`MoERunner`
inversion refactor，vllm PR #41184），把 MoE 所有权从 legacy 类路径改到 factory+runner 路径。
本仓提交 `1c57c6a0`（2026-06-29，`[Misc] main2main 0608(refactor fused moe) (#11081)`）用
`vllm_version_is("0.23.0")` 做版本门：legacy 实现搬进 `fused_moe_0_23_0.py`（自述
"Legacy vLLM 0.23.0 FusedMoE implementation. **Private module.**"），`fused_moe.py` 变成
"新路径实现 + 对外门面"（新路径下只有 `:299` 起的 `AscendMoERunner`，注册换成
`patch/platform/patch_fused_moe.py:32` 的 main-only factory patch；0.23.0 侧靠
`utils.py:763-766` 填 `REGISTERED_ASCEND_OPS["FusedMoE"]`）。

> ⚠️ **MoonEP 目前是 vllm 0.23.0-only**：`fused_moe.py` 里 `moonep/MOONEP` 命中数为 **0**，
> 三个 moonep 提交（`0a12d885` owner、`ab006ad3`、`47e3038e` slot）都只动
> `fused_moe_0_23_0.py` + `moonep_comm.py`/`moonep_balance.py`。
> **将来若跟进 main2main 升到 main，整条 MoonEP 链路要在新的 factory/runner 路径上重做一遍**，
> 本文 §4 的挂接点（`register_moonep_layer` 调用位、权重镜像/alias 注入位）也要跟着再落一次。
