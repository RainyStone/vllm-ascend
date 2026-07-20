---
name: pcp-dcp-kv-sharding
description: vllm-ascend 纯PCP/纯DCP方案原理与Q/KV切分、注意力计算、KV cache interleave分片存储完整链路（scheduler块膨胀→worker virtual block寻址），基于代码取证
metadata: 
  node_type: memory
  type: reference
  originSessionId: 95cd8c53-5161-4f38-aa8e-27d85407835b
---

# 纯 PCP / 纯 DCP 方案 与 KV Cache 切分链路

本文只讲**纯 PCP** 与**纯 DCP** 两条独立线，不含 DyCP（DyCP 见 [[dycp-design-principles]]）。所有结论经代码取证，文档仅作参考（见 [[vllm-debug-evidence-based]]）。

## 0. 定位（代码口径）

| | PCP | DCP |
|---|---|---|
| 全称 | Prefill Context Parallel | Decode Context Parallel |
| 通信域 | 独立 PCP 组 (`get_pcp_group()`) | 复用 TP 组 (`get_dcp_group()`) |
| 目的 | 长 prefill 算力切分 | KV cache 沿 seq 维分片存省显存 |
| 影响阶段 | prefill（含 chunked prefill） | decode、chunked/cached prefill |
| 激活 | `prefill_context_parallel_size>1`（runner `use_prefill_cp=pcp_size>1`，`model_runner_v1.py:605`） | `decode_context_parallel_size>1`，约束 `dcp_size<=tp_size//num_kv_heads`（`platform.py:393-400`） |

正交：`cp_size=pcp_size*dcp_size`，`cp_rank=pcp_rank*dcp_size+dcp_rank`。

## 1. Q / KV 切分结论

- **PCP**：✅切 Q（sequence 维，head-tail 不相交，算力切分，`pcp_utils.py:537-688`）；KV cache **interleave 分片存**（非每卡全量副本），但 attention 计算时 `pcp_group.all_gather(kv,0)` 聚齐成全 KV（`mla_cp.py:880-888`）。
- **DCP**：❌不切 Q 算力；KV cache 沿 seq 维 **interleave 分片存省显存**；Q 仅因复用 TP 域做 **head 维 all_gather**（`reorg_decode_q:937-942`），结果用 **all-to-all** 交换 output+LSE（`common_cp._process_attn_out_lse:116-121`），再 `_npu_attention_update` 归并。

> ⚠️ 纠偏：文档口径"PCP 每卡存全 KV 副本"与代码不符。纯 PCP 时 `block_table.py:237` `total_cp_world_size=pcp_world_size*1` 仍进 interleave 分支，KV 物理分片存；`common_cp.py` 注释明示"token i 的 KV 只存在 dcp_rank==i%pcp_world_size 的卡上"（纯PCP时等效 pcp_rank==i%pcp_world_size）。存的是分片，算时才聚齐。

## 2. Slot_mapping interleave（PCP/DCP 共用底座，`block_table.py`）

`compute_slot_mapping`（kernel `:230-249`，纯PCP/纯DCP都走）+ `compute_slot_mapping_with_dycp`（numpy `:166-228`，DyCP用）：

- `virtual_block_size = block_size × total_cp_world_size`（`block_size` 膨胀 cp_world_size 倍）
- 归属 mask：`(vblock_offset // cp_kv_cache_interleave_size) % total_cp_world_size == total_cp_rank`，非本 rank → slot=-1(PAD)
- 本地 slot 偏移：在 interleave_size 粒度内压缩到本 rank 连续区间
- `token_idx=i` 的 KV 只存归属 rank 上 → **KV 物理 seq 维分片，省显存**
- 约束：SFA(sparse) 模式强制 `cp_kv_cache_interleave_size==block_size`（块级 interleave，`platform.py:678`），保证膨胀因子为整数 cp_world_size、一虚拟块覆盖每卡恰一物理块
- init：`pcp*dcp>1` 时普通 attention 不膨胀块表行宽；仅 `MambaSpec` 额外 `max_num_blocks_per_req *= cp_world_size`（`:50-53`）。slot_mapping buffer 预留 `2*pcp_world_size*max_num_reqs`

**PCP 时序约束**：纯 PCP 必须在切 Q 之前用**原始全局 positions** 调 `compute_slot_mapping`（`model_runner_v1.py:894-908`），再调 `update_tokens_for_pcp` 切乱 positions——否则非连续 position 无法线性映射回 slot。

## 3. Scheduler 侧块膨胀（`vllm/v1/core/single_type_kv_cache_manager.py:59-63`，上游已patch）

```python
self.block_size = kv_cache_spec.block_size
if dcp_world_size * pcp_world_size > 1:
    self.block_size *= dcp_world_size * pcp_world_size   # 逻辑块=物理块×cp_world_size
```

`get_num_blocks_to_allocate`(:119)、`allocate_new_computed_blocks`(:205)、`cache_blocks`(:287) 等所有块数计算用膨胀后 `block_size`：`num_required_blocks = cdiv(num_tokens, self.block_size)`。

**为什么膨胀**：interleave 分片下，一段 `cp_world_size×block_size` 个逻辑 token 才在每卡填满一个物理块。scheduler 块数计量粒度必须与分片物理布局对齐——一个"逻辑块组(=膨胀后block_size)"对应 cp_world_size 张卡各一个物理块。这与 worker 的 `virtual_block_size` 是**同一膨胀因子在两层级的体现**：scheduler 用它做**块数计量/分配/prefix命中/free-evict**，worker 用它做 **position→slot 寻址**，必须一致才能正确分配索引回收分片 KV。用乘法膨胀(块变大→块数变少)而非除法，是为了让每卡分到的物理块数 = `cdiv(num_tokens, 原始block_size)` 不变（每卡持自己那 1/cp_world_size 的完整块容量）。

> 已证（如下）：block pool 按 **local shard** 定大小，每卡 N 物理块对应 scheduler 计的 N 逻辑块组（一逻辑块组=每卡1物理块），非 N×cp_world_size；每卡各自持有 block pool。

## 3a. 块计数的精确调用链（取证确认，纯PCP无bypass）

**真正算块数的那行**：`single_type_kv_cache_manager.py:119` `num_required_blocks = cdiv(num_tokens, self.block_size)`。
并行的 prefix 命中块数：`single_type_kv_cache_manager.py:470-472`（`FullAttentionManager.find_longest_cache_hit`）`max_num_blocks = max_length // block_size`（同膨胀）。

**初始化期（膨胀注入）**：EngineCore 选 Scheduler（纯PCP→基类 `Scheduler`；`CPAwareScheduler` 是 **DyCP 专用**、与PCP块计数无关，`num_cp_seqs>0` 才选） → `Scheduler.__init__` 读 dcp/pcp_world_size（`scheduler.py:149-150`，**直接读 parallel_config，不走 get_total_cp_world_size**） → 构造 `KVCacheManager`（`scheduler.py:223-235`） → `get_kv_cache_coordinator`（`kv_cache_manager.py:137-148`） → `UnitaryKVCacheCoordinator` 等（`kv_cache_coordinator.py:594-642`→`:66-78` 建单类型 managers） → `SingleTypeKVCacheManager.__init__:59-63` 膨胀注入。

**调度期（块数计算被触发）**：`Scheduler.schedule()`（`scheduler.py:310`）→ waiting/prefill 路 `get_computed_blocks`(`:574-576`)→`find_longest_cache_hit`(`:470-472`)，后 `allocate_slots`(`:702-712`)；running 路 `allocate_slots`(`:423-429`) → `KVCacheManager.allocate_slots`（`kv_cache_manager.py:339-347` 和 `:366-374`）→ `KVCacheCoordinator.get_num_blocks_to_allocate`（`kv_cache_coordinator.py:114-134` 遍历各 manager）→ `:119` 算出。**纯 PCP 在 schedule()/running/waiting 内无任何 CP 条件分支，标准 v1 块分配照常运行**。

**结果**：块数 = `cdiv(num_tokens, base_block_size × pcp_world_size)`（纯PCP, dcp=1）。一个逻辑块组对应 cp_world_size 张卡各一物理块。

## 3b. CP 世界数来源的两套路径（intentional，勿混）

- **Scheduler/KVCacheManager**：直接读 `parallel_config.decode_context_parallel_size`/`prefill_context_parallel_size`，**不含 dycp_world_size**。
- **Worker**（`block_table.py:248`、`gpu_model_runner.py:6648`、attention backend `:765`）：用 `get_total_cp_world_size()`（`cp_utils.py:47-64`），**含 dycp_world_size**。
- 有意分离：KVCacheManager 膨胀只管 PCP/DCP 分片，不能折 DyCP。

## 3c. 内存侧自洽

`num_gpu_blocks`(block pool大小) 按 **local shard** 算：`FullAttentionSpec.max_memory_usage_bytes`（`kv_cache_interface.py:197-206`）`max_model_len/cp_world_size`；`resolve_kv_cache_block_sizes`（`kv_cache_utils.py:569-594`）返回 `scheduler_block_size = cache_config.block_size×dcp×pcp`（hybrid 模式 `:596-600` 拒绝 CP）。故"pool按本地分片定大小 + 每请求块数用膨胀block_size算"自洽。

## 4. 注意力流程

**纯 PCP prefill**（`mla_cp.py`）：`update_tokens_for_pcp` 切 Q head-tail → preprocess 拿本地 Q、`pcp_group.all_gather(kv,0)` 聚全 KV+`pcp_allgather_restore_idx` 还原序、`reshape_and_cache`(借slot_mapping只写本rank分片) → `_forward_prefill:944-1009` head/tail 两段 attention(带mask) → `q_full_idx` 还原 → `_process_attn_out_lse:123-125` `pcp_group.all_gather(out_lse,0)` → `_npu_attention_update:171-197` 归并 → v_up_proj→o_proj。

**纯 DCP decode**：preprocess `reorg_decode_q` head维 all_gather Q、写本rank分片 KV → `npu_fused_infer_attention_score`(softmax_lse_flag=True) 本地算 out+LSE → `_process_attn_out_lse:116-121` `all_to_all_single(group=dcp_group)` 交换 out+LSE → `_npu_attention_update` 归并 → v_up_proj→o_proj。

## 5. 一句话链路（纯PCP为例）

scheduler 块膨胀(block_size×cp_world_size)按逻辑块组分块 → block_pool 分配 → runner 用原始positions调ComputeSlotMapping(interleave virtual block算slot) → pcp_utils 切Q head-tail → reshape_and_cache借slot_mapping只写本rank KV分片 → attention算时 all_gather 聚全KV配切分Q算 → all_gather(out+LSE)+归并。DCP 不聚KV、改head维ag(Q)+all_to_all(输出)对账。

## 6. PCP decode 阶段与执行模型（纯 PCP，代码取证）

### 执行模型：序列并行，非权重并行
PCP 每 rank 持完整模型权重，跑完整前向（embedding→MLP→o_proj→logits），只在 attention 层内部对 Q/KV 序列分片。区别于 TP（切权重+allreduce）。

### 各 rank hidden_states 相同的闭环
每层 attention 经 `common_cp.py:125` `pcp_group.all_gather(attn_out_lse, dim=0)` 聚齐 out+LSE → 各 rank attention output 相同 → o_proj 相同 → 下一层 hidden_states 相同 → 整条前向各 rank 一致 → logits 各 rank 相同。

### Q / KV 分布（decode，纯 PCP 路径 common_pcp_size>1 and dycp_size==1）
- Q：每 rank 一份相同副本（replica，零额外通信）。`mla_preprocess_decode`（mla_cp.py:912-923）每 rank 自算，不切、不 head ag（PCP 独立组 head 不分片）；Q 来自完整 hidden_states（各 rank 相同）→ 天然相同。
- KV：只归属 rank 一份（分片，无副本）。`mla_cp.py:132-136` strided 选本 rank decode slot、非归属 fill_(-1)；`exec_kv_decode`（mla_v1.py:1320）按有效 slot 写。归属 rank = position % pcp_size（interleave 轮转）。

### 重复计算边界
- 重复：MLP/norm/o_proj/logits/Q 计算（各 rank 算相同结果）。
- 不重复：attention out+LSE（各 rank 算不同 KV 段，all-gather 聚齐）；KV 存储（分片）；KV 读取（各 rank 读自己分片）。

### 采样对齐（纯 PCP vs DyCP 关键区别）
- 纯 PCP：无 sampled broadcast（grep 确认 pcp_group 无 sampled 通信；`_sync_dycp_sampled_token_ids` 对 dycp_size<=1 直接 return）。靠「logits 天然相同（attention all-gather 聚齐）+ 确定性采样（greedy/seed）」让各 rank 采出相同 token。
- ⚠️ 待实测：代码无 broadcast 兜底；temperature>0 无 seed 的非确定性采样，各 rank（独立进程、RNG 不同）可能采出不同 token——是否有上层保证未确认。

### PCP 收益主场
prefill 切 Q（head-tail）真并行省算力；decode 切不动 Q（单 token），退化为 KV 分片省显存（DCP 的活）。