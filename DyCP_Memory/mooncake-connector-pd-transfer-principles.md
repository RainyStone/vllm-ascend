---
name: mooncake-connector-pd-transfer-principles
description: vllm-ascend mooncake_connector.py P/D KV传输原理——双信道(ZMQ握手+mooncake RDMA)、逐层地址模型、start_load_kv链路、PCP/DCP两级CP切分(split_metadata)、TP冗余拉取与reformat；扁平dycp_size适配后CP切分(_get_kv_split_metadata起)未迁为待适配主战场
metadata:
  node_type: memory
  type: reference
---

# mooncake_connector P/D KV 传输原理

本文记 mooncake_connector 的 P/D KV cache 跨节点拉取**机制本身**，基于代码取证（见 [[vllm-debug-evidence-based]]）。当前代码已有 PCP/DCP 两级 CP 切分（`ReqMeta` 含 `remote_pcp_size`/`remote_dcp_size`/`remote_ptp_size`）与 domain 方案引入的扁平 `dycp_size`；版本无关的传输链路原理见正文第 1–6 节，扁平 dycp 的待适配现状见末节。PCP/DCP 的 Q/KV 切分/slot_mapping/scheduler 块膨胀原理见 [[pcp-dcp-kv-sharding]]，DyCP 总体方案见 [[dycp-design-principles]]，本文只讲连接器**传输链路**。

## 1. 定位与三角色架构

mooncake TransferEngine 实现的 vLLM v1 KV connector，做 P(prefill/producer) → D(decode/consumer) 的 **跨节点 KV cache 拉取**。

| 类 | 进程/线程 | 职责 |
|---|---|---|
| `MooncakeConnector` | 入口 | 按 `KVConnectorRole`(SCHEDULER/WORKER) 内含一个 scheduler 或 worker |
| `MooncakeConnectorScheduler` | scheduler 进程 | 决定哪些 req 需拉 KV，构建 `ReqMeta`，管理本地 block 延迟释放 |
| `MooncakeConnectorWorker` | worker 进程 | 建传输线程，真正发起/响应 KV 拉取，持有地址表 |
| `KVCacheSendingThread` | worker 后台线程(P端) | bind ZMQ ROUTER 响应 `GET_META_MSG`/`DONE_RECVING_MSG`，按计数释放延迟 block |
| `KVCacheRecvingThread` | worker 后台线程(D端) | 拉远端 agent metadata，用 mooncake 引擎按 block 直读 KV 写本地 block |

## 2. 两条关键信道

- **ZMQ RPC（轻量握手，只传元信息不传 KV）**：P 端 `KVCacheSendingThread` bind ROUTER 到 `side_channel_port + device_index`；`GET_META_MSG` → 回送 `MooncakeAgentMetadata`（engine_id/kv_group2layeridx/kv_caches_base_addr/block_size_scale/block_lens/local_ip）；`DONE_RECVING_MSG` → P 端释放该 req 延迟 block（`port_send_num` 对齐远端 `num` 次才放）。D 端用 REQ socket 经 socket 池发这两类消息。
- **mooncake TransferEngine（RDMA/直读，传真正 KV）**：D 端拿到远端 `kv_caches_base_addr` 后构造 `(src,dst,length)` 列表，`engine.batch_transfer_sync_read` 从远端 NPU 地址直接读到本地 NPU block。

## 3. KV cache group 概念（vLLM v1 上游定义 + 与传输的关联）

**定义**（`vllm/v1/kv_cache_interface.py`）：`KVCacheGroupSpec` = 一组**共享同一 KV cache block table、在 KV cache manager 里视为"一层"**的模型层（字段 `layer_names`/`kv_cache_spec`/`is_eagle_group`）。`KVCacheConfig.kv_cache_groups`：注意力类型相同的层归一组——单 attention 模型 1 个 group（全层）；hybrid 架构（Mamba+attention / DeepseekV4 等）多 group。`group_id` = 列表下标。

**connector 主索引**：`_build_kv_group2layeridx` 把上游 `kv_cache_groups` 转成 `{group_id: (序列化 group_spec, [layer_idx])}`；`_serialize_kv_group_spec` 产出 `kv_cache_spec_type`（=`type(kv_cache_spec).__name__`，如 `FullAttentionSpec`/`MambaSpec`/`UniformTypeKVCacheSpecs`）+ spec 细节（`num_key_value_heads` 等，MambaSpec 额外 `shapes`/`dtype_sizes`）。该序列化作为 `MooncakeAgentMetadata.kv_group2layeridx` 经 ZMQ 握手传对端。

**与传输的关联**：

| 关联点 | 说明 |
|---|---|
| P↔D 结构对齐校验 | D 端 `_get_remote_metadata` 校验 `remote.kv_group2layeridx == local`，并取对端 `num_key_value_heads` 供 CP head 分组 |
| `BlockIds` 是 per-group | local/remote_block_ids 为每 group 一段 block id 的元组，`local_block_ids[group_idx]` 取值；`_transfer_kv_cache_all_groups`/`request_finished_all_groups`/`get_unhashed_block_ids_all_groups` 的 "all_groups" 即跨所有 group |
| group = 传输外层维度 | 地址表逐 **layer** 建（`[layer_idx]`），但传输循环按 **group → layer → cache_idx(K/V) → block 配对** 组织 |
| CP head 切分依据 group spec | `get_kv_head_groups` 按 `kv_cache_spec_type`：MLA/sparse/Mamba 视作单 head 组 `(0,)`，普通 attention 按 tp 切 `num_kv_heads`；Mamba group 特殊切分（只传 final shard 最后一块 SSM 状态） |
| per-group 完成标志驱动 reformat | `GroupPull.is_group_transfer_end`：group 的 TP pull 齐了才对该 group reformat；reformat 也 per-group（`_get_group_kv_caches`） |

**对扁平 dycp 适配的意义**：group 维度本身不受 dycp 结构变化影响——CP 组拓扑/配对/reformat 都在 group 维度内进行；变化的是**组内 block_ids 从单段变为按 cp_ranks 分多段**（见末节）。

## 4. 逐层地址模型（`register_kv_caches` 建立，传输与切分的底层依赖）

每 worker 计算逐层[layer_idx][cache_idx=K/V] 三张表：
- `kv_caches_base_addr`：cache 张量 `data_ptr()`
- `block_len_per_addr`：一个张量 block 字节数 = `element_size × prod(block_shape)`
- `block_size_scale`：`tensor_num_blocks / num_blocks`（逻辑 block↔张量 block 倍数；`expand_block_ids(bid,scale)=bid*scale+offset` 把逻辑块展开为张量块）

地址表经 `MooncakeAgentMetadata` 在 P↔D 间互传，使 D 端知"远端 block N 物理地址 = `remote_kv_caches_base_addrs[layer][cache] + block_id × block_len`"。2M 对齐校验后 `global_te.register_buffer` 注册到全局 mooncake 引擎。

## 5. 端到端数据流

**Scheduler 侧**：`get_num_new_matched_tokens`（需传 token 数 = `len(prompt_token_ids) - num_computed_tokens`）→ `update_state_after_alloc`（命中 remote prefill 时取本地 unhashed block_ids 入 `_reqs_need_recv`）→ `build_connector_meta`（遍历 `_reqs_need_recv` 构 `ReqMeta`：本地/远端 block_ids、远端 host/port/engine_id）→ `request_finished`（本 D rank 前缀算完后**延迟释放**本地 block，记 `_reqs_need_send[req]=time`，等 P 端确认所有 D rank 收完才真正放）。

**Worker 侧**（`start_load_kv` 主驱动，每 req）：
1. `_get_kv_split_metadata` → 算"从哪些远端 port、拉哪些 block"，返回 `(remote_handshake_port_list, local_block_ids_list, remote_block_ids_list)` 三者按 shard 对齐
2. `_get_group_pulls_metadata` → 每 remote port 配一组 `GroupPull`（group_id/remote_tp_offset/num_group_pulls/prefill_pp_rank/is_group_transfer_end）
3. `kv_recv_thread.add_request` 入队 → `_transfer_kv_cache_all_groups`

## 6. CP 切分原理（核心，`_get_kv_split_metadata`，基于 PCP/DCP 两级 CP）— 待适配主战场

1. **退化**：两端 CP 都为 1（`remote_pcp×remote_dcp×self.pcp×self.dcp == 1`）→ 单 port 直拉，用 `_get_remote_rank`（单 CP）/`_get_hybrid_remote_rank_group_pulls`（hybrid MLA）选一个远端 rank。
2. **CP 组拓扑**（`get_cp_group_meta`）：按 kv_head 分组（MLA/sparse 视作单 head group），P/D 两端各建 cp_group，len = `pcp_size × dcp_size`，组内元素 = `port_base + pcp_rank_offset + dcp_rank_offset + kv_head_offset`。
3. **D↔P 端口配对**（`get_local_remote_block_port_mappings`）：按 kv_head 子集匹配 D/P，`select_cp_groups_id` 轮转选 P 端 cp_group，建 `D port → [[P port 列表]]` 映射，并算 `remote_port_send_num`（每 P port 被多少 D 拉读）。
4. **block 数量切分**：`num_prompt_blocks` 按 `remote_cp_size` 均分到各远端 CP rank（余数前排），扣掉 P↔D 前缀缓存命中块；**本 D rank 只取 `cp_rank % local_cp_size == local_cp_rank` 的部分**（`local_cp_rank = dcp_rank + pcp_rank × dcp_size`）；最后不满块移末尾 shard 对齐。
5. **block 配对**：每 shard 按 `num_blocks_to_pull` 从 remote/local block_ids 切片配对；Mamba state 特殊——只从 final shard 传最后一块状态。
6. **TP 冗余拉取**（`_get_remote_ranks_for_req`）：P 端 TP > D 端 TP 时，D 每 rank 拉哪几个 P TP rank 的 copy（seed=req_id 哈希，可复现随机分组），`tp_num_need_pulls = prefill_tp / decode_tp`（MLA/sparse 按 num_kv_head 折算）。

## 7. KV 拷贝与 reformat（`_transfer_kv_cache_all_groups`）

- 每 group_pull 按 layer（PP shard 截断）× cache_idx(K/V) 算精确地址：`src`(本地)=`local_base + local_block_id × block_len + inner_offset × inner_block_len`；`dst`(远端)=`remote_base + remote_block_id × inner_block_len`；`inner_block_len = block_len // tp_num_need_pulls`（TP 切分粒度）。
- 全部 (src,dst,length) 聚一个列表，**一次** `engine.batch_transfer_sync_read` 批量直读。
- 拉完**仅 `is_group_transfer_end` 的 group 才 reformat**：`tp_num_need_pulls > 1` 时远端 KV 按 `[block, split, token, head, dim]` 传回，需 transpose 还原成 `[block, token, split, head, dim]`；按需做 cat / NZ 格式转换（hybrid MLA 走 torch 路径 `reformat_kv_cache_hybrid_linear_torch`，普通走 fused op 或慢路径 `reformat_kv_cache`）。
- 完成后向 P 端发 DONE_RECVING，P 端计数到 `remote_port_send_num` 后释放延迟 block。

## 8. 扁平 dycp 适配现状（待适配，不含方案，细节逐步对齐）

domain 方案引入扁平 `dycp_size`，与 PCP 互斥。当前代码相对两级 CP 版本的改动：`ReqMeta` 增 `local_dycp_ranks`/`remote_dycp_ranks`（按 CP rank 分组 block_ids）；`update_state_after_alloc` 在 blocks 为 list 时按 cp_ranks 逐个取 unhashed block_ids（两级 CP 版本统一 `get_unhashed_block_ids_all_groups`）；新增 `dycp_port_base`、`_reqs_need_send_cp_ranks`、`clear_reqs_need_recv`；`build_connector_meta` 按 `scheduler_output.cp_rank_to_req_id` 过滤，只为本 cp_rank 相关 req 构建 meta；SendingThread 直接收 `handshake_port`，去掉 `device_index` 计算。

**待适配断点**：`_get_kv_split_metadata` 及其后所有方法（CP block 切分配对）**尚未按新 block_ids 结构改造**，留 TODO："此函数及其以后的修改都没有迁过来，由于 block_ids 数据结构变化，后期需结合 domain 方案代码及新 block_ids 结构修改"。适配目标：把第 6 节基于 PCP/DCP 两级 CP 的切分红程序改造为支持 **扁平 `dycp_size` + 按 cp_ranks 分组 block_ids 的新结构**。
