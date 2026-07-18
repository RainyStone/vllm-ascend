---
name: dycp-design-principles
description: DyCP（Dynamic Context Parallel）方案原理：长短分流、CP子组拓扑/路由、CPAwareScheduler子组共识、DP全组状态机、混合batch重排与计算布局
metadata: 
  node_type: memory
  type: reference
  originSessionId: 1a1608ce-a48f-4def-b9c9-180e664c2d7f
---

# DyCP 方案原理

DyCP = Dynamic Context Parallel：在 **DP（数据并行）** 之上按请求长短动态叠加 **CP（上下文并行）**。

## 1. 长短分流
- 判据：请求 **prefill token 数** 与调度器配置阈值 `long_request_threshold` 比较。
- **短请求**（< 阈值）：纯 DP，路由到**单个** DP 引擎，该引擎独立解码。
- **长请求**（≥ 阈值）：走 CP，由 `dycp_size` 个相邻 DP 引擎组成 **CP 子组**协作解码（组内做上下文切分 + 完成共识）。

## 2. 拓扑与路由
- `data_parallel_size` 个引擎按 DP rank 顺序排列，每 `dycp_size` 个相邻引擎构成一个 CP 子组（子组 g = rank `[g*cp_world_size, (g+1)*cp_world_size)`）。
- **CP 请求** → 路由到一个子组的全部 `dycp_size` 个引擎；owner = 子组内 cp_rank 0，是唯一输出流来源。
- **短请求** → 路由到单个最空闲引擎（LB 评分 = waiting*4 + running）。

## 3. 调度与子组共识（CPAwareScheduler）
- `num_cp_seqs > 0` 时选用 `CPAwareScheduler`。
- `schedule()` = 基类调度 + CP 元数据标注 + **逐拍子组共识 all_reduce（MIN）**：三态 `SCHEDULED=2 / NOT_SCHEDULED=1 / PREEMPTED=0`，min≥SCHEDULED→确认共同执行，min≥NOT_SCHEDULED→软回滚（重入队不重置 KV）。作用：让子组各 rank 对"本拍某 CP 请求是否真调度"达成一致，防错位。
- CP 完成时也走**子组内**共识，不涉及全 DP。

## 4. DP 全组状态机（run_busy_loop + sync_dp_state）
- 各 DP 引擎跑 busy loop，靠全 DP `sync_dp_state` all_reduce 判断"是否仍有未完成请求"，达成 wave 运行/暂停的全局共识。
- `engines_running` 决定 wave 结束；客户端据此决定新请求来时是否发 `FIRST_REQ` 唤醒暂停引擎、广播 `START_DP_WAVE`。
- `sync_dp_state` 每拍自增 `step_counter`，仅 `% N == 0`（N=32）时做真实 all_reduce，中间返回 True（性能优化）。

## 5. 混合 batch 的请求重排与计算布局

混合 batch（同一拍既有 CP 长请求又有短请求）时，DyCP **不混合计算长短**，而是先对 batch 内请求做**重排**，再据此布局计算：

- **重排函数**：`reorder_batch_to_split_cp_and_normal`（vllm `attention/backends/utils.py`），在 `gpu_model_runner._update_states` 里、构建 attention metadata **之前**调用，条件 `num_cp_request > 0`。
- **重排规则**（稳定分区）：
  - 判 CP：`scheduler_output.cp_rank_scheduled_tokens[req_id] > 1`（CP 请求=cp_world_size，短请求=1——即 `cp_aware_scheduler.schedule()` 里 `cp_rank_scheduled_tokens = cp_world_size if CP else 1`）。
  - **CP 请求排到 batch 前、按 req_id 字符串排序**（保证子组各 DP rank 把 CP 请求排成一致相对顺序——因各 rank running 队列可能插着不同普通 DP 请求，但 CP 请求子组共享、必须同序）。
  - 短请求排后、保留原相对顺序。
  - 目标顺序与现状相同则不操作；否则用 O(num_reqs) swap 物理重排 input_batch。
  - 最终布局：`[cp0, cp1, ..., ncp0, ncp1, ...]`。
- **判据澄清**：重排区分的是"是否 CP 请求"（`cp_rank_scheduled_tokens>1`），**不是**直接按 token 长度。CP 请求恰好都是达阈值的请求，故效果上等价"长在前、短在后"。
- **与后续计算的耦合**（重排使"前 N_cp 个=CP"成立，对应以下前缀假设）：
  - `SchedulerOutput.num_cp_request` = 前缀 CP 数量。
  - `PCPManager` 用 `num_scheduled_tokens[:num_cp_request]` 对 CP 区做 token 切分（`update_tokens_for_pcp`）。
  - `build_batch_req_id_to_cp_size` 用 `req_index < num_cp_request` 区分 CP/短。
  - attention / positions / slot_mapping / 子组采样对齐都基于这个"前 CP、后短"的连续布局：CP 区走切分 query + 子组共识/采样对齐；短区走普通 decode。
- **整体链路**：客户端路由长→CP子组 / 短→单引擎 → 各引擎 `CPAwareScheduler.schedule` 出 batch（可能长短混排）→ `reorder_batch_to_split_cp_and_normal` 把 CP 排到前 → PCPManager 按 num_cp_request 前缀对 CP 区切分/共识、短区普通 decode → 逐拍全 DP metadata all_reduce 强制所有引擎对齐。