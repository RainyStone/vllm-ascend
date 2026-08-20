---
name: domain-dycp-design-principles
description: "domain_implementation fork 的\"domain 方案\"中心化 DyCP——CrossDPScheduler域内单点调度+RequestManager路由取代子组共识、DPCoordinator跨域波次协调(MoE专用,非MoE仅stats)、计算面dycp复用PCP切Q+KV all-gather组换dycp_group、domain slot映射、采样broadcast、cudagraph按num_dycp_reqs特化；含已知未完成区"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 11be6c25-06b1-4485-8efc-ff861ef379d2
---

# domain 方案 DyCP 实现（中心化）

domain 方案是 DyCP 的**中心化**实现（`domain_implementation` fork，含 vllm + vllm-ascend 双 checkout），取代旧扁平 dycp（CPAwareScheduler 子组共识/DP busy loop）。旧扁平版原理见 [[dycp-design-principles]] 作对照（注：CPAwareScheduler 在本 fork 已不存在，被 `CrossDPScheduler` 取代）。mooncake KV 拉取的 domain DyCP 切分见 [[mooncake-connector-pd-transfer-principles]] §8。分析须基于代码+日志取证，见 [[vllm-debug-evidence-based]]。

## 0. 拓扑
- domain=CP子组。`dp_per_domain`(parallel.py:338)=域内CP world size=`cp_world_size`(cross_dp_scheduler.py:228)。域数=`dp_size//dp_per_domain`(engine_count)，每域一进程(domain_rank=engine_index)。
- CP通信组=`get_dycp_group()`(parallel_state.py:1561建_DYCP组,聚域内dp rank)。CP rank=dycp组内 rank_in_group=域内dp rank。
- `request.cp_ranks`(request.py:181)=该req落到的dycp rank集合(短=1,长=域内全部)。
- SchedulerOutput加字段 cp_rank/cp_rank_to_req_id/cp_rank_scheduled_tokens/num_cp_request/req_id_to_cp_size(output.py:216,244-253)。

## 1a. 域内中心化调度（CrossDPScheduler+RequestManager）
- 单 CrossDPScheduler 持域内全局视图。`schedule()`一次产 list[SchedulerOutput](长=cp_world_size,每rank一份)(cross_dp_scheduler.py:789,1402)直接下发,**无逐拍子组共识 collective**(all_reduce/all_gather_object 都不上,区别旧扁平版 CPAwareScheduler 的 all_gather_object by req_id 子组共识)。
- 长短分流进调度器:LongShortRequestQueue 按 long_request_threshold 分(request_queue.py:235)，长请求受 num_cp_seqs(=max_long_requests) 限额。
- 路由 RequestManager.select_dp(:91-114):长→域内全部cp_rank(:101);短→greedy取(rank_budgets,-num_req_per_dp)最优单rank(:114);已绑cp_ranks复用。写 request.cp_ranks=selected(:1245)。
- KV块:CrossDPKVCacheManager(cross_dp_kv_cache_manager.py)每cp_rank独立DPBlockPool,round-robin把token/块均分各rank;free/get按cp_ranks跨rank汇总。⚠️走 NoPrefixCache(prefix cache未实现,:308-316,:499)。
- 引擎 step_domain(core.py:1187)把 list[SchedulerOutput]整体 execute_model;gpu_worker按cp_rank取slice(:776)。run_domain_engine_core(core.py:1221);dp_per_domain>1切CrossDPScheduler(core.py:136)。

## 1b. 跨域波次协调（DPCoordinator）
- DPCoordinator/DPCoordinatorProc(coordinator.py:22,121)独立进程,持全局 current_wave/engines_running(:168-169)。launch_domain_core_engines 在 `needs_dp_coordinator and domain_rank==0` 时启(utils.py:1273-1288)。
- needs_dp_coordinator(vllm.py:450-470):dp>1且(MoE或非external_lb)。MoE时 enable_wave_coordination 走波次协调;非MoE internal/hybrid LB 仅 stats 收集发布。
- 三类输入:front-end新请求→engines_running=True+广播 START_DP_WAVE(:402);engine stats 上报(:321);wave_complete(rank0域上报,:366)/陈旧wave→重广播。
- 域引擎 run_busy_loop(core.py:1969):处理 START_DP_WAVE(:1941);每32步 _has_global_unfinished_reqs all-reduce 判完成(:2023);全局空闲rank0上报 wave_complete。
- ⚠️过时/约束:a) utils.py:1252 "domain coordinator is not implemented yet" 注释**已过时**(其下1273-1288即创建coordinator);b) 波次协调 MoE专用,但 domain 暂不支持 MoE(core.py:1263 TODO),实际非MoE下coordinator仅stats。

## 2. 计算面CP（dycp，复用PCP DualChunkSwap，组换dycp_group）
- CP世界数=dp_per_domain(gpu_model_runner:439,model_runner_v1:310);cp_rank=dycp_rank(:312,440)。
- **仍是"切Q head/tail+KV子组all-gather还原"(PCP DualChunkSwap)**,载体 pcp_group→dycp_group。common_pcp_size=dycp_size if dycp>1 else pcp_size(mla_cp.py:105)二选一,切分代码PCP/dycp复用。
- 切Q:PCPManager.update_tokens_for_pcp 对前 num_dycp_reqs 个CP请求 DualChunkSwap(pcp_utils.py:517,572);PCPManager含 dycp_world_size/rank(:89)。
- KV:prefill mla_preprocess_prefill get_dycp_group().all_gather(kv)+pcp_allgather_restore_idx还原(mla_cp.py:830-838);decode各rank各存1/dycp份interleave,不gather KV,改 attn_out+lse all-gather合并。
- dycp可与dcp叠加(common_pcp_size*dcp_size*dycp_size,model_runner_v1:339),约束 assert not(pcp>1 and dycp>1)、assert not(pp>1 and pcp>1)(mooncake_connector.py:1162,1166)。

## 3. batch重排
- reorder_batch_to_split_cp_and_normal(backends/utils.py:867):判CP改用 `cp_rank_scheduled_tokens[req_id]>1`(=len(request.cp_ranks),cross_dp_scheduler:1003)(:878-887);CP排前且按 req_id 字符串稳定排序保跨rank一致(:893,注释不用hash());触发仅 num_cp_request>0(gpu_model_runner:1000)。apply_permutation O(n)(:899)。

## 4. block/slot寻址
- 新增 compute_domain_slot_mapping(vllm block_table.py:221;ascend block_table.py:148):前 num_dycp_reqs 个用 virtual_block_size=block_size*dycp_world_size,按 position//interleave_size%dycp_world_size==dycp_rank 标本rank token(非本rank写-1);尾部DP请求回退pcp*dcp/朴素。
- scheduler块膨胀因子=dp_per_domain(每rank独立pool round-robin),与worker dycp_world_size=dp_per_domain**同因子对齐**(每worker SchedulerOutput按cp_rank取slice只含本rank block_ids)。
- ⚠️两套口径并存(未完成):kv_cache_interface.max_memory_usage_bytes 与基础Scheduler仍只按 pcp*dcp 算KV预算(kv_cache_interface.py:117-122注释"Do not include dp_per_domain here until DyCP KV ownership fully supported");get_total_cp_world_size()也只返pcp*dcp(cp_utils.py:57)。物理配额未按dycp缩减——嫌疑区(易超配)。
- CrossDPKVCacheManager.allocate_slots:len(cp_ranks)!=1且!=cp_size直接 NotImplementedError(:422)——部分副本CP未支持。

## 5. attention分段执行
- mla_cp.py build→split_attn_metadata(:1289)按请求边界切 dycp_metadata(前num_dycp_reqs,CP段)/dp_metadata(尾部短段);forward(:614,665-677)先CP段 _forward_common(dycp_metadata)再短段 _forward_common(dp_metadata),输出拼 oproj_input 不同切片。
- decode对齐 _npu_update_dycp_attn(common_cp.py:122):dycp组 all-gather[out+lse],npu_attention_update_v2 log-sum-exp合并,只合并前 num_dycp_reqs 行(尾部DP行不动,mla_cp.py:1174);dycp_size==1 走旧 _process_attn_out_lse+_npu_attention_update(:1180)。

## 6. 采样对齐
- owner=dycp rank0。_sync_dycp_sampled_token_ids(model_runner_v1:469):采样后对前 num_cp_request 行 broadcast(sync_token_ids,src=0)(:492)。机制从旧"全量logits对齐"收敛为"attn层all-gather+lse合并+采样结果broadcast"。

## 7. cudagraph
- 按 num_dycp_reqs 特化:cudagraph_dispatcher.initialize_cudagraph_keys 枚举 dycp_reqs∈[0,min(max_dycp_reqs,bs)](:196,227),dispatch选图(:249-331);ascend acl_graph GraphKey=(num_tokens,num_dycp_reqs)(acl_graph.py:225),workspace按key缓存。原因:CP段空洞/padding使同token数不同dycp份数需不同图。

## 8. 配置
num_cp_seqs(scheduler.py:149)、long_request_threshold(:153)、dp_per_domain(parallel.py:338)。

## 9. 已知未完成/嫌疑区
- prefix cache域版禁用(CrossDPKVCacheCoordinatorNoPrefixCache,find_longest_cache_hit返0,:499)。
- KV内存预算未按dycp缩减(kv_cache_interface.py:117-122)。
- domain不支持MoE(core.py:1263);波次协调虽实现但MoE路径未达。
- 调度TODO:DCP rank token budget(:813)、长前缀阈(:887)、encoder inputs(:901)、PRIORITY(:952)、PD disagg(:1253)。
- utils.py:1252 coordinator未实现注释过时。
- mla_cp.py:1180 纯DP decode(dycp>1但num_dycp_reqs==0)走 return _v_up_proj 未跨rank lse合并,待确认是否本就不需要。
