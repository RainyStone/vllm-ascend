---
name: dycp-scheduler-rollback-semantics
description: DyCP scheduler 侧(CPAwareScheduler)调度与回滚全量语义:CP 共识 object 化、soft_rollback/hard_rollback/降级、emit-as-NEW 撤回 advance、保序、步进节奏守恒、dycp all_gather 配对死锁根因。排查 scheduler 侧卡死/死锁/输出错位时必读(对照 dycp-design-principles 的方案原理)。
metadata:
  node_type: memory
  type: reference
---

# DyCP scheduler 侧调度与回滚语义

记录 `CPAwareScheduler`(vllm `vllm/v1/core/sched/cp_aware_scheduler.py`)的 CP 共识、回滚、emit、步进对齐等调度语义,以及卡死/死锁的根因与修复。本文自包含、看文档即懂,不依赖日志恢复。方案层面拓扑/路由/切分计算见 [[dycp-design-principles]]。

## 1. CP 共识机制(object 版)

`CPSyncProtocol.sync_schedule_confirm`:子组(`dycp_group`,=dycp_size 个相邻 CP rank,可>2、上界全 DP,各子组独立 ProcessGroup)内**一次 gloo all_gather_object** 交换各 rank 的 `{req_id: status}` dict,本地按 **req_id 取三态 MIN** 合并。

### dict 结构
`{"__scheduled_cp__": local_scheduled_cp, "<req_id>": <status>, ...}`。
- `__scheduled_cp__` 是保留键(代替旧固定槽位机制的标志槽[0]),=本 rank 本步是否**调度到**长(1/0)。注意名字从 `local_has_cp` 改为 `local_scheduled_cp`,表"本步是否调度到"而非"是否持有"(active 有但本步未调度到=0)。
- 其余 key=req_id,value=该 req 本端三态状态。

### 三态与来源
`SCHEDULED=2 / NOT_SCHEDULED=1 / PREEMPTED=0`。PREEMPTED 源自 `_preempted_this_step`(本 rank schedule 内抢占过该 req);SCHEDULED=本端本拍排到;NOT_SCHEDULED=本端 active 有但本拍未排到。

### 缺 key 语义(关键)
某 req 不在某端 dict 中(未收到 add / 已 finish 清出)→ 合并时一律按 `NOT_SCHEDULED` 取。**不可误判成 `PREEMPTED`**(仅显式投 PREEMPTED 才算硬回滚危险值)。故"一端 SCHEDULED / 另一端还没收到该 req"时 MIN=NOT_SCHEDULED→软回滚,语义="peer 没排到它"。

### 阶段2协商一并合并
`__scheduled_cp__` 全子组 MIN = `subgroup_all_schedule_cp_request`(=1 当且仅当子组所有端本拍都调度到长;=0 表示至少一端本拍没排到长)。驱动 §4 降级:本端含长但子组不全含长→全批剔除本端 CP 对齐两端 num_cp。

### 返回合约
`(confirmed, soft_rollback_ids, hard_rollback_ids, subgroup_all_schedule_cp_request)`,下游 `_degrade_cp_from_output`/`_soft_rollback`/`_hard_rollback`/dummy 门控零改动。

### 为何按 req_id 而非槽位 position(object 化根因)
旧固定槽位 tensor+`all_reduce` MIN 机制:每个 rank 把 active CP 请求按 sorted 顺序写进固定槽位数组(index 1~32 + 标志槽 0),`all_reduce` MIN 按**数组相同 position** 取。这要求各 rank `sorted(active_ids)` 同 position 是同一 req_id。但 `add_request_async`(core_client.py)在子组内是**串行 await send**:owner(cp_rank0)先 `await` 落地、再逐个 `await` non-owner。叠加 ZMQ 入队+EngineCore 异步消费,**owner 与 non-owner 收到 add 的拍存在到达窗口**。窗口内两端 active 集合不一致→sorted 后同 position 不是同一 req→共识把两个不同 req 的状态投到同一槽 MIN→**position 错位**(误判 confirmed/soft/hard,致两端 num_cp 不对称→死锁)。确定性注入开关 `_dycp_probe_nonowner_delay_s` 只是把这个微秒级窗口放大到确定性复现,关掉它窗口仍在。

按 req_id 合并彻底不依赖 position:**并集无截断**(旧 `_MAX_CP_SYNC_SLOTS=32` 超出即丢共识不进 soft/hard,两端 num_cp 可能不对称)、**不要求各端同 N**、**不要求同 key 集合**,从根消除错位与截断。**代价**:每拍共识付 gloo object 序列化开销(毫秒级),共识在 `schedule()` 每 step 必经;该权衡换来按 req_id 精确对齐的治根性。

## 2. soft_rollback(单 req 语义)

触发:共识后某 req 的 min≥`NOT_SCHEDULED`(本端排到但子组内 peer 没排到)。处理"本端排到、peer 没排到"的 CP 请求——剔出本拍 output + 回退本拍虚推进度 + 保留已算 KV/状态/队列位。公共子方法 `_rollback_cp_req_from_output`(soft/降级共用,消除"两套剔除代码"分叉):

- **pop `num_scheduled_tokens` + 扣减 `total_num_scheduled_tokens`**;`_remove_req_from_output` 同步清理 scheduled_new/cached、cp_rank_scheduled_tokens、cp_req_id、req_id_to_cp_size、num_cp_request 等 output 侧计数。
- **回退本拍虚推 `num_computed_tokens = max(0, num_computed - num_scheduled)`**:soft/降级都发生在 `execute_model` 之前,本拍 CP 段实际未执行,`_update_after_schedule` 的 advance 只是把 `num_computed_tokens += num_scheduled` 当成"调度乐观值"提前推进。不回退会导致严重后果——base scheduler 下拍算 `num_new_tokens = num_tokens - num_computed_tokens` 时,本拍已虚推的进度让 num_new=0,判定为"该请求已算完不用再排"→SKIP,于是 prefill 再不被重排、再不 execute、sample 再不做(max_tokens=1 的 sample 本该在 prefill execute 拍采样)→active_cp 永远排不空→引擎 idle 死循环卡死。回退只减本拍 num_scheduled,**保留之前拍真实计算的进度**。
- **不调用 `kv_cache_manager.free()`**:**已算的物理 KV 块不释放**(这是"已算 KV 不动"原则),否则会释放该请求全部历史块、违背 soft 语义、且破坏正在向 D 传输的 KV。
- **不降级 status、不移出 running**:**状态原样保留**(保持 RUNNING),下一拍重排续算。
- **本拍新分配的尾部块保留**:回滚在 execute 前发生,这些块尚未写入、不属"已计算 KV",下一拍真正执行时写入并复用。FullAttentionManager 对 running 请求的 `get_num_blocks_to_allocate`/`allocate_new_blocks` 在块已足够时均 `max(...,0)` 钳到 0,故保留不触发负分配或断言。

## 3. hard_rollback(全 rank 抢占语义)

触发:共识后某 req 的 min==`PREEMPTED`(子组内至少一端抢占/驱逐了该 req)。全 rank 统一处理:
- `num_computed_tokens = 0`(完全重置进度)。
- **release KV**(`kv_cache_manager.free`,释放该请求全部 KV 块)。
- 移出 running。
- 放回 **waiting**(用 `WAITING` 不用 `PREEMPTED` 状态:CP rollback 在 execute_model 之前、worker 从未见过该 req 的 KV 状态,若用 PREEMPTED 会让 base 当 resumed 请求下发→worker `self.requests[req_id]` KeyError)。
- 仍留 active_cp_requests,下拍重排。

PREEMPTED 源自 `_preempted_this_step`(本 rank schedule 内做过 `_preempt_request`);hard_rollback 让所有 rank 一致重置,下拍从头 prefill。

## 4. 降级 `_degrade_cp_from_output`(全批对齐 num_cp)

触发:`local_scheduled_cp=1 且 subgroup_all_schedule_cp_request=0`(本端本拍排到了长请求,但子组并非所有端都排到长——即"本端含长但子组不全含长")。处理:把本端本拍**所有** CP 请求**批量** soft-rollback,对齐子组两端 `num_cp` 同 0(两端都不进 per-layer dycp all_gather)。

- 与 soft_rollback 的区别只在**触发条件与覆盖范围**:`soft_rollback` 逐个处理(共识后 per-req,某 req 本端排到 peer 没排到);**降级全批**(阶段2协商后,本端虽有长但子组不全有长,把本端全部 CP 一次性剔出)。**底层执行语义完全一致**(都走 `_rollback_cp_req_from_output`:剔 output+回退 num_computed+保留 KV/running)。
- 为何要降级:不降级则本端进 per-layer dycp all_gather、peer 不进→**两端 num_cp 不对称→per-layer 集合通信不配对→worker hang/全 DP 互锁**(§8 配对死锁根因之一)。降级让两端 num_cp 同 0,都不进 dycp all_gather,消除不对称。
- 互斥安全:降级场景至少一端空拍(其 scheduled_cp=0、dict 不含该 CP req→合并取 NOT_SCHEDULED),该端不投 PREEMPTED→hard_rollback_ids 空;soft_rollback_ids 本会含本端全部 cp,但已被降级从 output 剔除(num_scheduled_tokens 已 pop),`_soft_rollback` 的 SCHEDULED 分支判 not in num_scheduled 走 else 无害空转,不重复回退。

## 5. emit-as-NEW 撤回 advance

背景:首调被 soft_rollback/降级 的 CP 请求(`was_first_schedule=True`:这拍是首次调度、worker 从未收到该 req 的 `new_req` 注册)不能按"留 running 续算"处理——下拍 base 会以 `scheduled_cached_reqs` 下发,worker `self.requests[req_id]` 查无→KeyError。处理:留 running 原位(保序,见§6)+ 标 pending(`_cp_reqs_pending_new_emit`),下拍子组对齐时 `_emit_pending_new_cp_as_new` 把它从 `scheduled_cached_reqs` 挪到 `scheduled_new_reqs` 发出,worker 走 new 注册路径。续算回滚(`was_first_schedule=False`,worker 已注册)不标 pending,仍走 cached 续算。

### 关键修复:挪动时必须撤回本拍虚推 advance
`_emit_pending_new_cp_as_new` 在 `schedule()` **末尾**调用,此时 base 的 `_update_after_schedule` 已把 `req.num_computed_tokens += num_scheduled` 这拍虚推。`NewRequestData.from_request` 会沿用 advance **后**的值(如已在前面拍虚推到 9)。worker NEW 路径按 `num_new = prompt_len - num_computed_tokens` 取 token:**带 9→算出 0 个新 token**(全 -1 slot 不写 KV),而 peer 这拍走 cached-cont 写真实 slot→两端同拍 num_new 不对称→**per-layer dycp all_gather / KV 写入发散→层间集合通信错位死锁**(§8 配对死锁根因之一)。

修复:`num_scheduled_tokens_this_step = num_scheduled_tokens.get(req_id, 0)`;`num_computed_tokens_before_advance = max(0, num_computed - num_scheduled)`;置 `new_req_data.num_computed_tokens = num_computed_tokens_before_advance` 并回退 `req.num_computed_tokens` 同值(撤销 advance)。使 emit-as-NEW 与正常 fresh-NEW 语义一致(都是 advance **前**值,首调 fresh 起拍点正常无 prefix 命中时为 0),worker 据此算出真实 num_new 写真实 slot,与 peer 对称。was_first_schedule 的 pending 请求 worker 从未注册、无真实 KV,advance 前值即 fresh 起拍点,撤回安全。`num_scheduled_tokens`/total/CP 元数据不动(与 emit 挪动语义一致)。

## 6. 保序

首调被回滚 CP 请求**留 running 原位**(不回 waiting queue),保证长请求处理顺序与接收顺序一致——原 scheduler 即使抢占也保证请求处理顺序与接收顺序一致,CP 回滚不能破坏这点。下拍子组对齐时由 `_emit_pending_new_cp_as_new` 以 new 发出,running 位置不动→无持续失序。hard_rollback 放回 waiting(属真抢占/重置、语义不同,不在此列)。

## 7. 步进节奏守恒与空拍对齐

- **共识对齐的不是"两端排到相同数量的请求"**,而是"本拍两端是否都排到同一长请求"。两端排到不同的长请求集合是正常的(各自 budget/WAITING_FOR_REMOTE_KV 等本地状态不同);只要共识后对"两端都排到的长请求"达 confirmed、"本端排到 peer 没排到的"进 soft_rollback/降级即可。
- **空拍端也必须进共识**(发 `{"__scheduled_cp__":0}` 参与 gather),否则忙端(有 active_cp)的共识一直等空端→**hang**。object 版形状不对齐安全(不要求 key 集合一致),但调用对齐(各 rank 都调 all_gather_object)仍必须满足。旧 tensor 机制下空拍还曾因"跳过共识+忙端调共识"配合当时独立的 align all_reduce 形状 32≠1 错配、抢同一 dycp_group→gloo Connection reset 崩。object 化后空拍端发 `{"__scheduled_cp__":0}` 参与同一次 all_gather_object,根治此 hang/崩。

## 8. dycp all_gather 配对死锁根因(共因与各触发场景)

**共因**:子组两端 `num_cp` 不对称(一端进 per-layer dycp `all_gather`、一端不进)→ 层间集合通信错位→60s `shm_broadcast` timeout 死锁。per-layer dycp all_gather 在 attention 每层 forward 内(`mla_cp.py`),按 `num_dycp_reqs` 决定是否进入;两端 `num_cp_request 不一致即撞此。

各触发场景与对应修复(按 § 交叉引用):

- **position 错位**:旧槽位 + add 异步到达窗口→同 position 非同 req→两端对同一"槽位"共识的是不同 req,决策与剔除错位→两端 num_cp 不对称。→ §1 object 化按 req_id 精确合并消除。
- **emit-as-NEW 带 advance 后 num_computed**:emit 端 worker 算出 0 个 new token 不写真实 slot(全 -1),peer 走 cached-cont 写真实 slot→两端虽都进 dycp all_gather 且 shape 配对,但 KV 写入发散→层间错位。→ §5 emit 撤回 advance 修复(emit 取 advance 前值,worker 算真实 num_new 写真实 slot)。
- **两端都含长但某 req 不齐 + 降级缺失**:本端含长、peer 也含长(各自别的长请求),但某 req 只一端排到→共识前两端 num_cp 含该 req 不一致;若无阶段2降级则两端都进 dycp all_gather 但该 req 段不对称。→ §4 降级(subgroup_all=0 时全批对齐 num_cp 同 0)+ §1 按 req_id 共识覆盖(该 req 进 soft_rollback)。

排查信号:日志里 subgrp 各 DP 的 dycp all_gather 入/出口探针层数不对齐(dn_ag_enter/exit);Probe/ag_merge 探针看各 req_id 合并决策是否两端一致。

## 9. 死代码与历史(已清理)

object 化后以下旧机制代码删除(非废弃保留,直接删):
- `sync_empty` 方法(全仓无调用,死代码)。
- `_confirm_tensor`(旧 tensor 槽位数组,object 版不用)。
- `_MAX_CP_SYNC_SLOTS = 32` 固定槽位常量(并集无截断,不再用)。
- 过时 TODO(`cp_world_size 参数未用到`,object 版已用)。

`local_has_cp`→`local_scheduled_cp` 改名(语义精确化:"是否含 CP"→"本步是否调度到 CP",体现 active 有但本步未调度到=0 的区别)。
