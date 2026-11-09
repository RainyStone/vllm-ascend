# 118 单机 DP8 LongBench：仅开启 prefetch 副流（待通知执行）

当前 Slot 配置：MOONEP_SLOT_CACHE=0、MOONEP_PF_STREAM=2、MOONEP_EARLY_PREFETCH=0、MOONEP_DISABLE_UDMA=1，使用 MTE。
本次仅调整脚本；用户要求等卡空闲并收到启动通知后再执行，不自动等待空闲后开跑。

保持其余参数：DSV4 Flash、DP8/TP1/EP、服务及aisbench并发16、输出1 token、128K上限、316条原始全文、chunked prefill、8192 token预算、24 GiB/卡KV、关闭prefix cache、eager、相同seed和三项原有优化。
Off配置不变；执行顺序仍为Off1→Slot1→Off2→Slot2→Off3→Slot3，每次独立起服只测1轮316条，保留最长样本校验和16条预热；按全部已完成配对数据取均值，每对结束更新结果。

活动入口：start_off.sh、start_slot.sh、bench_off.sh、bench_slot.sh及run_cross.py。
模式校验要求：Off不启用MoonEP；Slot为slot模式、缓存关闭、副流开启，并在8个rank看到副流执行记录。aisbench --debug日志和原始逐请求结果继续保存。

历史记录：
- 目录名保留cache8_pfstream，以维护既有报告链接；当前活动参数以本文件、start_slot.sh和CURRENT_EXECUTION.json的pending_slot_profile为准。
- 原cache8+副流、仅缓存、仅副流的历史运行和故障记录均保留，未改写历史服务快照与结果。
- 本次修改前的活动脚本和元数据备份见data/LATEST_CONFIG_BACKUP.txt。
- 仅副流的128条诊断曾通过，不代表316条完整正式矩阵已经完成。
- 用户已选择暂不应用普通NPU内存替代VMM缓存的候选修改，该修改未应用。
- 下次收到启动通知后，先复查源码版本、8卡和端口状态，处理未完成Slot1目录，再执行新配置；保留Off1原始结果。

本次没有起服、发送模型请求、运行NPU算子、提交或push。

## 20261105_190404 续跑

用户已通知在118续跑。保留有效Off1，依次运行Slot1、Off2、Slot2、Off3、Slot3；缓存0、副流2、MTE。版本与旧失败记录见 /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp8_longbenchv2_cache8_pfstream/data/executions/run_20261105_132106/resume_20261105_190404。

2026-11-05T19:13:50.543496+08:00：仅副流Slot1完成316/316、0失败，8 rank均通过副流审计。按用户要求暂停，未启动Off2，8卡及端口已释放。Off1/Slot1为一对探索性数据，尚未完成三对均值。结果见data/executions/run_20261105_132106/RESULT.md。
