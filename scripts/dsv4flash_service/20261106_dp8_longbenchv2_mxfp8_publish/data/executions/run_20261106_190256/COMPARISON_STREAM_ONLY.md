# 两次LongBench-v2单机DP8实验对照

每组均使用各自Off基线；变化率=(Slot/对应Off−1)×100%。TTFT负值表示变快，吞吐正值表示提高。

|Slot配置|Off TTFT(ms)|Slot TTFT(ms)|TTFT相对Off|Off输入吞吐(token/s)|Slot输入吞吐(token/s)|吞吐相对Off|
|---|---:|---:|---:|---:|---:|---:|
|仅prefetch副流|14448.5|14724.2|+1.91%|62993.0|60949.9|-3.24%|
|副流＋MXFP8 dispatch＋融合publish|14463.9|14384.5|-0.55%|62507.4|62426.7|-0.13%|

相对各自Off的TTFT变化改善2.46个百分点；输入吞吐变化改善3.11个百分点。

若只比较两次Slot绝对值，TTFT变化-2.31%，输入吞吐变化+2.42%；这不是相对Off归一后的变化。

两次均单对独立起服，每服务16条预热后正式测1轮316条；原生路由、输出1 token、并发16、8K chunk、128K全文上限、24GiB/卡KV、缓存关闭、MTE、副流2。两次日期、SDK构建及其他源码优化有差异，不能仅据跨实验数字把差异全部归因于两个新开关。
两组内部均316/316请求成功。没有启动任何新服务或请求。

来源：
- 仅prefetch副流：/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261105_dp8_longbenchv2_cache8_pfstream/data/executions/run_20261105_132106/results.json
- 副流＋MXFP8 dispatch＋融合publish：/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261106_dp8_longbenchv2_mxfp8_publish/data/executions/run_20261106_190256/results.json
