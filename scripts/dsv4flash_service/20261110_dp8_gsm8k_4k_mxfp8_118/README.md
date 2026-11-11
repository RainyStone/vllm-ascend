# 118 单机 8 卡 GSM8K 4K

Off → MXFP8 Slot 各独立起服。使用 DeepSeek-V4-Flash、DP8 TP1 EP、4096 batched tokens、24 GiB KV/卡、eager、chunked prefill、关闭 prefix cache。AISBench 使用 test 全部 1319 条原始问题，并发 16、流式输出 1 token。每个正式服务校验最长样本和路由路径，预热 16 条后测量。Slot 沿用 MTE、prefetch 副流 2、融合 publish、按有效行解包，关闭专家持久缓存。

正式轮不采 rank 数据。诊断轮各自起服，独立回放同一批请求。每步刷新逐 rank 日志，使用回放前后的边界排除最长样本，落盘范围外的尾部步骤不参与分析。峰均比按每个共同 step 和 MoE 层计算，另报按 rank 跨步跨层累加的全局峰均比。Slot 同时报专家原属 rank 的需求和实际执行 rank 的负载。
