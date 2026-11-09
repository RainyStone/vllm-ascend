# 124 DP1 TP8 LongBench-v2 预检结果

- 124 的八卡均空闲后启动 Off。八个 TP/EP worker 都是 DP rank 0，Off 的运行参数、worker 环境及库映射审计通过。模型已加载，API ready。24 GiB/卡 KV 配置的服务日志显示容量为 464,278 token。
- 最短和最长原文 prompt 的 /tokenize 检查通过。最长样本 130,285 输入 token、1 输出 token 的首个推理请求返回 HTTP 500。
- 第一项设备错误来自 DSV4 Flash 预填充的 AICPU 算子 KvQuantSparseAttnSharedkvMetadata。设备 errcode 22007，主机 ACL 507018。模型总共有 64 个 query head。DSA metadata 传入本 rank 的 64/8=8 个 query head。当前仓内该算子的 AICPU 参数校验只接受 64 或 128 个 query head，足以解释本次失败。错误发生于 attention metadata 构建，Off 配置下尚未进入 MoonEP。
- Off 正式计时未开始。Slot 和两轮逐 rank 负载补采均未启动。实验进程已退出，八卡及端口已释放。
- 保留 formal/off/service_console.log、formal/off/bench_console.log、worker_plog/ 与 device_log/ 的八 rank 原始失败日志，以及 SOURCE_PROVENANCE.json。
