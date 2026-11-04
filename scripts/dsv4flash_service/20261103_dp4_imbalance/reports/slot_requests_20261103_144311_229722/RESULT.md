# MoonEP slot短请求检查结果

结论：自然路由下，slot服务成功启动，两个正常短请求均HTTP 200并生成16个token；四个DP/EP rank均有dispatch、combine和prefetch调用增长。

配置：物理卡4,5,6,7；DP4/TP1/EP4；FP8 DeepSeek-V4-Flash；KV cache每卡4GiB；max-model-len=8193；max-num-batched-tokens=8192。关闭SIM_ROUTE，开启诊断计数；本次不做性能测量。

请求1：The capital of France is，5输入/16输出token，输出以Paris开头。
请求2：计算1+1，8输入/16输出token，输出以2开头。完整请求和输出见request_*.json、response_*.json。

证据：service.log中的MOONEP_SLOT_MODE=1 env audit、MoonEP初始化、各worker的moonep-tstats调用；activity_verification.json保存按DP rank复核的前后计数及差值。

检查器最初按逻辑设备npu:0分组，但四个独立DP worker都使用逻辑设备0，导致自动证据检查误报；原始check.json和脚本退出状态保持未改。已修正解析器，排除混行后按Worker_DP rank复核现有日志，结果通过，不重跑请求。

本次只确认短请求可执行且MoonEP通信路径生效，未认证长输入/高并发容量、数值等价或性能收益。此前8K并发16压测仍OOM。
服务已清理、端口8444释放，检查时后四卡无NPU进程。性能实验按用户要求暂缓，等其它卡空闲后另行安排。
