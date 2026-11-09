# 118 DP8 LongBench-v2：最新Slot优化版对照

用户确认：仅一对独立起服Off→Slot。每服务最长样本校验、16条预热、1轮316条原始全文；输出1 token、并发16、131073最大长度、8192 chunk、24GiB/卡KV，DP8/TP1/EP、eager、关闭prefix cache。

Slot：REAL_T、既有三项优化、持久缓存0、副流2、提前预取0、MTE。额外开启VLLM_ASCEND_MOONEP_DISPATCH_DTYPE=mxfp8及VLLM_ASCEND_MOONEP_FUSED_PUBLISH=1；Home GEMM模式3关闭，combine仍BF16传输、FP32加权累加。Off不启用MoonEP。

沿用已确认的原文模板修复。aisbench perf --debug；保留原始日志、逐请求输出、模式/优化审计、源码和库SHA。仅一对是探索性结果。

本目录使用原仓及默认包目录，无隔离源码副本。新功能库的部署清单与构建记录保留。两项新优化此前通过小算子验证，完整模型集成在本轮起服/请求中验证。

已完成授权的一对实验。结果：/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261106_dp8_longbenchv2_mxfp8_publish/data/executions/run_20261106_190256/RESULT.md。两组均316/316成功，8卡已释放。首个缓存失败尝试已排除。
