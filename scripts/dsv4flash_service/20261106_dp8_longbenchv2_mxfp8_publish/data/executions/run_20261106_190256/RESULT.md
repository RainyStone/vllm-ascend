# 118 DP8 LongBench-v2：MXFP8 dispatch + 融合publish + prefetch副流

仅一对独立起服，Off→Slot；每服务16条预热后正式测316条；原生路由、并发16、输出1 token、128K全文上限、8K chunk、24 GiB/卡KV。

Slot显式开启VLLM_ASCEND_MOONEP_DISPATCH_DTYPE=mxfp8、VLLM_ASCEND_MOONEP_FUSED_PUBLISH=1、MOONEP_PF_STREAM=2；缓存与提前预取关闭、MTE、Home GEMM模式3关闭。Off未启用MoonEP。

|模式|平均TTFT(ms)|TTFT相对Off|输入吞吐(token/s)|吞吐相对Off|
|---|---:|---:|---:|---:|
|Off|14463.9|0.00%|62507.4|0.00%|
|Slot|14384.5|-0.55%|62426.7|-0.13%|

TTFT变化为负表示变快；吞吐变化为正表示提高。本次仅一对，尚未显示明确吞吐收益，不能拆分归因到某个优化。

两组316/316成功，每组输入18,518,962 token、输出316 token，最高并发16；完整prompt及输入token数校验通过。8个rank的MXFP8/MTE日志、副流记录、融合publish库映射及开关检查通过，正式请求结束后再次校验。

首token文本299/316一致、17条不同；不等于完整生成任务准确率，当前单对不能判定差异来自量化迁移或运行数值波动。

最初一次Off客户端因容器根盘100%导致数据缓存加载失败，未生成有效压测结果，已排除；缓存迁到共享盘、临时文件转/dev/shm后完整重跑本对。failed attempt位于failure_evidence/off_dataset_cache_full。

两项新库从原仓全新编译并安装到原仓默认包目录；未更换editable源码指向、未修改功能源码、未提交或push。构建与部署见../../build_libraries.log、../../LIBRARY_INSTALL.json；源码及库SHA见SOURCE_PROVENANCE.json。

debug日志和原始逐请求JSONL/SQLite见RAW_LOG_INDEX.json及results.json的client_dir。最终服务已停止、8卡释放、8444端口可绑定。FINAL_VALIDATION.json保存最终审计。

[与此前仅副流实验的对照](COMPARISON_STREAM_ONLY.md)
