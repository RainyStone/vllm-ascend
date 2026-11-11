# 124 单机 8 卡 LongBench-v2 4K dispatch 对照

本目录已准备，等 124 八卡空闲并收到执行通知后起服。编译与主机侧校验无需使用 NPU。

## 配置

沿用同机历史 pair03。DeepSeek-V4-Flash，DP8 TP1 EP，单 DP 4096 token，chunked prefill，并发 16，输出 1 token，24 GiB KV/卡，最大模型长度 131073，关闭 prefix cache。使用同一份 316 条未截断的原始全文样本和同一套 AISBench 请求格式。每次独立起服先校验最长样本，再预热 16 条，最后跑一轮正式数据。保留 AISBench debug 日志、原始请求结果、服务日志及八 rank 路径校验。

| 脚本目录 | 组别 | Dispatch | 配置差异 |
|---|---|---|---|
| mxfp8/formal | Off | 原生 MoE | 禁用 MoonEP |
| mxfp8/formal | Slot | MXFP8 | 启用新版按有效行解包 |
| bf16_copy/formal | Slot | BF16 | MOONEP_DISPATCH_IN_SHARD=0 |
| bf16_zero_copy/formal | Slot | BF16 | MOONEP_DISPATCH_IN_SHARD=1 |

三个 Slot 组都用 MTE、prefetch 副流 2 和融合 publish，关闭持久缓存，保持 Real-T 与已有优化设置。BF16 两组只改变 dispatch zero-copy 开关。MXFP8 组的结果与历史同机 MXFP8 对照一起解读。BF16 两组之间的变化用于估计 zero-copy 收益。

## 构建和执行

算子库编译产物在 native/，对应 ascend-moonep 合并提交 4d681d0。已记录库的 SHA-256、导出符号和 ABI。启动 MXFP8 前，控制脚本检查 124 的八卡与端口、代码版本和数据指纹，将旧 wire 库按校验和备份，并把已验证的新库原子安装到原 ascend-moonep 仓。服务通过原仓加载 Python 和 native 库。

获准运行后，通过 run_experiment.py 的 --plan 和 --execute 选择实验。mxfp8 执行 Off、MXFP8 Slot。zero-copy-pair 执行 Off、BF16 zero-copy Slot。zero-copy 执行 Off、BF16 copy Slot、BF16 zero-copy Slot。both 执行四组。起服和 AISBench 的参数各自写在对应 sh 和配置中。通过校验的结果写入 FORMAL_RESULTS.md 和 FORMAL_RESULTS.json。
