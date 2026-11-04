# LongBench-v2 prefill 对照实验：脚本与配置审阅稿

日期：2026-11-04（Asia/Shanghai）。状态：**等待脚本/配置内容确认；未起服，未压测。**

## 已确认的输入与服务参数

- 权重：/mnt/share/weight/DeepSeek-V4-Flash（DSV4 Flash）。
- 使用完整 prompt 不超过 131072 token 的 LongBench-v2 原始全文样本，不截断。
- CPU 筛选得到 316 条，排除 187 条；保留原始顺序、context、问题和选项。
- 筛选后完整 prompt 长度：10285～130285 token。max_model_len=131073 为输出1 token留出空间。
- 8 卡 DP8 / TP1 / EP8；服务 max-num-seqs=16；chunked prefill；max-num-batched-tokens=8192。
- off / slot 两组统一 KV cache 24 GiB/卡（25769803776 bytes），这是待实际起服验证的候选预算。
- 每个服务对应一份完整 sh；不要求调用者 export 环境变量或传入参数。
- aisbench 并发16、max_out_len=1、流式性能测试、--debug。

## 本次提交确认的具体设计

1. 接口采用流式 /v1/completions，沿用 aisbench 的 LongBench-v2 原始 prompt 模板，使用模型原生专家路由。
2. Slot 开启 REAL_T、三项优化 PREFETCH_RELAXED / NO_RECONCILE / SKIP_DUP；持久专家缓存、提前预测预取、独立流预取、KEEP_MC2均关闭。每卡同形状层共享32个工作槽位。
3. 两组统一 enforce-eager，并关闭 prefix cache。模拟路由及上轮诊断环境变量由脚本清理。
4. 已确认重复方案：两对独立起服，pair01按 off→slot，pair02按 slot→off。每次服务先由 aisbench预热16条，再测3轮完整316条，保留第1轮但不计均值，取第2、3轮；预热不重复于后续轮次。
5. aisbench retry=0、temperature=0、ignore_eos=True、request_rate=0。只做性能测评，不以输出1token评估准确率。
6. 两组正式采样请求共 4服务 × 3轮 × 316 = 3792条，另有64条预热。若用户选择更轻方案，可修改每份压测脚本顶部 ROUNDS 和按计划调整起服次数。
7. 保留 TTFT均值/中位数/P90/P99、输入token吞吐、请求耗时、成功失败数和实际输入/输出token数。主要判读prefill占主导的端到端表现；TTFT仍包含排队与网络等开销。

第4项重复方案已确认；第2项及脚本/配置全文仍待用户确认。确认脚本内容不代表现在开始执行。

## 文件

| 文件 | 用途 |
|---|---|
| start_off.sh | off独立起服入口 |
| start_slot.sh | Slot独立起服入口 |
| bench_off.sh | 已启动off服务的aisbench性能入口 |
| bench_slot.sh | 已启动Slot服务的aisbench性能入口 |
| aisbench_off.py / aisbench_slot.py | 本实验隔离配置，公共预置配置不修改 |
| prepare_dataset.py | 按已完成token审计生成316条全文子集，已在CPU执行 |
| validate_round.py | 校验每轮316条成功、输出1token、输入token数与审计一致 |
| data/selection_manifest.json | 冻结样本ID、原索引、长度、数据hash |
| length_audit.csv / length_summary.json | 503条完整prompt的tokenizer审计 |
| VALIDATION.txt | 静态检查与CPU合成数据验证记录 |

四份sh顶部均有 SCRIPT_REVIEWED=0。内容确认后才会改为1，当前直接运行会在启动服务/发送请求前退出。其余调优参数都直接在对应sh中修改，aisbench的模型和数据配置在对应py文件中修改。

## 按已确认两对方案的手工执行顺序（现在不执行）

1. 确认8卡空闲、端口8444可用；把四份sh的PAIR_LABEL保持pair01。
2. 在服务终端运行 start_off.sh，在压测终端运行 bench_off.sh。
3. 停止该服务，确认其子进程和NPU资源释放，再运行 start_slot.sh 与 bench_slot.sh。
4. 将四份sh内 PAIR_LABEL 改为pair02，先Slot后off，再各测3轮。每次均为独立重启。
5. 两组共用8444端口、相同输入和参数，依次运行。两个 served-model-name 不同，用于防止客户端连接错模式；不改变权重或计算配置。

服务max-num-seqs=16是每个DP engine的调度上限；aisbench batch_size=16控制本实验全局在途请求数。DP8不把单条请求的KV分片到8卡。

## 原始输出保留方式

服务日志：data/services/<pair>-<mode>-<time>-<pid>/service.log。
压测结果：data/aisbench/<pair>-<mode>-<time>-<pid>/round_<n>/。
每轮 --debug 的 stdout/stderr 完整保存到 aisbench_debug.log。aisbench自身保存的配置、逐请求 *_details.jsonl、关联SQLite .db / db_data、失败明细、统计CSV/JSON/HTML均保留在整份 aisbench work_dir 中；只复制JSONL可能缺失被数据库引用的时间数组，因此不拆分原始目录。

validate_round.py 在每轮结束后校验完整性。服务实际token数与CPU审计不一致、存在失败、输出不为1、ID重复或缺失时，脚本停止并保留所有日志。正式采样前应先核对服务tokenize/usage口径及最长样本，确认131073与24GiB预算有效。

## 已完成与尚未完成的验证

已完成：四份sh的bash -n；Python AST；两份mmengine配置真实解析；316条数据的原文字段及顺序核对；合成CPU文件验证原始结果校验器的正反例。原始LongBench-v2/data.json未修改。

未执行：服务启动、动态库链接验证、24GiB KV分配、130285-token最长请求、并发16的运行时显存和aisbench实际请求。因此目前不承诺起服或运行时无OOM，也不自动降参数；后续实验若失败，保留错误并分析。

## 脚本与配置全文

以下内容由当前磁盘文件直接生成，供逐项确认。
