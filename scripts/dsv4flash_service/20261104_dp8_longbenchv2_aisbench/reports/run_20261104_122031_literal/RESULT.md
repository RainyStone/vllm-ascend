# LongBench-v2 prefill：off / MoonEP Slot

316条128K以内原始全文，原生专家路由；DP8/TP1/EP8，客户端并发16，输出1token，8K chunk，24GiB KV/卡，两组关闭prefix cache。

每台独立服务测3轮，取后2轮均值。旧run_20261104_114303因输入完整性问题整轮排除。

| 服务对 | off TTFT(ms) | Slot TTFT(ms) | TTFT降低 | off输入吞吐(token/s) | Slot输入吞吐(token/s) | 吞吐提高 |
|---|---:|---:|---:|---:|---:|---:|
| pair01 | 14471.15 | 15363.35 | -6.17% | 62294.51 | 58827.86 | -5.56% |
| pair02 | 14415.20 | 15286.50 | -6.04% | 62239.51 | 58813.24 | -5.50% |

两对等权平均：off TTFT 14443.17 ms，Slot 15324.92 ms；TTFT降低 -6.10%。off输入吞吐 62267.01 token/s，Slot 58820.55 token/s；吞吐提高 -5.53%。负值表示Slot退化。

全部12轮、3792条正式请求通过成功数、输出token、完整输入token数及prompt SHA256校验。每轮输入总数18,518,962 token；纳入对比的8轮客户端最高并发均为16。模式核验文件逐轮保留，8个实际worker的参数/环境/映射库及初始化日志均检查。

原始结果、debug日志、逐请求JSONL及关联SQLite数据库保留在各client_dir；详见results.json和rounds.csv。

首token文本一致性仅作辅助核查，不替代完整生成精度测试：
- pair01: round1 225/316, round2 229/316, round3 227/316
- pair02: round1 228/316, round2 226/316, round3 229/316

同模式round2/round3首token文本一致性：
- pair01_off: 302/316
- pair01_slot: 302/316
- pair02_slot: 301/316
- pair02_off: 303/316

已经观察到首token文本不完全一致；同模式重复也存在差异。差异原因尚未定位，本性能实验不能证明完整生成精度等价。
