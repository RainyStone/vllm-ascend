# 后四卡 DP4：MoonEP off / slot 负载不均衡性能实验

准备日期：2026-11-03。目标：141.61.45.118 的 xzx_moonep 容器。
本次交付仅准备脚本，未启动服务、未执行性能测试，data/ 中没有真实实验数据。

## 已确认的实验口径

- 对照组 off：enable_moonep=false，不设置 MOONEP=1。
- 实验组 slot：enable_moonep=true、MOONEP_SLOT_MODE=1、real-T=1、min_dispatch_tokens=16。
- slot 开启历史三项优化：ASCEND_MOONEP_PREFETCH_RELAXED=1、
  ASCEND_MOONEP_NO_RECONCILE=1、ASCEND_MOONEP_SKIP_DUP=1。
- 仅使用物理卡 4,5,6,7；在该可见卡集合内逻辑 rank 0..3 对应物理卡 4..7。
- DP4 / TP1 / EP4，enforce-eager、chunked-prefill、API server 数为 1。
- 模型 /mnt/share/weight/DeepSeek-V4-Flash，FP8，E=256、K=6、43 层。
- 输入 8192/16384 × max_num_batched_tokens 8192/16384，输出固定 1 token。
- 每轮请求 32，并发 16，request-rate=inf，ignore-eos。
- 每台全新服务先做 2 请求 × 1024 输入的预热；再测同形状 3 轮，丢弃第 1 轮，取第 2/3 轮算术均值。
- 四个路由 × 四种形状 × 两模式 = 32 次起服、128 次 bench 调用（含预热）。
- 每个条件下 off/slot 紧邻执行；按形状交替先后顺序，降低固定执行顺序的影响。
- 两模式同形状使用相同 bench seed：9000 + 输入长度 + 轮次；服务 seed=1024，route seed=0。
- 保留 9 GiB KV cache、gpu-memory-utilization=0.80、max-model-len=20480。
  这些值沿用历史实验，96 GiB A5 上能否起服仍以实际日志为准；脚本不会自动降低负载或变更配置。

## 不均衡比的定义

复用 ../sim/ 的四个路由档案；脚本计算并保存文件 SHA256 与每个 home rank 的目标份额。
s = 最大 home-rank 路由份额 / 四个 home-rank 的平均路由份额（25%）。
这是在原始专家归属下、均衡之前的 rank 负载峰均比；不是单专家峰均比，也不是 slot 后的实测计算峰均比。

| 档位 | 文件 | DP4 下 s（近似） |
|---|---|---:|
| uniform | sim_route_uniform.txt | 1.0000 |
| 4hot | sim_route_4hot.txt | 1.6614 |
| 3hot1r | sim_route_3hot_1rank.txt | 2.4822 |
| 4hot1r | sim_route_4hot_1rank.txt | 2.9841 |

有限 token 数会产生整数舍入；SIM_ROUTE 生成的路由还随 token shape 改变。
off/slot 使用同一档案和 seed，但调度可能改变 shape，所以不将本实验作为数值正确性验证。
不启用 profiling、MoE 负载落盘、slot 统计或 balance 计数，避免给性能窗口加入采样开销。
两模式都将 route 心跳和完整表输出关闭；仍保存服务启动、错误及 MoonEP env audit 日志。

## 文件

- config.json：全部运行参数；当前使用用户确认的完整矩阵。
- RUNS.txt：当前配置生成的 32 项计划清单；执行器以 config.json 实时生成计划，不读取这份清单。
- VALIDATION.txt：本次语法和纯逻辑检查记录，不是性能测试结果。
- run_experiment.py：计划预览、资源检查、顺序起服、预热和压测、清理自身进程、归档。
- summarize.py：从归档 JSON 生成 CSV、JSON 和 Markdown 表格，无需 NPU。
- test_harness.py：仅使用标准库和合成数据检查编排逻辑；不导入 torch、不探测 NPU、不起服。
- data/：仅用于真实运行产物，每次运行使用新的上海时区时间戳目录。

## 使用方法

以下命令应在 xzx_moonep 容器内部执行。

```bash
cd /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261103_dp4_imbalance

# 默认只打印计划，不运行 NPU 查询、服务或 benchmark：
python run_experiment.py --plan

# 仅纯逻辑自检：
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v test_harness.py

# 以后真正运行时，确认后四卡空闲后显式执行：
python run_experiment.py --execute
```

默认省略参数也只打印计划。此交付没有执行 --execute。
如需改参数，可复制 config.json 为独立配置，使用 --config /绝对路径/config.json；每次实际运行都会归档配置。

本机网络默认 enp34s0f1 / 141.61.45.118，服务仅监听 127.0.0.1:8444。
CANN 环境脚本为 /usr/local/Ascend/cann/set_env.sh，执行前读取并校验接口 IP、权重文件和端口。
Python 使用容器内 /usr/local/python3.11.10/bin/python。

## 资源与失败处理

- 只对卡 4..7 判断空闲；卡 0..3 有其他任务不影响判断。npu-smi 查询本身会展示整机。
- 动态解析 npu-smi 的 HBM used/total，兼容本机 98304 MB，不依赖 PCI Bus-Id 文本。
- 所选卡出现任何 NPU 进程、HBM >=12000 MB 或状态不健康时，不启动服务；等待释放最多 300 秒后失败。
- 8444 已占用或同一目录的端口锁被另一个编排器持有时退出，不触碰已有服务。
- 不启动占卡器；不使用 pkill、killall 或按端口杀进程；只终止自己创建的服务进程组。
- 就绪后要求每张所选卡恰好一个 NPU worker；每轮前后保存快照并检查进程集合是否改变。
- 运行期间仍应确保四卡独占。快照不能证明采样间隔内绝无外部任务，因此结果需结合资源使用情况解释。
- 服务退出、等待超时、bench 失败、请求不全、token 数错误或 worker 集合变化时停止整个矩阵并保留失败现场。
- 不自动重试、跳过失败条件或覆盖旧数据。排查后重新执行会建立新的 session；避免混合不同服务状态。
- Ctrl-C / SIGTERM 会进入清理；kill -9 无法触发 Python finally。
- 未安装或应用 tokenizer 20ms 等待窗口补丁；两组沿用当前 vLLM 同一默认实现。
  若以后专门研究调度对齐，应单开实验并同步变更两组配置，不能混入本批数据。

## 数据与统计

实际运行才会建立：

```text
data/<YYYYMMDD_HHMMSS_microseconds>/
  config.json, plan.json, environment.json, pip_freeze.txt
  run_experiment.py, summarize.py, EXPERIMENT.md
  sim_route_*.txt
  <route>_L<input>_B<budget>_<off|slot>/
    metadata.json
    service.log
    bench_0.log ... bench_3.log
    round_0.json ... round_3.json
    npu_before.txt, npu_ready.txt, npu_round_*_before.txt, npu_round_*_after.txt, npu_after.txt
  metrics.csv
  comparison.csv
  summary.json
  summary.md
```

服务与 bench 的完整命令、筛选后的环境变量、路由 hash、进程 ID 和状态写入每个 case 的 metadata.json；
仓库 HEAD 和已跟踪文件修改状态、pip freeze、模型配置与索引 hash 写入 session 根目录。
不读取或复制全部进程环境，避免把无关凭据写入归档。

原始 bench JSON 不修改，保留 TTFT/E2EL 的均值、标准差和 P50/P90/P99 等全部字段。
metrics.csv 保留各模式后两轮均值与跨轮样本标准差；summary.json 包含逐轮完整结果。
input_throughput = total_input_tokens / duration。比较表展示 TTFT、输入吞吐、总吞吐和整轮耗时及相对变化。

只比较双方完整成功且同轮请求数、输入/输出 token 总量一致的配对；缺失/失败项明确标 incomplete，不当作 0 或成功样本。
降低比例 = 100 × (1 - slot/off)；吞吐提高比例 = 100 × (slot/off - 1)，正值表示 slot 更好。
两轮样本只能描述本次数据，不据此声称统计显著性。

重新汇总：

```bash
python summarize.py data/<实际session目录>
```

汇总会重建该 session 的 summary.json、summary.md、metrics.csv、comparison.csv，不改原始结果。

## 2026-11-03 执行范围与启动修正

用户将实际执行范围收窄为 s≈2.4822（3hot1r），输入/预算8K/16K及off/slot保持不变，共8次起服；独立配置为config_s248.json。
首次session data/20261103_141216_015297在第一组模型初始化阶段失败，无有效bench数据。
根因：编排器原来仅设置VLLM_PLUGINS=ascend，屏蔽了Ascend通用模型注册插件，回退到上游NVIDIA实现并报FlashMLA FP8 KV cache断言。
已修正为同时允许ascend平台、ascend_kv_connector、ascend_model_loader、ascend_service_profiling、ascend_model。
修正后5项逻辑测试通过；load_general_plugins后注册项已确认指向vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM。
KV cache类型、模型、路由、DP/TP和负载参数未因该错误而改动。首次失败数据不计入性能比较，重跑另建session。

第二次session data/20261103_141611_350344：第一组off（L8192/B8192）完成；slot在分配9GiB KV cache时NPU OOM，未形成有效配对。
用户确认先验证5.5GiB KV cache、max-model-len=16385的起服能力。独立配置config_s248_kv55.json；verify_startup.py只起slot服务并验证/v1/models，随后清理，不发送benchmark或推理请求。
命令：python verify_startup.py --config config_s248_kv55.json --batched-tokens 8192 --execute。
该检查通过仅说明服务可启动，不保证最大形状推理不会OOM；性能矩阵需要在相同新参数下重新测两组。

用户随后明确只实验8K，不实验16K。当前实际配置为config_s248_8k_kv55.json：仅3hot1r，输入8192、token预算8192、输出1、max-model-len=8193、KV=5.5GiB，slot/off两组都用相同参数。
为优先检查96GiB上slot是否可运行，first_mode=slot，slot成功后再测off。已加入致命启动日志提前退出机制及回归检查，避免引擎OOM后仅等待HTTP就绪超时。
5.5GiB/maxlen16385的独立起服验证因KV容量不足失败，实际日志要求至少6.81GiB；不把这次失败计入性能结果。

5.5GiB KV、maxlen8193的slot虽起服和预热成功，首轮8K真实压测仍OOM（hidden_states.clone申请256MiB）。该session data/20261103_143223_920608无有效配对。
用户确认继续两组统一4GiB KV；新配置config_s248_8k_kv4.json，仅KV参数由5905580032降到4294967296，其余8K形状、并发16、请求32、轮次3及slot-first不变。后续单独session保存，不复用旧off基线。

最新用户指令：性能实验暂缓，等其它卡空闲；先检查MoonEP slot是否能处理正常请求且实际生效。
新增check_slot_requests.py：卡4..7，4GiB KV、maxlen8193、token预算8192；关闭所有SIM_ROUTE变量，使用自然路由，开MOONEP_BALANCE_STATS和MOONEP_TIME_STATS诊断计数，不测性能。
检查两个短completion请求（每个最多16输出token）HTTP成功、非空输出，并检查四个rank的dispatch/combine调用计数相对请求前均增长以及worker的slot env audit；完成后清理自身服务。
