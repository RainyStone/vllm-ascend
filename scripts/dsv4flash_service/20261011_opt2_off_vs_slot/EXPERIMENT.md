# 实验：off vs slot（+历史收益优化组合）性能矩阵 · 2026-10-10/11

> 本文件说明这批数据**是怎么测出来的**；`RUNS.txt` 只是编排器的运行清单（每行一台服务）。

## 1. 要回答的问题

DSV4-Flash（FP8 MoE，E=256，K=6，43 个 MoE 层）在 DP4/EP4 上，MoonEP slot 模式
（真实 token 负载均衡 real-T + 历史有收益的算子侧优化组合）相对标准 MC2/all-to-all EP（off）：

1. 不同**峰均比**（专家负载倾斜度）下 TTFT / 吞吐谁好？交叉点在哪？
2. 输入长度（8k / 16k token，输出恒 1 token，prefill 主导）的影响？
3. max_num_batched_tokens（= MoonEP dispatch 静态容量 S）= 8192 / 16384 的影响？

## 2. 矩阵（32 台服务，每形状一台）

| 变化轴 | 取值 |
|---|---|
| 腿 | `off`（对照） / `slot`（+优化组合） |
| 峰均比（仿真路由） | uniform s=1.00 / 4hot s=1.66 / 3hot1r s=2.48 / 4hot1r s=2.98（档案在 `../sim/`） |
| 形状 | L8k（输入 8192）/ L16k（输入 16384），输出 1 token |
| max_num_batched_tokens | 8192 / 16384 |

## 3. 每台服务的流程（run_opt_phase.sh 自动执行）

停占卡器 → 等卡 0-3 HBM 回落基线 → `setsid` 起全新服务（起服命令模板）→
curl `/v1/models` 探活（就绪后）→ `bench_rt_one.sh` 压测 → 按进程组停服 → 写 `.done`。
中断可断点续跑（`.done` 存在即跳过）。

**起服命令模板**（slot 腿，off 腿去掉三件套前缀）：

```bash
setsid env ASCEND_MOONEP_PREFETCH_RELAXED=1 ASCEND_MOONEP_NO_RECONCILE=1 \
  ASCEND_MOONEP_SKIP_DUP=1 \
  OUT_DIR=/tmp/mc2_probe/grid_opt/<run> LOG_FILE=/tmp/mc2_probe/grid_opt/<run>.log \
  BATCHED_TOKENS=<8192|16384> MOONEP_MIN_DISPATCH_TOKENS=16 \
  SIM_ROUTE=@../sim/<档案>.txt \
  bash ../service/start_service_mode_dp4.sh slot
```

服务固定参数：DSV4-Flash、DP=4/TP=1/EP on、卡 0-3、端口 8444、model 名 `test_auto`、
enforce-eager、chunked prefill、seed 1024、max_model_len 20480、kv-cache 9 GiB、
gpu-mem-util 0.80、max-num-seqs 16。两仓代码版本见报告 §0（git commit）。

**slot 腿开启的优化**（历史依据详见报告 §1）：三件套 env 前缀（RELAXED 摘 prefetch 跨卡
barrier / NO_RECONCILE 跳每层 3 次设备排空 / SKIP_DUP 短路 source-dup）+ 默认开项
（real-T、dispatch 输出复用、slot FP8 量化镜像、planning 核数自适应等）。BARRIER_V4 因
R=4 实测中性未开。**生效证据**：每台服务日志里的 `[moonep-env-audit]` 行（worker 运行时
实际 env 快照）。**两腿专家分布一致**：off 腿同样注入 SIM_ROUTE。

## 4. bench 方法（bench_rt_one.sh）

```bash
vllm bench serve --backend openai --endpoint /v1/completions \
  --base-url http://127.0.0.1:8444 --model <模型路径> --served-model-name test_auto \
  --dataset-name random --num-prompts 32 --max-concurrency 16 \
  --random-input-len <8192|16384> --random-output-len 1 --random-range-ratio 0.0 \
  --request-rate inf --ignore-eos --seed <档位序号×1000+轮次> \
  --percentile-metrics ttft,e2el --metric-percentiles 50,90,99 \
  --save-result --result-dir <run目录> --result-filename <形状>_c<轮>.json
```

每台服务：先 warmup（2 条 × 1024 token），再同形状 3 轮（seed 逐轮 +1）。
**统计口径：轮 1 视为热身弃掉，取轮 2/3 均值**（n=2 时均值=中位）；bench json 的全部
标量字段（mean/median/std/P50/P90/P99、各吞吐、token 计数等 34 项）逐轮保留不省略。

## 5. 数据文件说明（data/）

| 文件 | 含义 |
|---|---|
| `op2_*/L*_n32_c{1,2,3}.json` | 该台服务的 3 轮 bench 结果（c1=热身轮，统计弃用） |
| `op2_*/warmup_c0.json` | 起服后 warmup（不计入任何统计） |
| `op2_*.log` | 该台服务完整日志：横幅（生效配置）、`[moonep-env-audit]`（优化开关证据）、`[sim-route]`（每步 token 数直方图——报告 §5 机制分析即由此算出步数） |
| `op2_*.done` / `op2_*.boot.out` | 断点标记 / 起服器输出 |
| `RUN_opt.log` | 编排器全程日志（每台服务的起停时间线） |

原始数据当时在 `/tmp/mc2_probe/grid_opt/`（tmpfs），本目录 data/ 为跑完归档；
`gen_opt_report.py --grid data/` 可从归档直接重生成报告。

## 6. 报告

`python3 gen_opt_report.py` → `report/op2_off_vs_slot_opt_report.html`（含：实验条件
明细、逐轮全字段数据、收益统计、§4 图表、§5 机制分析（max_num_batched_tokens 与输入
长度的"起步价/里程费"分解）、§6 综合结论）。

## 7. 一句话结论

slot 对倾斜几乎免疫（TTFT 极差 1~7% vs off 的 39~47%），max_num_batched_tokens=16384
时交叉点在峰均比 ≈1.75；s=2.98 时 slot 快 20.7%、吞吐高 30.6%；max_num_batched_tokens
小于输入长度时每步"起步价"翻倍，slot 全线吃亏（最差 +50.6%）。
