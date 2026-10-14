# 实验：real-T 重跑矩阵（off / slot / slot0 × 24 台）· 2026-10-08

> 本实验为**上一会话**（整理前代码）所跑，本目录归档其方法与数据。

## 要回答的问题

real-T（内核按真实 token 负载计算，MOONEP_REAL_T=1）落地后，与旧行为（slot0=
MOONEP_REAL_T=0，pad 全量计算）及 off 的三腿对照；同时交叉 max_num_batched_tokens
两档。是 H16 轮（min16 条件）的重跑。

## 方法（run_rt_phase.sh + bench_rt_grid.sh，RUNS.txt 为运行清单）

- 矩阵：2 档 max_num_batched_tokens（8192/16384）× 4 路由（同 `../sim/` 四档峰均比）
  × 3 腿（off / slot[real-T] / slot0[消融]）= 24 台；**每台服务内跑双形状**
  （L8k+L16k 各 3 轮，轮 1 弃）——与 20261011 实验的"每形状一台"粒度不同。
- 起服带 `MOONEP_MIN_DISPATCH_TOKENS=16`（min16 实验条件）与 SIM_ROUTE 注入；
  slot0 腿额外前缀 `MOONEP_REAL_T=0`。
- bench 口径与 20261011 实验相同（vllm bench serve，32 prompts/并发 16/输出 1 token）。

## 数据（data/）

`rt_*/`（每台 6 个 bench json：双形状×3 轮 + warmup）、`rt_*.log`（服务日志）、
`RUN_rt.log`（编排日志）。报告生成器 `gen_rt_report.py`（读 /tmp/mc2_probe/grid，
输出三腿对照 MD+HTML；重生成需数据在位）。
