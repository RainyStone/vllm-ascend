# dsv4flash_service —— DSV4-Flash × MoonEP 实验工作区

**组织原则：一个实验一个目录，目录名 = `日期_实验名`。** 实验的计划、编排/测试脚本、
数据、报告全部收在自己目录里，互不混放；只有跨实验复用的基础设施放在共享目录。

```text
dsv4flash_service/
├── service/                     # 共享：起服与占卡（所有实验复用）
│   ├── start_service_mode.sh    #   DP8 三模式起服（off/owner/slot）
│   ├── start_service_mode_dp4.sh#   DP4 版（日常验证与性能矩阵用）
│   └── hold_cards_vllm.sh(+py)  #   占卡器（pid 文件在 /tmp/mc2_probe/）
├── sim/                         # 共享：仿真路由档案（峰均比 s=1.00/1.66/2.48/2.98）
│
├── 20261008_rt_min16/           # 实验：real-T 重跑矩阵（off/slot/slot0 ×24 台）
│   ├── PLAN.txt  run_rt_phase.sh  bench_rt_grid.sh  gen_rt_report.py
│   └── data/                    #   24 台 bench json + 服务日志（自 /tmp 归档）
├── 20261010_slot_smoke/         # 实验：首次 slot 服务连通性验证（服务日志）
├── 20261011_opt2_off_vs_slot/   # 实验：off vs slot(+优化组合) ×32 台性能矩阵
│   ├── PLAN.txt  run_opt_phase.sh  bench_rt_one.sh  gen_opt_report.py
│   ├── data/                    #   32 台 bench json + 服务日志 + RUN_opt.log
│   └── report/                  #   HTML/MD 报告（图表+机制分析+综合结论）
├── 20261011_acc_off_vs_slotopt/ # 实验：精度对比（off/off2/slot×2 四份输出 json）
│   ├── acc_test.py
│   └── data/
└── legacy_gsm8k_perf/           # 历史遗留：gsm8k 四腿对照报告生成器（数据在仓外）
```

## 新开实验的做法

1. 建目录 `mkdir <日期>_<实验名>`，固定放四类东西：
   - **`EXPERIMENT.md`：实验方案说明**——目的、矩阵设计、起服/bench 的确切命令与参数、
     统计口径、数据文件含义、结论。日后回看"这批数据怎么测的"就看它。
   - **`RUNS.txt`：编排器运行清单**（机器可读，每行一台服务：run 名/腿/参数/路由/形状），
     不是方案本身——方案在 EXPERIMENT.md。
   - 该实验专用的编排/bench/测试脚本、`gen_*_report.py`（报告生成器）。
   - `data/`（运行产物归档）、`report/`（成品报告）。
2. 起服脚本、仿真档案**不要复制**，直接用共享的 `../service/`、`../sim/`
   （编排器里 `SVC=$HERE/../service/...` 的写法即为此约定）。
3. 运行期数据照旧写 `/tmp/mc2_probe/...`（tmpfs 快但重启即丢），**跑完立刻归档进本实验 `data/`**。
4. bench 口径若与已有实验一致，可跨目录引用旧脚本；口径变了就拷贝一份改，避免动旧实验。

## 常用命令（示例）

```bash
# 起一台服务（中性 cwd，OUT_DIR/LOG_FILE 指到 /tmp）
cd 20261011_opt2_off_vs_slot && OUT_DIR=/tmp/x LOG_FILE=/tmp/x.log \
  bash ../service/start_service_mode_dp4.sh slot
# 跑一个矩阵（断点续跑，.done 跳过）
nohup bash run_opt_phase.sh RUNS.txt > /tmp/RUN.log 2>&1 &
# 生成报告（默认读 /tmp/mc2_probe/grid_opt，归档后用 --grid data/）
python3 gen_opt_report.py [--grid data/]
# 占卡/放卡
bash ../service/hold_cards_vllm.sh start 0 1 2 3   # stop 放卡
```

## 注意事项（历史坑）

- 起服 cwd 不能在有裸 `vllm/` 的目录（CANN set_env 空 PYTHONPATH 条目会抢解析）。
- 容器有公司代理：本地 curl 要 `--noproxy '*'`，python 用 `ProxyHandler({})`。
- 停服后驱动侧 HBM 释放滞后 ~1 分钟，立刻起下一台会被显存检查判死。
- slot 性能腿优化前缀固定三件套（全 rank 一致）：
  `ASCEND_MOONEP_PREFETCH_RELAXED=1 ASCEND_MOONEP_NO_RECONCILE=1 ASCEND_MOONEP_SKIP_DUP=1`
- worker 实际生效 env 以服务日志 `[moonep-env-audit]` 行为准，勿信 /proc/environ。
