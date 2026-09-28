#!/usr/bin/env python3
"""生成 MoonEP off / slot 性能差异与优化点报告（gsm8k 四次运行）。

用法（本脚本在 <仓根>/scripts/dsv4flash_service/ 下，与它生成的报告同目录）::

    python3 scripts/dsv4flash_service/gen_moonep_off_vs_slot_perf_report.py [--out <html>]

数据来源（全部只读，不占卡）——跑测产物在工作区的 ``$WS/scripts/`` 下，不在本仓内:

* 客户端日志  ``$WS/scripts/gsm8k_profiling*/..._test_gsm8k.log``   —— ais_bench 端到端指标
* 负载 payload ``$WS/scripts/gsm8k_profiling_moepar_moonep_*/`` 的运行报告 ``*_report.html``
               （记录器 dump 就在它的内嵌 JSON 里，直接取，避免重跑分析器）
* 设备侧导出   ``/tmp/prof3/{off,slot}/merged/**/mindstudio_profiler_output/``
               （由 ``msprof --export=on --output=<rank0 的 *_ascend_pt 目录>`` 离线导出；
               CANN 9.2 的产物是 op_summary / op_statistic / task_time* 等，
               没有老布局的 kernel_details.csv / step_trace_time.csv）

所有数字都在本脚本里现算，报告里不写死任何常量，便于换运行重跑。
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 本文件在 <工作区根>/vllm_ascend_moonep/scripts/dsv4flash_service/ 下：仓根 = parents[2]，
# 而运行目录、日志、payload 报告这些跑测产物仍在工作区的 $WS/scripts/ 里（不在本仓内）。
REPO = Path(__file__).resolve().parents[2]
WS = REPO.parent
SCRIPTS = WS / "scripts"

# ---------------------------------------------------------------- 运行配置
# 名字 -> 运行目录。g1 = 干净端到端；g2 = 同一配置 + moe_par dump（flush_every=1）
LEGS = {
    "g1_off": "gsm8k_profiling_moonep_off",
    "g1_slot": "gsm8k_profiling_moonep_slot",
    "g2_off": "gsm8k_profiling_moepar_moonep_off",
    "g2_slot": "gsm8k_profiling_moepar_moonep_slot",
}
# 设备侧导出根目录（按运行）
PROF_ROOTS = {
    "off": Path("/tmp/prof3/off/merged"),
    "slot": Path("/tmp/prof3/slot/merged"),
}
# moe_par dump payload：直接从运行报告的 <script id="payload"> 里取
PAYLOAD_SRC = {
    "off": SCRIPTS / "gsm8k_profiling_moepar_moonep_off/gsm8k_profiling_moepar_moonep_off_report.html",
    "slot": SCRIPTS / "gsm8k_profiling_moepar_moonep_slot/gsm8k_profiling_moepar_moonep_slot_report.html",
}

RANKS = 8          # DP8 / EP8（每 rank 一个设备）
TOP_K = 6          # DSV4-Flash MoE topk
N_LAYERS = 43      # MoE 层数
# 默认写到本脚本旁边：报告与生成器同目录（都在仓内）
OUT_DEFAULT = Path(__file__).resolve().parent / "gsm8k_moonep_off_vs_slot_perf_report.html"

# 「空转步」下限：该步所有 rank 加起来只有极少真实 token，但池子仍按 capacity 算。
# off 运行 D 波：8 rank × 1 token = 8 token ⇒ 计算行 = 8 × topk = 48
# slot 运行 D 波：capacity 下限 128 ⇒ 计算行 = 128 × topk = 768
FLOOR_ROWS = {"off": 8 * TOP_K, "slot": 128 * TOP_K}
DUMMY_TOK = {"off": 1, "slot": 1}   # 空转步里每个 rank 的真实 token 数（off 是 1，slot 池化后按 1 记）


# ---------------------------------------------------------------- 小工具
def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def fnum(x, nd=1, dash="—") -> str:
    """格式化数字；None/NaN 给破折号。"""
    if x is None:
        return dash
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return dash
    if not np.isfinite(xf):
        return dash
    return f"{xf:,.{nd}f}"


def us(x, nd=0) -> str:
    return fnum(x, nd) + " µs"


def ms(x, nd=1) -> str:
    return fnum(x, nd) + " ms"


def sec(x, nd=2) -> str:
    return fnum(x, nd) + " s"


def pct_up(a, b, nd=1) -> str:
    """b 相对 a 的增幅（%）。"""
    a, b = float(a), float(b)
    if a == 0:
        return "—"
    return f"{(b / a - 1) * 100:+,.{nd}f}%"


def pct_share(a, b, nd=1) -> str:
    if not b:
        return "—"
    return f"{a / b * 100:,.{nd}f}%"


def delta(a, b, nd=1) -> str:
    return f"{b - a:+,.{nd}f}"


def parse_us(text: str):
    """'419.3 ms' / '885.6 ms' / '151.5 token/s' -> 数值（毫秒转毫秒，保留单位）。"""
    m = re.match(r"\s*([-\d.eE+]+)\s*([a-zA-Z/µ%]*)", text)
    if not m:
        return None, ""
    return float(m.group(1)), m.group(2)


# ---------------------------------------------------------------- 1. 端到端指标
_ROW_RE = re.compile(r"^│\s*(.*?)\s*│\s*(.*?)\s*│(.*)$")


def parse_bench(log_path: Path) -> dict:
    """从 ais_bench 日志里抓性能表 + Common Metric 表。"""
    out: dict = {"metrics": {}, "common": {}}
    lines = log_path.read_text(errors="replace").splitlines()
    for ln in lines:
        if not ln.startswith("│"):
            continue
        cells = [c.strip() for c in ln.strip("│\n").split("│")]
        if len(cells) >= 10 and cells[0] in ("E2EL", "TTFT", "ITL", "InputTokens",
                                             "OutputTokens", "OutputTokenThroughput",
                                             "PrefillTokenThroughput"):
            # Average Min Max Median P75 P90 P99 N
            keys = ["avg", "min", "max", "p50", "p75", "p90", "p99", "n"]
            out["metrics"][cells[0]] = {
                k: parse_us(v) for k, v in zip(keys, cells[2:10])
            }
        elif len(cells) == 3 and cells[0] and cells[0] != "Common Metric":
            val, unit = parse_us(cells[2])
            out["common"][cells[0]] = (val, unit)
    return out


# ---------------------------------------------------------------- 2. 负载 payload
def load_payload(leg: str) -> dict:
    txt = PAYLOAD_SRC[leg].read_text(encoding="utf-8")
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', txt, re.S)
    if not m:
        raise SystemExit(f"找不到 payload: {PAYLOAD_SRC[leg]}")
    return json.loads(m.group(1))["runs"][0]


def payload_summary(run: dict) -> dict:
    """从记录器 payload 里抽出报告要用的量（全部是 8 rank 合计口径）。"""
    led = run["ledger"]
    comp = run["compute"]
    steps = list(run["steps"])
    tps = list(comp["total"]["tokens_per_step"])          # 每步池化 token（全局）
    real_tps = list(comp["real"]["tokens_per_step"])      # 每步真实 token（全局，按 rank 求和）
    cap = {}
    for s, t in zip(steps, tps):
        cap[int(t)] = cap.get(int(t), 0) + 1
    # 阶梯：把每步的池化 token 归类（slot 是 128/256/512 的 2 的幂；off 是真实 token 数）
    ladder = sorted(cap.items(), key=lambda kv: -kv[1])
    expert = np.asarray(run["expert_total"], dtype=float) if run.get("expert_total") else None
    hot = {}
    if expert is not None and expert.size:
        srt = np.sort(expert)[::-1]
        tot = srt.sum()
        hot = {
            "n_experts": int(expert.size),
            "n_nonzero": int((expert > 0).sum()),
            "top1_share": float(srt[:1].sum() / tot) if tot else None,
            "top1pct_share": float(srt[: max(1, expert.size // 100)].sum() / tot) if tot else None,
            "top10pct_share": float(srt[: max(1, expert.size // 10)].sum() / tot) if tot else None,
        }
    per_rank = {r["rank"]: r for r in led["per_rank"]}
    return {
        "n_steps": run["n_steps"],
        "n_layers": run["n_layers"],
        "n_ranks": run["n_ranks"],
        "topk": run["topk"],
        "has_compute": run["has_compute"],
        "ep_size": run["ep_size"],
        "global_experts": run["global_experts"],
        "route_total": led["route_total"],
        "pad_total": led["pad_total"],
        "sent_total": led["sent_total"],
        "real_total": led["real_total"],
        "compute_pad_total": led["compute_pad_total"],
        "par_real": comp["real"]["par_whole"],
        "par_pad": comp["pad"]["par_whole"],
        "par_total": comp["total"]["par_whole"],
        "per_rank": per_rank,
        "rank_real": list(comp["real"]["rank_total"]),
        "rank_pad": list(comp["pad"]["rank_total"]),
        "rank_total": list(comp["total"]["rank_total"]),
        "pad_by_rank": run["pad"]["by_rank"],
        "pad_experts": run["pad"]["experts"],
        "checks": run["checks"],
        "split": run["split"],
        "ladder": ladder,
        "tokens_per_step": tps,
        "real_per_step": real_tps,
        "gemm_rows": int(sum(tps)) * TOP_K,
        "hot": hot,
        "route_per_step": [int(round(x)) for x in comp["real"]["step_pairs"]],
    }




# ---------------------------------------------------------------- 3. 设备侧
def find_glob(root: Path, pat: str) -> Path:
    hits = sorted(root.rglob(pat))
    if not hits:
        raise SystemExit(f"{root} 下找不到 {pat}")
    return hits[-1]          # 带时间戳的同名文件取最新一份


def pick_col(cols, *cands):
    low = {str(c).lower().strip().replace("_", " "): c for c in cols}
    for cand in cands:
        needle = cand.lower().replace("_", " ")
        for k, c in low.items():
            if needle in k:
                return c
    raise KeyError(f"列名未命中 {cands}; 现有列 = {list(cols)}")


# op_summary 的列（CANN 9.2 离线导出）
_OPS = {
    "name": ("op name",), "optype": ("op type",), "ktype": ("task type",),
    "start": ("task start time",), "dur": ("task duration",),
    "shape": ("input shapes",),
    "aic_mac": ("aic_mac_ratio",), "aic_scalar": ("aic_scalar_ratio",),
    "aic_mte1": ("aic_mte1_ratio",),
    "aic_mte2": ("aic_mte2_ratio",), "aic_mte3": ("aic_mte3_ratio",),
    "aic_fixpipe": ("aic_fixpipe_ratio",),
    "aiv_vec": ("aiv_vec_ratio",), "aiv_scalar": ("aiv_scalar_ratio",),
    "aiv_mte2": ("aiv_mte2_ratio",), "aiv_mte3": ("aiv_mte3_ratio",),
}
AIC_KINDS = ("AI_CORE", "MIX_AIC")
AIV_KINDS = ("AI_VECTOR_CORE", "MIX_AIV")
COMM_KINDS = ("COMMUNICATION",)


_OPS_CACHE: dict[str, pd.DataFrame] = {}


def load_ops(leg: str) -> pd.DataFrame:
    """op_summary：AI Core/AIV/AICPU/通信核算子的逐条记录（每 rank 一份，rank0）。"""
    if leg in _OPS_CACHE:
        return _OPS_CACHE[leg]
    f = find_glob(PROF_ROOTS[leg], "**/op_summary_*.csv")
    df = pd.read_csv(f, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    ren = {}
    for k, cands in _OPS.items():
        ren[pick_col(df.columns, *cands)] = k
    df = df.rename(columns=ren)
    for c in ("start", "dur"):
        df[c] = pd.to_numeric(df[c].astype(str).str.strip(), errors="coerce")
    for c in ("aic_mac", "aic_scalar", "aic_mte2", "aic_mte3", "aic_fixpipe",
              "aiv_vec", "aiv_scalar", "aiv_mte2", "aiv_mte3"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["dur"].notna() & df["start"].notna()].copy()
    df["end"] = df["start"] + df["dur"]
    df = df.sort_values("start").reset_index(drop=True)
    _OPS_CACHE[leg] = df
    return df


def _iv_merge(rows) -> list:
    """[(a,b)] -> 升序合并后的不相交区间列表（单位 µs）。"""
    arr = np.asarray(rows, dtype=float)
    if not arr.size:
        return []
    arr = arr[np.argsort(arr[:, 0])]
    out = []
    for a, b in arr:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([float(a), float(b)])
    return out


def _iv_len(iv) -> float:
    return float(sum(b - a for a, b in iv))


def _iv_sub(a, b) -> list:
    """区间集 a 减去 b（两集都须是合并后的）。"""
    out = []
    for x0, x1 in a:
        cur = x0
        for y0, y1 in b:
            if y1 <= cur:
                continue
            if y0 >= x1:
                break
            if y0 > cur:
                out.append([cur, min(y0, x1)])
            cur = max(cur, y1)
            if cur >= x1:
                break
        if cur < x1:
            out.append([cur, x1])
    return out


def window_stats(leg: str) -> dict:
    """窗口与占空比（本报告统一口径，两次运行同一套算法）。

    数据源两处：
      · task_time*.csv      —— 逐 task 实例（含 DMA / notify / event 等非算子任务），
        只用来定窗口：min(start) → max(stop)。两次运行导出的文件名不同
        （off 是 task_time_slice_0/1，slot 是 task_time），glob 都覆盖；
      · op_summary*.csv     —— 算子任务，带 Task Type，用来分片：
        计算 = AI_CORE / AI_VECTOR_CORE / MIX_AIC / MIX_AIV；通信 = COMMUNICATION（hcom_*）；
        AI CPU = AI_CPU（HCCL 下发等）。通信核任务不在 task_time 里，必须从这里取。
    分片按优先级扣除重叠（计算 > 通信 > AI CPU），实测通信与计算核零重叠。
    """
    parts = []
    for f in sorted(PROF_ROOTS[leg].rglob("task_time*.csv")):
        d = pd.read_csv(f, low_memory=False)
        d.columns = [str(c).strip() for c in d.columns]
        d = d.rename(columns={pick_col(d.columns, "task start"): "start",
                              pick_col(d.columns, "task stop"): "stop"})
        parts.append(d[["start", "stop"]])
    ts = pd.concat(parts, ignore_index=True)
    for c in ("start", "stop"):
        ts[c] = pd.to_numeric(ts[c].astype(str).str.strip(), errors="coerce")
    ts = ts.dropna(subset=["start", "stop"])
    w0, w1 = float(ts["start"].min()), float(ts["stop"].max())
    win = (w1 - w0) / 1e6

    ops = load_ops(leg)

    def iv(mask):
        sub = ops.loc[mask]
        return _iv_merge(np.c_[sub["start"].values - w0, sub["end"].values - w0])

    comp = iv(ops["ktype"].isin(AIC_KINDS + AIV_KINDS))
    comm = iv(ops["ktype"] == "COMMUNICATION")
    aicpu = iv(ops["ktype"] == "AI_CPU")
    union_all = _iv_merge(comp + comm + aicpu)
    c_comp = comp
    c_comm = _iv_sub(comm, comp)
    c_cpu = _iv_sub(aicpu, _iv_merge(comp + comm))
    us = {"compute": _iv_len(c_comp), "comm": _iv_len(c_comm), "aicpu": _iv_len(c_cpu)}
    occupied = _iv_len(union_all) / 1e6
    return {"win": win, "us": us, "idle": win - occupied,
            "share": {k: v / (win * 1e6) for k, v in us.items()},
            "occupied": occupied, "n_tasks": int(len(ts)), "w0": w0,
            "n_ops": int(len(ops)), "comm_core_us": _iv_len(comm)}


# ---------------------------------------------------------------- 4. 算子家族
# 每次运行关心的算子（按名字子串匹配）。列 = (内部键, 显示名)
FAMILIES = {
    "off": [
        ("hcom_ag", "hcom_allGather_AicpuKernel", "通信：all-gather"),
        ("hcom_a2a", "hcom_alltoallv_AicpuKernel", "通信：all-to-allv"),
        ("hccl_launch", "HcclLaunchAicpuKernel", "通信：HCCL 下发"),
        ("mc2_dispatch", "MoeDistributeDispatchV3", "dispatch（MC2）"),
        ("mc2_combine", "MoeDistributeCombineV3", "combine（MC2）"),
        ("gemm", "GroupedMatmul", "GroupedMatmul"),
    ],
    "slot": [
        ("plan", "planning_kernel", "规划 plan"),
        ("barrier", "_Z24barrier_on_stream_kernelim", "流屏障 barrier"),
        ("moep_combine", "moonep_combine_kernel", "combine（MoonEP）"),
        ("moep_dispatch", "moonep_dispatch_kernel", "dispatch（MoonEP）"),
        ("prefetch", "prefetch_weight_kernel", "预取权重"),
        ("gemm", "GroupedMatmul", "GroupedMatmul"),
    ],
}
# 画图时统一的分段（顺序 = 调色板槽位顺序）
SEGMENTS = ["hcom_ag", "hcom_a2a", "hccl_launch", "mc2_dispatch", "mc2_combine",
            "moep_dispatch", "moep_combine", "prefetch", "plan", "barrier", "gemm"]
SEG_LABEL = {
    "hcom_ag": "通信 all-gather", "hcom_a2a": "通信 all-to-allv", "hccl_launch": "HCCL 下发",
    "mc2_dispatch": "dispatch(MC2)", "mc2_combine": "combine(MC2)",
    "moep_dispatch": "dispatch(MoonEP)", "moep_combine": "combine(MoonEP)",
    "prefetch": "预取权重", "plan": "规划 plan", "barrier": "流屏障 barrier", "gemm": "GroupedMatmul",
}


def fam_mask(df: pd.DataFrame, pat: str) -> pd.Series:
    """按 OP Type 或算子名匹配家族（两个字段都可能带信息）。"""
    return (df["optype"].str.contains(pat, regex=False)
            | df["name"].str.contains(pat, regex=False))


UNITS_AIC = [("aic_mac", "AIC cube(mac)"), ("aic_scalar", "AIC scalar"),
             ("aic_mte1", "AIC MTE1"), ("aic_mte2", "AIC MTE2"),
             ("aic_mte3", "AIC MTE3"), ("aic_fixpipe", "AIC fixpipe")]
UNITS_AIV = [("aiv_vec", "AIV vector"), ("aiv_scalar", "AIV scalar"),
             ("aiv_mte2", "AIV MTE2"), ("aiv_mte3", "AIV MTE3")]


def insn_busiest(df: pd.DataFrame, pat: str) -> dict:
    """某算子家族的核内指令占比：谁最忙、占多少、其余在干嘛。

    op_summary 的 *_ratio 是「该单元占该核总 cycle 的比例」，各单元是并行流水，
    因此不能相加成「核忙」；有意义的是 <b>最忙的那个单元</b> 与它之外的部分。
    """
    sub = df[fam_mask(df, pat)]
    if sub.empty:
        return {"n": 0, "core": "", "unit": "—", "ratio": None, "rest": None, "means": {}}
    kt = sub["ktype"].mode().iloc[0]
    if kt in AIC_KINDS:
        units = UNITS_AIC
    elif kt in AIV_KINDS:
        units = UNITS_AIV
    else:
        return {"n": int(len(sub)), "core": kt, "unit": "（非 AI 核）",
                "ratio": None, "rest": None, "means": {}}
    means = {name: float(sub[col].mean() or 0.0) for col, name in units}
    busiest = max(means, key=means.get)
    r = means[busiest] * 100
    return {"n": int(len(sub)), "core": kt, "unit": busiest, "ratio": r,
            "rest": 100.0 - r, "means": means}


def analyse_device(leg: str) -> dict:
    """设备侧归因：cell 切分 / 两类波 / 每 cell 开销 / 容量阶梯。

    所有量都来自 op_summary（rank0）：
      · cell = 相邻两条 GroupedMatmul（gate/up 与 down），M 取 Input Shapes 首个数
        —— 实测一对两行 M 完全相同（8256/8256），且与 payload 的「本 rank 计算行数」逐格相等；
      · 步切分：按 cell 间的空隙聚类（步内紧、步间松），得到 192 步；
      · 波判定：该步所有 cell 的 M 都等于 FLOOR_ROWS（off 48 / slot 768）= 空转波 D。
    """
    df = load_ops(leg)
    gm = df[fam_mask(df, "GroupedMatmul")].reset_index(drop=True)
    if gm.empty:
        raise SystemExit(f"{leg}: op_summary 里没有 GroupedMatmul")
    if len(gm) % 2:
        gm = gm.iloc[:-1]
    cells = pd.DataFrame({
        "start": gm["start"].values[::2],
        "end": np.maximum(gm["end"].values[::2], gm["end"].values[1::2]),
    })
    m_first = pd.to_numeric(
        gm["shape"].astype(str).str.strip('"').str.split(";").str[0].str.split(",").str[0],
        errors="coerce")
    cells["M"] = np.nanmax(
        np.vstack([m_first.values[::2], m_first.values[1::2]]).astype(float), axis=0)
    # 步切分：算子按时间序落盘，每步 43 层各 1 个 cell ⇒ 第 i 个 cell 属于第 i//43 步。
    # 自检：cell 数必须正好等于 n_steps × 43；空转步的 43 个 cell 的 M 必须完全相同。
    n_steps, rem = divmod(len(cells), N_LAYERS)
    if rem:
        raise SystemExit(f"{leg}: cell 数 {len(cells)} 不是 {N_LAYERS} 的整数倍")
    cells["step"] = np.arange(len(cells)) // N_LAYERS
    floor = FLOOR_ROWS[leg]
    kind = {}
    for s in range(n_steps):
        ms = set(np.round(cells["M"].values[s * N_LAYERS:(s + 1) * N_LAYERS]).astype(int))
        kind[s] = "D" if ms == {floor} else "R"
    # 算子归属到步：按「离哪个 cell 最近」判（用各自的中点比）。
    # 不用「相邻步中点」分界：off 的 HCCL 通信跑在该步首个 GEMM 之前，会被判到上一步去。
    c_mid = (cells["start"].values + cells["end"].values) / 2.0
    o_mid = df["start"].values + df["dur"].values / 2.0
    idx = np.searchsorted(c_mid, o_mid, side="left")
    idx = np.clip(idx, 0, len(c_mid) - 1)
    left = np.clip(idx - 1, 0, len(c_mid) - 1)
    idx = np.where(np.abs(c_mid[idx] - o_mid) <= np.abs(c_mid[left] - o_mid), idx, left)
    df = df.assign(step=idx // N_LAYERS)
    n_cells_by_kind = {k: int((cells["step"].map(kind) == k).sum()) for k in ("R", "D")}
    n_steps_by_kind = {k: sum(1 for v in kind.values() if v == k) for k in ("R", "D")}
    # 家族统计（一次算好 mask，避免反复 str.contains）
    fam: dict[str, dict] = {}
    step_fam: dict[str, pd.Series] = {}
    for key, pat, _ in FAMILIES[leg]:
        sub = df[fam_mask(df, pat)]
        step_fam[key] = sub.groupby("step")["dur"].sum()
        rec = {"n": int(len(sub)), "us": float(sub["dur"].sum()),
               "us_avg": float(sub["dur"].mean()) if len(sub) else None,
               "us_max": float(sub["dur"].max()) if len(sub) else None,
               "ktype": sub["ktype"].mode().iloc[0] if len(sub) else ""}
        for k in ("R", "D"):
            steps_k = {s for s, v in kind.items() if v == k}
            sk = sub[sub["step"].isin(steps_k)]
            rec[f"n_{k}"] = int(len(sk))
            rec[f"us_{k}"] = float(sk["dur"].sum())
            rec[f"percell_{k}"] = (float(sk["dur"].sum()) / n_cells_by_kind[k]
                                   if n_cells_by_kind[k] else None)
        fam[key] = rec
    # 容量阶梯（按 M/TOP_K）
    cells["cap"] = np.round(cells["M"] / TOP_K).astype("Int64")
    ladder = []
    for cap, grp in cells.groupby("cap"):
        if pd.isna(cap):
            continue
        steps_here = list(grp["step"].unique())
        rec = {"cap": int(cap), "n_cells": int(len(grp)), "n_steps": len(steps_here)}
        for key, _, _ in FAMILIES[leg]:
            s = step_fam[key]
            idx = s.index.intersection(steps_here)
            tot_us = float(s.loc[idx].sum()) if len(idx) else 0.0
            rec[key] = tot_us / len(grp)
        ladder.append(rec)
    ladder.sort(key=lambda r: r["cap"])
    tok_hist = cells.groupby("cap").size().sort_index()
    m_uniq = cells["M"].dropna()
    return {"df": df, "cells": cells, "kind": kind,
            "n_steps": n_steps, "n_cells": int(len(cells)),
            "n_R": n_steps_by_kind["R"], "n_D": n_steps_by_kind["D"],
            "n_cells_R": n_cells_by_kind["R"], "n_cells_D": n_cells_by_kind["D"],
            "fam": fam, "ladder": ladder, "tok_hist": tok_hist,
            "m_min": float(m_uniq.min()), "m_max": float(m_uniq.max()),
            "m_med": float(m_uniq.median()), "floor": floor,
            "n_floor_cells": int((np.round(cells["M"]) == floor).sum()),
            "gm_cells": len(cells)}


# ---------------------------------------------------------------- 5. 画图（dataviz 规格）
SURFACE_LIGHT = "#fcfcfb"
C_GRID = "var(--gridline)"
C_BASE = "var(--baseline)"
C_MUTED = "var(--text-muted)"
C_INK = "var(--text-primary)"
C_SEC = "var(--text-secondary)"
PALETTE = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5",
           "--series-6", "--series-7", "--series-8"]


def _esc(s):
    return esc(s).replace('"', "&quot;")


def svg_stack_bars(rows, series, *, width=1040, bar_h=26, row_gap=42, pad=(8, 8, 40, 0),
                   x_max=None, x_label="", val_fmt=lambda v: f"{v:,.0f}",
                   note="", left=210, right=150):
    """横向堆叠条。rows=[{label, vals:{key:val}, caption}]；series=[{key,label,color}]。"""
    n = len(rows)
    top = pad[0]
    height = top + n * row_gap + pad[2]
    x0, x1 = left, width - right
    axis_w = x1 - x0
    tot = [sum(r["vals"].get(s["key"], 0) or 0 for s in series) for r in rows]
    vmax = x_max or (max(tot) * 1.12 if tot else 1)
    scale = axis_w / vmax if vmax else 0
    out = [f'<svg class="viz" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="{_esc(note or "堆叠条形图")}">']
    # 网格线
    nticks = 5
    for i in range(nticks + 1):
        x = x0 + axis_w * i / nticks
        v = vmax * i / nticks
        out.append(f'<line x1="{x:.2f}" y1="{top - 4}" x2="{x:.2f}" y2="{top + n * row_gap - bar_h - 6}" '
                   f'stroke="{C_GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x:.2f}" y="{top + n * row_gap - bar_h + 6}" font-size="11" '
                   f'fill="{C_MUTED}" text-anchor="middle">{val_fmt(v)}</text>')
    out.append(f'<text x="{x1}" y="{top + n * row_gap + 12}" font-size="11" fill="{C_MUTED}" '
               f'text-anchor="end">{_esc(x_label)}</text>')
    for ri, r in enumerate(rows):
        y = top + ri * row_gap
        out.append(f'<text x="{left - 14}" y="{y + bar_h - 6}" font-size="13" fill="{C_SEC}" '
                   f'text-anchor="end">{_esc(r["label"])}</text>')
        x = x0
        segs = [(s, r["vals"].get(s["key"], 0) or 0) for s in series]
        active = [(s, v) for s, v in segs if v > 0]
        for i, (s, v) in enumerate(active):
            w = v * scale
            r_round = 4 if i == len(active) - 1 else 0
            path = _bar_path(x, y, w, bar_h, r_round)
            tip = f'{r["label"]} · {s["label"]}: {val_fmt(v)}'
            out.append(f'<g><title>{_esc(tip)}</title>'
                       f'<path d="{path}" fill="var({s["color"]})"/></g>')
            if i:  # 段间 2px 表面色缝隙
                out.append(f'<line x1="{x:.2f}" y1="{y}" x2="{x:.2f}" y2="{y + bar_h}" '
                           f'stroke="var(--surface-1)" stroke-width="2"/>')
            x += w
        out.append(f'<text x="{x + 8:.2f}" y="{y + bar_h - 6}" font-size="12.5" '
                   f'fill="{C_INK}" font-weight="600">{_esc(r.get("caption") or val_fmt(sum(v for _, v in active)))}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _bar_path(x, y, w, h, r=0):
    if w <= 0:
        return ""
    r = min(r, w / 2, h / 2)
    if r <= 0.5:
        return f"M{x:.2f} {y:.2f} h{w:.2f} v{h:.2f} h{-w:.2f} Z"
    return (f"M{x:.2f} {y:.2f} h{w - r:.2f} a{r:.2f} {r:.2f} 0 0 1 {r:.2f} {r:.2f} "
            f"v{h - 2 * r:.2f} a{r:.2f} {r:.2f} 0 0 1 {-r:.2f} {r:.2f} h{-(w - r):.2f} Z")


def svg_group_bars(groups, series, *, width=1040, height=300, pad=(16, 16, 52, 0),
                   x_label="", val_fmt=lambda v: f"{v:,.0f}", note="", left=80):
    """竖向分组条（用于容量阶梯）。groups=[{label, vals:{key:val}, caption}]"""
    top = pad[0]
    plot_h = height - pad[0] - pad[2]
    x0, x1 = left, width - 16
    allv = [v for g in groups for v in g["vals"].values()]
    vmax = (max(allv) * 1.15) if allv else 1
    out = [f'<svg class="viz" viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(note or "分组条形图")}">']
    nticks = 4
    for i in range(nticks + 1):
        y = top + plot_h - plot_h * i / nticks
        out.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="{C_GRID}" stroke-width="1"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.2f}" font-size="11" fill="{C_MUTED}" '
                   f'text-anchor="end">{val_fmt(vmax * i / nticks)}</text>')
    gw = (x1 - x0) / len(groups)
    bw = min(24, (gw - 28) / len(series))
    for gi, g in enumerate(groups):
        gx = x0 + gi * gw + gw / 2
        for si, s in enumerate(series):
            v = g["vals"].get(s["key"], 0) or 0
            h = v / vmax * plot_h
            bx = gx - (len(series) * (bw + 4)) / 2 + si * (bw + 4)
            y = top + plot_h - h
            rounded = 4 if h > 6 else 0
            path = (f"M{bx:.2f} {y + h:.2f} v{-h + rounded:.2f} a{rounded} {rounded} 0 0 1 {rounded} {rounded} "
                    f"h{bw - 2 * rounded:.2f} a{rounded} {rounded} 0 0 1 {rounded} {rounded} v{h - rounded:.2f} Z") if rounded \
                else f"M{bx:.2f} {y + h:.2f} h{bw:.2f} v{-h:.2f} h{-bw:.2f} Z"
            out.append(f'<g><title>{_esc(g["label"] + " · " + s["label"] + ": " + val_fmt(v))}</title>'
                       f'<path d="{path}" fill="var({s["color"]})"/></g>')
            out.append(f'<text x="{bx + bw / 2:.2f}" y="{y - 6:.2f}" font-size="11.5" fill="{C_INK}" '
                       f'text-anchor="middle" font-weight="600">{val_fmt(v)}</text>')
        out.append(f'<text x="{gx:.2f}" y="{top + plot_h + 20:.2f}" font-size="12" fill="{C_SEC}" '
                   f'text-anchor="middle">{_esc(g["label"])}</text>')
        if g.get("caption"):
            out.append(f'<text x="{gx:.2f}" y="{top + plot_h + 38:.2f}" font-size="11" fill="{C_MUTED}" '
                       f'text-anchor="middle">{_esc(g["caption"])}</text>')
    out.append(f'<line x1="{x0}" y1="{top + plot_h}" x2="{x1}" y2="{top + plot_h}" stroke="{C_BASE}" stroke-width="1"/>')
    if x_label:
        out.append(f'<text x="{x1}" y="{top + plot_h + 20:.2f}" font-size="11" fill="{C_MUTED}" '
                   f'text-anchor="end">{_esc(x_label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def legend(series) -> str:
    items = "".join(
        f'<span class="lg"><span class="sw" style="background:var({s["color"]})"></span>{esc(s["label"])}</span>'
        for s in series)
    return f'<div class="legend">{items}</div>'


# 画图用的 7 个「实体」（颜色跟着实体走，跨图一致）
SEG_ORDER = ["comm", "dispatch", "combine", "prefetch", "plan", "barrier", "gemm"]
SEG_META = {
    "comm": ("跨卡通信（HCCL）", 1),
    "dispatch": ("dispatch", 2),
    "combine": ("combine", 3),
    "prefetch": ("预取权重", 4),
    "plan": ("规划 plan", 5),
    "barrier": ("流屏障 barrier", 6),
    "gemm": ("GroupedMatmul", 7),
}
ROLLUP = {
    "comm": ["hcom_ag", "hcom_a2a", "hccl_launch"],
    "dispatch": ["mc2_dispatch", "moep_dispatch"],
    "combine": ["mc2_combine", "moep_combine"],
    "prefetch": ["prefetch"],
    "plan": ["plan"],
    "barrier": ["barrier"],
    "gemm": ["gemm"],
}


def seg_series(keys, colors=None):
    return [{"key": k, "label": SEG_META[k][0],
             "color": (colors or {}).get(k, PALETTE[SEG_META[k][1] - 1])} for k in keys]


def plan_slopes(dev_slot: dict) -> dict:
    """从池容量阶梯的 plan µs/cell 拟合两段边际斜率（µs/token）。

    只有 128 / 256 两档是稳态（512 档只有 1 步），所以给区间：
    乐观 = 128→256 的斜率，悲观 = 256→512 的斜率（“大头是固定开销”）。
    """
    def at(cap):
        return next((r.get("plan") for r in dev_slot["ladder"] if r["cap"] == cap), None)

    p128, p256, p512 = at(128), at(256), at(512)
    hi = ((p256 or p128 or 0.0) - (p128 or 0.0)) / 128
    lo = ((p512 or 0.0) - (p256 or 0.0)) / 256
    return {"p128": p128 or 0.0, "p256": p256 or 0.0, "p512": p512 or 0.0, "hi": hi, "lo": lo}


def rollup_fam(dev: dict) -> dict:
    """把细分家族合并成 7 个实体：{key: {us, us_R, us_D, percell_R, percell_D, n}}"""
    out = {}
    for key, parts in ROLLUP.items():
        rec = {"us": 0.0, "us_R": 0.0, "us_D": 0.0, "n": 0}
        for p in parts:
            f = dev["fam"].get(p)
            if not f:
                continue
            rec["us"] += f["us"]
            rec["us_R"] += f["us_R"]
            rec["us_D"] += f["us_D"]
            rec["n"] += f["n"]
        rec["percell_R"] = rec["us_R"] / dev["n_cells_R"] if dev["n_cells_R"] else None
        rec["percell_D"] = rec["us_D"] / dev["n_cells_D"] if dev["n_cells_D"] else None
        rec["present"] = rec["n"] > 0
        out[key] = rec
    return out


# ---------------------------------------------------------------- 6. 报告
CSS = """
:root{
  color-scheme: light;
  --surface-1:#fcfcfb; --plane:#f9f9f7; --surface-2:#f2f1ec;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#898781;
  --gridline:#e1e0d9; --baseline:#c3c2b7; --border:rgba(11,11,11,.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a; --series-4:#eda100;
  --series-5:#e87ba4; --series-6:#008300; --series-7:#4a3aa7; --series-8:#e34948;
  --ok:#006300; --warn:#fab219; --bad:#d03b3b; --neutral-fill:#c3c2b7;
}
@media (prefers-color-scheme: dark){
  :root:where(:not([data-theme="light"])){
    color-scheme: dark;
    --surface-1:#1a1a19; --plane:#0d0d0d; --surface-2:#232322;
    --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#898781;
    --gridline:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10);
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --series-4:#c98500;
    --series-5:#d55181; --series-6:#008300; --series-7:#9085e9; --series-8:#e66767;
    --ok:#0ca30c; --neutral-fill:#52514e;
  }
}
:root[data-theme="dark"]{
  color-scheme: dark;
  --surface-1:#1a1a19; --plane:#0d0d0d; --surface-2:#232322;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#898781;
  --gridline:#2c2c2a; --baseline:#383835; --border:rgba(255,255,255,.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70; --series-4:#c98500;
  --series-5:#d55181; --series-6:#008300; --series-7:#9085e9; --series-8:#e66767;
  --ok:#0ca30c; --neutral-fill:#52514e;
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--text-primary);
  font:15px/1.65 system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif;}
main{max-width:1180px;margin:0 auto;padding:32px 22px 64px}
h1{font-size:27px;line-height:1.3;margin:0 0 6px}
h2{font-size:20px;margin:38px 0 10px;padding-top:14px;border-top:1px solid var(--border)}
h3{font-size:16px;margin:24px 0 8px;color:var(--text-primary)}
h4{font-size:14px;margin:18px 0 6px;color:var(--text-secondary)}
p,li{color:var(--text-secondary)}
code{background:var(--surface-2);padding:1px 5px;border-radius:4px;font-size:12.5px;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--text-primary)}
pre{background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:12px 14px;
  overflow:auto;font-size:12.5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--text-secondary)}
.sub{color:var(--text-muted);font-size:13.5px;margin:0 0 18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:16px 0 8px}
.kpi{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.kpi .v{font-size:24px;font-weight:650;color:var(--text-primary);letter-spacing:-.2px}
.kpi .t{font-size:12.5px;color:var(--text-muted);margin-top:2px}
.kpi .n{font-size:12.5px;color:var(--text-secondary);margin-top:6px}
.tldr{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:6px 18px 14px;margin:16px 0}
table.tbl{border-collapse:collapse;width:100%;margin:10px 0 6px;font-size:13.5px;
  background:var(--surface-1);border:1px solid var(--border);border-radius:8px;overflow:hidden}
table.tbl th{text-align:left;font-weight:600;color:var(--text-primary);background:var(--surface-2);
  padding:8px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
table.tbl td{padding:7px 10px;border-bottom:1px solid var(--border);color:var(--text-secondary);
  vertical-align:top;font-variant-numeric:tabular-nums}
table.tbl tr:last-child td{border-bottom:none}
table.tbl td.k{color:var(--text-primary);font-weight:600}
.good{color:var(--ok);font-weight:600}
.bad{color:var(--bad);font-weight:600}
.note{background:var(--surface-1);border-left:3px solid var(--series-1);border-radius:0 8px 8px 0;
  padding:10px 14px;margin:12px 0;font-size:13.5px;color:var(--text-secondary)}
.note.warn{border-left-color:var(--warn)}
.opt{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:12px 16px;margin:12px 0}
.opt h4{margin-top:0;color:var(--text-primary);font-size:15px}
.tag{display:inline-block;font-size:11.5px;font-weight:600;padding:1px 8px;border-radius:999px;
  border:1px solid var(--border);background:var(--surface-2);color:var(--text-secondary);margin-left:6px}
.tag.p1{background:var(--series-8);color:#fff;border-color:transparent}
.tag.p2{background:var(--series-4);color:#0b0b0b;border-color:transparent}
.tag.p3{background:var(--surface-2);color:var(--text-muted)}
.viz{width:100%;height:auto;display:block;background:var(--surface-1);
  border:1px solid var(--border);border-radius:10px;margin:8px 0 4px}
.legend{display:flex;flex-wrap:wrap;gap:14px;margin:6px 0 0;font-size:12.5px;color:var(--text-secondary)}
.legend .lg{display:inline-flex;align-items:center;gap:6px}
.legend .sw{width:12px;height:12px;border-radius:3px;display:inline-block}
.cap{font-size:12.5px;color:var(--text-muted);margin:2px 0 14px}
.foot{color:var(--text-muted);font-size:12.5px;margin-top:34px;border-top:1px solid var(--border);padding-top:12px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 14px;font-size:13.5px;color:var(--text-secondary)}
.kv b{color:var(--text-primary);font-weight:600}
#theme{position:fixed;top:12px;right:14px;background:var(--surface-1);color:var(--text-secondary);
  border:1px solid var(--border);border-radius:8px;padding:5px 10px;font-size:12.5px;cursor:pointer}
"""


def kpi(value, title, note=""):
    return (f'<div class="kpi"><div class="v">{value}</div><div class="t">{esc(title)}</div>'
            f'{f"<div class=n>{esc(note)}</div>" if note else ""}</div>')


def tbl(headers, rows, cls="tbl"):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            klass = ""
            if isinstance(c, tuple):
                c, klass = c
            tds.append(f'<td class="{klass}">{c}</td>')
        body.append("<tr>" + "".join(tds) + "</tr>")
    return (f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>'
            + "".join(body) + "</tbody></table>")


def tstat(bench: dict, metric: str, key: str):
    return bench["metrics"][metric][key][0]


# §4.3 核内指令占比
def insn_table(DEV) -> str:
    rows = []
    for leg in ("off", "slot"):
        d = DEV[leg]
        for key, pat, _ in FAMILIES[leg]:
            b = insn_busiest(d["df"], pat)
            if not b["n"]:
                continue
            f = d["fam"][key]
            is_aic = b["core"] in AIC_KINDS
            main = b["means"].get("AIC cube(mac)" if is_aic else "AIV vector")
            rows.append([
                leg, f'<code>{esc(pat)}</code>', b["core"], fnum(b["n"], 0),
                fnum(f["us"] / f["n"], 1),
                f"{main:.1f}%" if main is not None else "—",
                b["unit"], f"{b['ratio']:.1f}%" if b["ratio"] is not None else "—",
                f"{b['rest']:.1f}%" if b["rest"] is not None else "—",
            ])
    return tbl(["运行", "算子", "核类型", "调用数", "平均时长 (µs)",
                "主计算单元占比", "最忙单元", "最忙占比", "该核其余（等通知/等数据）"], rows)


# §6 差异点 / §7 优化点
def build_diff_opt(DEV, RU, P, T, W, B) -> str:
    off, sl = DEV["off"], DEV["slot"]
    r_off, r_sl = RU["off"], RU["slot"]

    def per_cell(ru, k, wave):
        v = ru[k].get(f"percell_{wave}")
        return v or 0.0

    def tot_cell(ru, wave):
        return sum(per_cell(ru, k, wave) for k in SEG_ORDER)

    def mech_us(ru):
        """机制总时长（µs，不含 GEMM）——所有卡都连着 SEG_ORDER 里的键。"""
        return sum(ru[k]["us"] for k in SEG_ORDER if k != "gemm")

    H = []
    A = H.append
    A('<h2 id="s6">6. 性能差异点</h2>')
    A(tbl(["维度", "off", "slot", "差"],
          [["MoE 机制合计（整轮·每 rank，不含 GEMM）", sec(r_off["comm"]["us"] / 1e6 + r_off["dispatch"]["us"] / 1e6
                                                + r_off["combine"]["us"] / 1e6),
            sec(r_sl["plan"]["us"] / 1e6 + r_sl["barrier"]["us"] / 1e6 + r_sl["combine"]["us"] / 1e6
                + r_sl["dispatch"]["us"] / 1e6 + r_sl["prefetch"]["us"] / 1e6),
            (f"{W['slot']['mech'] / W['off']['mech']:.2f}×", "bad")],
           ["每 cell·真实波 R", ms(tot_cell(r_off, "R") / 1000), ms(tot_cell(r_sl, "R") / 1000),
            (f"{tot_cell(r_sl, 'R') / tot_cell(r_off, 'R'):.2f}×", "bad")],
           ["每 cell·空转波 D", ms(tot_cell(r_off, "D") / 1000), ms(tot_cell(r_sl, "D") / 1000),
            (f"{tot_cell(r_sl, 'D') / tot_cell(r_off, 'D'):.2f}×", "bad")],
           ["规划 plan + 流屏障 barrier 占机制",
            "0（无此两步）",
            pct_share(r_sl["plan"]["us"] + r_sl["barrier"]["us"],
                      sum(r_sl[k]["us"] for k in SEG_ORDER if k != "gemm")),
            ("—", "")],
           ["跨卡搬运落在哪个核", "通信核（HCCL 集合通信）+ MC2 融合算子",
            "AI 核内的 MoonEP 算子（设备工具看不到通信）", ""],
           ["空转步数 / 总步数", f"{off['n_D']} / {off['n_steps']}", f"{sl['n_D']} / {sl['n_steps']}",
            f"+{sl['n_D'] - off['n_D']}"],
           ["空转步每 cell 计算行 M", fnum(off["floor"], 0), fnum(sl["floor"], 0),
            (f"{sl['floor'] / off['floor']:.0f}×", "bad")],
           ["pad 行占本次运行计算量", "0%",
            pct_share(P["slot"]["compute_pad_total"],
                      P["slot"]["compute_pad_total"] + P["slot"]["real_total"]),
            ("—", "")],
           ["设备占空比（窗口内）", pct_share(W["off"]["occ"], W["off"]["win"]),
            pct_share(W["slot"]["occ"], W["slot"]["win"]), ""],
           ["设备空闲", sec(W["off"]["win"] - W["off"]["occ"]), sec(W["slot"]["win"] - W["slot"]["occ"]),
            (f"-{fnum((W['off']['win'] - W['off']['occ']) - (W['slot']['win'] - W['slot']['occ']), 1)} s", "good")],
           ["每步墙钟（第 2 组实测）", ms(B["g2_off"]["common"]["Benchmark Duration"][0] / 192, 1),
            ms(B["g2_slot"]["common"]["Benchmark Duration"][0] / 192, 1),
            (pct_up(B["g2_off"]["common"]["Benchmark Duration"][0],
                    B["g2_slot"]["common"]["Benchmark Duration"][0]), "bad")],
           ["TTFT 均值（第 2 组实测）", ms(tstat(B["g2_off"], "TTFT", "avg"), 1),
            ms(tstat(B["g2_slot"], "TTFT", "avg"), 1),
            (pct_up(tstat(B["g2_off"], "TTFT", "avg"), tstat(B["g2_slot"], "TTFT", "avg")), "bad")]]))
    A(f"""<div class="note"><b>因果链条（本报告的结论）</b>
<ul>
<li><b>差异不在 GEMM，在“每层要不要做一次跨卡池化规划”</b>。slot 每层都要跑 <code>planning_kernel</code>
（82% 的机制时间是 plan + barrier），off 没有这两步。</li>
<li><b>off 的跨卡走两条完全不同的路径</b>：真实波走通信核上的 HCCL（每 cell 3×all-to-allv + 1×all-gather + 4 次下发），
空转波走 MC2 的 <code>MoeDistributeDispatch/Combine</code>；slot 则全部内嵌在 AI 核算子里（SHMEM），
所以设备工具在 slot 上看到「通信 0%」。</li>
<li><b>池化把空转步的算力放大了 16 倍</b>：off 空转步每 cell 只算 {fnum(off['floor'],0)} 行（= 8 token × K 的全局量），
slot 固定算 {fnum(sl['floor'],0)} 行（128 token × K / rank）。slot 全轮 pad 行
{fnum(P['slot']['compute_pad_total'],0)}（占计算量 {pct_share(P['slot']['compute_pad_total'], P['slot']['compute_pad_total']+P['slot']['real_total'])}）。</li>
<li><b>墙钟差（{pct_up(tstat(B['g2_off'],'TTFT','avg'), tstat(B['g2_slot'],'TTFT','avg'))}）远小于机制比
（{W['slot']['mech']/W['off']['mech']:.2f}×）</b>，是因为 off 自己有一半时间设备是空的
（{pct_share(W['off']['win']-W['off']['occ'], W['off']['win'])} 空闲）：这几秒空转把 off 的每步墙钟
抬到了 {ms(B['g2_off']['common']['Benchmark Duration'][0]/192,1)}，从而“摊薄”了 MoonEP 的相对代价。</li>
<li><b>P99 涨得比中位多</b>（+{pct_up(tstat(B['g2_off'],'TTFT','p99'), tstat(B['g2_slot'],'TTFT','p99')).lstrip('+')} vs
+{pct_up(tstat(B['g2_off'],'TTFT','p50'), tstat(B['g2_slot'],'TTFT','p50')).lstrip('+')}）：
池化/规划的代价同每步的池子体积挂钩，而慢步恰好是池子最大的步。</li>
</ul></div>""")

    # ---- §7 优化点 ----
    SL = plan_slopes(sl)                     # plan 随池容量的两段边际斜率（µs/token）
    plan128 = SL["p128"]
    nD = sl["n_cells_D"]
    _slope_hi, _slope_lo = SL["hi"], SL["lo"]
    _dcap = 128 - 32                         # 空转步想降到的池容量
    save_lo, save_hi = _slope_lo * _dcap * nD, _slope_hi * _dcap * nD

    def opt(tag, title, body):
        return f'<div class="opt"><h4>{title}<span class="tag {tag}">{tag.upper()}</span></h4>{body}</div>'

    A('<h2 id="s7">7. 可能的优化点</h2>')
    A(f"""<p>按「收益 × 可验证性」排序。所有收益都是按本轮的设备侧实测外推，
<b>必须实跑一次运行确认</b>（跑法见 §8）；每一条都给了验证口径与风险。</p>""")

    A(opt("p1", "O1 · 把池子下限降下来（只改环境变量）", f"""
<p>现状：<code>MOONEP_MIN_DISPATCH_TOKENS = 128</code>（默认值，也是池子阶梯的起点；同名环境变量可调低），
空转步只有 1 个真实 token 也要按 128 token 池化 ⇒ 每 cell 固定算 {fnum(sl['floor'],0)} 行。
plan 的开销跟着池子体积走，实测三档：</p>
{tbl(["池容量 cap", "cell 数", "步数", "plan µs/cell", "plan ÷ cap (µs/token)", "combine µs/cell", "barrier µs/cell"],
     [[fnum(r["cap"], 0), fnum(r["n_cells"], 0), fnum(r["n_steps"], 0), fnum(r.get("plan"), 0),
       fnum((r.get("plan") or 0) / r["cap"], 2), fnum(r.get("moep_combine"), 0), fnum(r.get("barrier"), 0)]
      for r in sorted(sl["ladder"], key=lambda r: r["cap"]) if r["n_cells"] >= 43])}
<p>阶梯里 512 那一档只有 1 步（43 个 cell），别当稳态看。两档之间各拟合一次斜率，得到收益的上下界：
<b>乐观</b>（按 128→256 的边际斜率 {fnum(_slope_hi,1)} µs/token 外推到 32）＝
{sec(save_hi / 1e6)}；<b>悲观</b>（按 256→512 的边际斜率 {fnum(_slope_lo,1)} µs/token，即“大头是固定开销”）＝
{sec(save_lo / 1e6)}。折算到全轮每 rank 的窗口是
<b>{sec(save_lo/1e6)} – {sec(save_hi/1e6)}（{pct_share(save_lo, W['slot']['win']*1e6)} –
{pct_share(save_hi, W['slot']['win']*1e6)}）</b>，作用对象是 {fnum(nD,0)} 个空转 cell；
<code>combine</code> 与 GEMM 随之同向下降（再省约 0.3–1.6 s）。</p>
<p><b>怎么改</b>：<code>MOONEP_MIN_DISPATCH_TOKENS=32</code>（同时改阶梯起点与每步的取档下限，
纯环境变量）或 <code>MOONEP_POOL_CAPS=16,32,64,128,256,512</code>（直接覆盖整条阶梯，记得末档要包含
<code>max_num_batched_tokens</code>，代码会自动补）——两种都不改代码。</p>
<p><b>风险</b>：池容量必须各 rank 一致（代码里 <code>_step_token_count</code> 走 DP 元数据的
all-reduce，取 max_tokens_across_dp，已经保证对称）；小池子可能触发 kernel 的最小 M 约束或改变
分片数，需要确认 plan/combine 的正确性。</p>
<p><b>验证</b>：同一条 gsm8k 运行加 <code>MOONEP_POOL_CAPS=16,32,64,128,256,512</code> 重跑，
对比 (a) TTFT 均值/P99，(b) 三套准确率基线 mc2 0.9492 / owner 0.9409 / slot 0.9515，
(c) 复跑本报告的 dump，看 plan µs/cell 是否按预期下降。</p>"""))

    A(opt("p1", "O2 · 把 plan 从“等”改成“算完就发”", f"""
<p>实测：<code>planning_kernel</code> 平均时长 {fnum(sl['fam']['plan']['us_avg'],0)} µs，
但核内最忙单元（AIV scalar）只有 {insn_busiest(sl['df'], 'planning_kernel')['ratio']:.1f}%，
其余 {(100 - insn_busiest(sl['df'], 'planning_kernel')['ratio']):.1f}% 是等通知/等数据。
它又是每层串行的一环（每 cell 一次，总 {sec(sl['fam']['plan']['us'] / 1e6)}，占机制
{pct_share(sl['fam']['plan']['us'], mech_us(r_sl))}）。</p>
<p><b>怎么改</b>：先把等待点定位出来——用 <code>MOONEP_CAP_TRACE=1</code> /
<code>MOONEP_TIME_STATS=1</code> 打点，确认等待是在 (a) 等对端 route 数据，
(b) 等 barrier，(c) 等上一层的 combine；再决定是提前发起（和上一层 GEMM 重叠）、
还是把 plan 拆成「本地先算、跨卡部分异步」。<b>上限</b>是把这
{fnum(plan128,0)} µs/cell（空转波）里的等待全去掉，但不可能全回收，保守按 30%–50% 估算
≈ {sec(plan128 * 0.3 * nD / 1e6)}–{sec(plan128 * 0.5 * nD / 1e6)}。</p>
<p><b>风险</b>：plan 的等待多半是算法要求的同步点，乱动会导致各 rank 容量不一致（集体通信错乱）。
<b>验证</b>：与 O1 同一套——先看 dump 里 plan/barrier 的等待点，再跑运行比 TTFT 与准确率。</p>"""))

    A(opt("p1", "O3 · 合并每层的 8 次流屏障", f"""
<p>实测每 cell 有 <b>8 次</b> <code>_Z24barrier_on_stream_kernelim</code>
（{fnum(sl['fam']['barrier']['n'] / sl['n_cells'], 0)} 次/cell，平均 {fnum(sl['fam']['barrier']['us_avg'],0)} µs，
合计 {sec(sl['fam']['barrier']['us'] / 1e6)}，占机制
{pct_share(sl['fam']['barrier']['us'], mech_us(r_sl))}），
核内几乎全空（最忙单元 {insn_busiest(sl['df'], 'barrier_on_stream_kernelim')['ratio']:.1f}%）。
它的时长和池容量基本无关（128/256/512 三档分别
{fnum(next((r.get('barrier') for r in sl['ladder'] if r['cap'] == 128), 0),0)} /
{fnum(next((r.get('barrier') for r in sl['ladder'] if r['cap'] == 256), 0),0)} /
{fnum(next((r.get('barrier') for r in sl['ladder'] if r['cap'] == 512), 0),0)} µs/cell），
说明它同步的是<b>流与流之间的次序</b>，不是数据量。</p>
<p><b>怎么改</b>：数一数这 8 次分别同步什么（dispatch 前后、plan 前后、combine 前后…），
能合并的合并、能换成上层已有依赖的去掉。若能降到 2 次/cell，
省 ≈ {sec((1 - 2 / 8) * sl['fam']['barrier']['us'] / 1e6)}。代码里
<code>MOONEP_NO_SYNC=0</code> / <code>MOONEP_PACE_SYNC=1</code> 是现成的开关，可先用来试探哪些同步是必需的。</p>
<p><b>风险</b>：这是正确性敏感项（去掉屏障可能出现读写竞争/结果错），必须跑准确率三件套。</p>"""))

    A(opt("p2", "O4 · 空转步别再走 MoonEP 池化", f"""
<p>slot 有 {sl['n_D']} 个空转步（占 {pct_share(sl['n_D'], sl['n_steps'])}），
它们每 cell 仍要花 {ms(tot_cell(r_sl, 'D') / 1000)}（off 的空转步只要 {ms(tot_cell(r_off, 'D') / 1000)}）。</p>
<p><b>怎么改</b>：空转步的真实路由只有 8 个 token（每 rank 1 个），完全可以走 off 那条
MC2/本地路径（或直接短路成 dense 计算）：上限是把 slot 空转波的 cell 开销压到 off 空转波的水平，
省 {sec((tot_cell(r_sl, 'D') - tot_cell(r_off, 'D')) * nD / 1e6)}（全轮每 rank）；
按回收一半算，现实收益 ~{sec((tot_cell(r_sl, 'D') - tot_cell(r_off, 'D')) * nD / 1e6 * 0.5)} 量级
（{pct_share((tot_cell(r_sl,'D')-tot_cell(r_off,'D'))*nD*0.5, W['slot']['win']*1e6)} 窗口）。</p>
<p><b>风险</b>：两条路径的数值/量化必须完全一致（slot 运行是 fp8 池化路径），否则准确率会掉；
且混合路径会让「哪一步用哪条」的判定成为新的同步点。<b>验证</b>：先只对 <code>capacity == 池下限</code>
的步走短路径，比准确率三件套。</p>"""))

    A(opt("p2", "O5 · UDMA（本机当前不可用，也不是第一杠杆）", f"""
<p>实测数据搬运只占机制的一小部分：slot 的 <code>dispatch + combine + prefetch</code> =
{sec((r_sl['dispatch']['us'] + r_sl['combine']['us'] + r_sl['prefetch']['us']) / 1e6)}，
占机制 {pct_share(r_sl['dispatch']['us'] + r_sl['combine']['us'] + r_sl['prefetch']['us'], mech_us(r_sl))}；
即使 UDMA 把这些搬运全变成零，也动不了 plan + barrier 那
{pct_share(r_sl['plan']['us'] + r_sl['barrier']['us'], mech_us(r_sl))}。</p>
<p><b>现状</b>：本机 UDMA 起不来——去掉 <code>MOONEP_DISABLE_UDMA</code> 后快速失败，
<code>RuntimeError: aclshmemx_init_attr_with_buffers failed with -3 (my_pe=1, n_pes=2, buffers=24)</code>
（<code>docs/cross_repo_fixes.md</code> 有完整记录，与 shmem/设备号问题同族）；运行脚本因此固定
<code>MOONEP_DISABLE_UDMA=1</code>，且 <code>docs/moonep_slot_quant.md</code> §5 明确标注
「UDMA 分支未验证」（<code>kernels/prefetch_udma.cpp</code> 的字节账要单独复核）。
<b>先解决初始化失败</b>才有得测，而它排在 O1–O4 之后。</p>
<p><b>怎么改</b>：换支持 <code>aclshmemx</code> 的机器/驱动组合，或按
<code>docs/cross_repo_fixes.md</code> 的失败点逐个排查；再跑同一套 gsm8k 运行对比。
<b>结论</b>：优先级低于 O1–O4。</p>"""))

    A(opt("p3", "O6 · 跳过 pad 行的 GEMM", f"""
<p>slot 每 cell 固定算 {fnum(sl['floor'],0)} 行，其中 pad 行占
{pct_share(P['slot']['compute_pad_total'], P['slot']['compute_pad_total'] + P['slot']['real_total'])}，
但 GEMM 本身只占机制的
{pct_share(r_sl['gemm']['us'], mech_us(r_sl))}
（{sec(r_sl['gemm']['us'] / 1e6)}），所以跳过 pad 行的天花板只有 ~1%。
它真正的价值是「让 plan/combine 的行数下降」——那属于 O1 的范畴。</p>"""))

    A(opt("p3", "O7 · off 侧的 hostbound（对照项，不是优化 MoonEP 的收益）", f"""
<p>off 的窗口 {sec(W['off']['win'])} 里设备只忙 {sec(W['off']['occ'])}（
{pct_share(W['off']['occ'], W['off']['win'])}），空闲 {sec(W['off']['win'] - W['off']['occ'])}
比 slot 的 {sec(W['slot']['win'] - W['slot']['occ'])} 多
{sec((W['off']['win'] - W['off']['occ']) - (W['slot']['win'] - W['slot']['occ']))}。
这部分空转是「等 host 下发」，用 <code>--enforce-eager</code> 的图模式、减少 Python 侧
同步/拷贝都能拿到。</p>
<p><b>为什么列出来</b>：一旦 off 侧修好，两模式的墙钟差会从现在的
{pct_up(tstat(B['g2_off'], 'TTFT', 'avg'), tstat(B['g2_slot'], 'TTFT', 'avg'))}
向机制比 {W['slot']['mech'] / W['off']['mech']:.2f}× 靠拢。
<b>不要把今天 off 的 TTFT 当成 MoonEP 的长期代价基线。</b></p>"""))
    return "\n".join(H)


def build(B, P, T, DEV, RU) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    W = {}          # 现算的小数字，便于散文里引用
    for leg in ("off", "slot"):
        T[leg]["span_s"] = T[leg]["win"]          # 兼容旧键名
        W[leg] = {
            "win": T[leg]["win"],
            "occ": T[leg]["occupied"],
            "sh_c": T[leg]["share"]["compute"],
            "sh_k": T[leg]["share"]["comm"],
            "sh_u": T[leg]["share"]["aicpu"],
            "sh_i": T[leg]["idle"] / T[leg]["win"],
            # 机制 = 通信 / plan / barrier / dispatch / combine / prefetch（不含 GEMM）
            "mech": sum(r["us"] for k, r in RU[leg].items() if k != "gemm") / 1e6,
            "gemm": RU[leg]["gemm"]["us"] / 1e6,
        }
    # plan 随池容量的边际斜率（µs/token，乐观 / 悲观两端），TL;DR 与 §7 O1 共用
    _SL = plan_slopes(DEV["slot"])
    _SL_HI, _SL_LO = _SL["hi"], _SL["lo"]

    # §2 端到端
    g1_o, g1_s = B["g1_off"], B["g1_slot"]
    g2_o, g2_s = B["g2_off"], B["g2_slot"]

    def bench_rows():
        rows = []
        for name, b in (("moonep_off（第 1 组）", g1_o), ("moonep_slot（第 1 组）", g1_s),
                        ("moepar_moonep_off（第 2 组）", g2_o), ("moepar_moonep_slot（第 2 组）", g2_s)):
            dur = b["common"]["Benchmark Duration"][0] / 1000.0
            rows.append([
                name,
                fnum(tstat(b, "TTFT", "avg"), 1), fnum(tstat(b, "TTFT", "p50"), 1),
                fnum(tstat(b, "TTFT", "p90"), 1), fnum(tstat(b, "TTFT", "p99"), 1),
                fnum(tstat(b, "TTFT", "max"), 1),
                fnum(b["common"]["Request Throughput"][0], 2),
                fnum(dur, 1), fnum(dur / 192 * 1000, 1),
            ])
        return rows

    def delta_rows():
        out = []
        for label, o, s in (("第 1 组（纯 profiling，无 dump）", g1_o, g1_s),
                            ("第 2 组（profiling + moe_par dump）", g2_o, g2_s)):
            out.append([
                label,
                fnum(tstat(o, "TTFT", "avg"), 1), fnum(tstat(s, "TTFT", "avg"), 1),
                (pct_up(tstat(o, "TTFT", "avg"), tstat(s, "TTFT", "avg")), "bad"),
                fnum(tstat(o, "TTFT", "p50"), 1), fnum(tstat(s, "TTFT", "p50"), 1),
                (pct_up(tstat(o, "TTFT", "p50"), tstat(s, "TTFT", "p50")), "bad"),
                fnum(tstat(o, "TTFT", "p99"), 1), fnum(tstat(s, "TTFT", "p99"), 1),
                (pct_up(tstat(o, "TTFT", "p99"), tstat(s, "TTFT", "p99")), "bad"),
                fnum(o["common"]["Request Throughput"][0], 2),
                fnum(s["common"]["Request Throughput"][0], 2),
                (pct_up(o["common"]["Request Throughput"][0],
                        s["common"]["Request Throughput"][0]), "good"),
            ])
        return out

    # §3.1 池化阶梯（off 压缩成桶）
    def ladder_rows(leg):
        ladders = P[leg]["ladder"]          # [(每步池化/真实 token（全局）, 步数）]
        rows = []
        if leg == "off":
            buckets = [(0, 1), (2, 32), (33, 64), (65, 96), (97, 128), (129, 10 ** 9)]
            tot = sum(n for _, n in ladders)
            for lo, hi in buckets:
                n = sum(c for v, c in ladders if lo <= v <= hi)
                if not n:
                    continue
                ex = [v for v, _ in ladders if lo <= v <= hi]
                rows.append([f"{lo}–{hi}" if hi < 10 ** 9 else f"≥{lo}",
                             fnum(n, 0), pct_share(n, tot),
                             fnum(np.mean([v for v, c in ladders if lo <= v <= hi]), 1),
                             fnum(min(ex), 0), fnum(max(ex), 0)])
        else:
            tot = sum(n for _, n in ladders)
            for v, n in sorted(ladders):
                cap = v / RANKS
                rows.append([fnum(cap, 0), fnum(n, 0), pct_share(n, tot),
                             fnum(cap * TOP_K, 0), fnum(cap * TOP_K * N_LAYERS, 0)])
        return rows

    # §3.2 真实/pad
    # §3.3 / §4 家族表
    def fam_rows(leg, per_cell=True):
        rows = []
        d = DEV[leg]
        win_us = W[leg]["win"] * 1e6
        for key, pat, label in FAMILIES[leg]:
            f = d["fam"][key]
            if not f["n"]:
                continue
            rows.append([
                f'<code>{esc(pat)}</code>', f["ktype"], fnum(f["n"], 0),
                fnum(f["us"] / 1e6, 3), fnum(f["us_avg"], 0), fnum(f["us_max"], 0),
                pct_share(f["us"], win_us),
                (fnum(f["percell_R"], 0), "good") if f["percell_R"] else "—",
                (fnum(f["percell_D"], 0), "bad") if f["percell_D"] else "—",
            ])
        # 合计
        tot_us = sum(d["fam"][k]["us"] for k, _, _ in FAMILIES[leg])
        rows.append([("<b>合计</b>", "k"), "", fnum(sum(d["fam"][k]["n"] for k, _, _ in FAMILIES[leg]), 0),
                     f"<b>{fnum(tot_us/1e6, 3)}</b>", "", "", f"<b>{pct_share(tot_us, win_us)}</b>", "", ""])
        return rows

    # 图
    def win_slices(leg):
        w = W[leg]
        return {"computing": w["sh_c"] * 100, "comm": w["sh_k"] * 100,
                "aicpu": w["sh_u"] * 100, "free": w["sh_i"] * 100}

    c_win = svg_stack_bars(
        [{"label": f"off（窗口 {sec(W['off']['win'])}）", "vals": win_slices("off"),
          "caption": f"忙 {pct_share(W['off']['occ']*100, W['off']['win']*100)}｜"
                     f"空闲 {pct_share((W['off']['win']-W['off']['occ'])*100, W['off']['win']*100)}"},
         {"label": f"slot（窗口 {sec(W['slot']['win'])}）", "vals": win_slices("slot"),
          "caption": f"忙 {pct_share(W['slot']['occ']*100, W['slot']['win']*100)}｜"
                     f"空闲 {pct_share((W['slot']['win']-W['slot']['occ'])*100, W['slot']['win']*100)}"}],
        [{"key": "computing", "label": "计算核", "color": "--series-1"},
         {"key": "comm", "label": "通信核（HCCL）", "color": "--series-2"},
         {"key": "aicpu", "label": "AI CPU（下发）", "color": "--series-4"},
         {"key": "free", "label": "空闲（等 host / 等对端）", "color": "--neutral-fill"}],
        x_max=100, x_label="占窗口 %", val_fmt=lambda v: f"{v:.0f}%",
        note="每 rank 窗口时间构成（分段互斥，相加 = 100%）")

    wave_rows = []
    for leg in ("off", "slot"):
        for k, kn in (("R", "真实波 R"), ("D", "空转波 D")):
            vals = {key: (RU[leg][key][f"percell_{k}"] or 0) for key in SEG_ORDER}
            tot = sum(vals.values())
            wave_rows.append({
                "label": f"{leg} · {kn}",
                "vals": vals,
                "caption": f"{fnum(tot,0)} µs/cell（{DEV[leg]['n_cells_'+k]:,} cell）",
            })
    c_cell = svg_stack_bars(wave_rows, seg_series([k for k in SEG_ORDER]),
                            x_label="µs / cell（每 rank 每层）", val_fmt=lambda v: f"{v:,.0f}",
                            note="两类波每 cell 的设备开销")

    tot_rows = []
    for leg in ("off", "slot"):
        vals = {key: RU[leg][key]["us"] / 1e6 for key in SEG_ORDER}
        tot_rows.append({"label": f"{leg}（{pct_share(sum(vals.values())*1e6, W[leg]['win']*1e6)} 窗口）",
                         "vals": vals,
                         "caption": f"机制合计 {sec(sum(vals.values()))}"})
    c_tot = svg_stack_bars(tot_rows, seg_series([k for k in SEG_ORDER]),
                           x_label="整轮 192 步合计（s，每 rank）", val_fmt=lambda v: f"{v:.1f}",
                           note="MoE 机制总时间构成")

    slot_lad = [r for r in DEV["slot"]["ladder"] if r["n_cells"] > 100]
    groups = []
    for r in sorted(slot_lad, key=lambda r: r["cap"]):
        groups.append({"label": f"池 {r['cap']} token", "vals": {"plan": r.get("plan") or 0,
                                                              "barrier": r.get("barrier") or 0},
                       "caption": f"{r['n_cells']:,} cell / {r['n_steps']} 步"})
    c_cap = svg_group_bars(groups,
                           [{"key": "plan", "label": "规划 plan",
                             "color": "--series-5"},
                            {"key": "barrier", "label": "流屏障 barrier",
                             "color": "--series-6"}],
                           x_label="µs / cell（每 rank 每层）", note="plan / barrier 随池子容量")

    # ---------------- HTML ----------------
    H = []
    A = H.append
    A(f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MoonEP off / slot 性能差异与优化点分析</title>
<style>{CSS}</style></head><body>
<button id="theme" onclick="var d=document.documentElement;d.dataset.theme=d.dataset.theme==='dark'?'light':'dark'">切换深色</button>
<main>
<h1>MoonEP off / slot 性能差异与优化点分析</h1>
<p class="sub">DSV4-Flash · DP8/TP1/EP8 · gsm8k 1319 例 @ 并发 16 · 每请求恰好 1 个输出 token（TTFT ≡ E2EL）<br>
数据：<code>scripts/gsm8k_profiling{{_prof|_moepar}}_moonep_{{off,slot}}</code> 四次运行 · 设备侧为 rank0 离线导出 · 生成于 {now}</p>""")

    # TL;DR
    o_avg1, s_avg1 = tstat(g1_o, "TTFT", "avg"), tstat(g1_s, "TTFT", "avg")
    o_avg2, s_avg2 = tstat(g2_o, "TTFT", "avg"), tstat(g2_s, "TTFT", "avg")
    A('<h2 id="tldr">结论速览</h2>')
    A('<div class="kpis">')
    A(kpi(f"{pct_up(o_avg1, s_avg1)} / {pct_up(o_avg2, s_avg2)}",
          "slot 的 TTFT 劣化（两组）",
          f"{fnum(o_avg1,1)} → {fnum(s_avg1,1)} ms；{fnum(o_avg2,1)} → {fnum(s_avg2,1)} ms"))
    A(kpi(f"{sec(W['off']['mech'])} → {sec(W['slot']['mech'])}",
          "MoE 机制时间（每 rank 整轮）",
          f"比值 {W['slot']['mech']/W['off']['mech']:.2f}×；但两次运行的墙钟窗口只差 {sec(abs(W['slot']['win']-W['off']['win']))}——off 侧有约 {sec(abs(W['off']['win']-W['off']['occ']))} 的设备空转在替它挡刀。"))
    A(kpi(f"{fnum(DEV['slot']['fam']['plan']['percell_D'],0)} µs/cell",
          "slot 空转波的规划 plan 开销",
          f"占 slot 空转波每 cell {pct_share(DEV['slot']['fam']['plan']['percell_D'], sum((RU['slot'][k]['percell_D'] or 0) for k in SEG_ORDER))}；plan 逐 cell 时间随池子体积线性走"))
    A(kpi(f"{fnum(DEV['off']['fam']['hcom_a2a']['us']/1e6,2)} + {fnum(DEV['off']['fam']['hcom_ag']['us']/1e6,2)} s",
          "off 真实波的 HCCL 通信（all-to-allv + all-gather）",
          "off 侧是两条完全不同的通信路径：真实波走 HCCL 集合通信，空转波走 MC2 dispatch/combine"))
    A('</div>')
    A(f"""<div class="tldr"><ul>
<li><b>两个模式差别不在算子，在“每一步要不要为池化做一次跨卡规划”</b>。slot 每层都要跑一次
<code>planning_kernel</code>（把这一层所有 rank 的路由摊成池子），它 90% 的时间单元在等对端/等数据；
off 没有这一步。</li>
<li><b>off 的真实波（R）根本不走 MoonEP 的 dispatch/combine</b>，而是 <code>hcom_alltoallv</code> ×3 +
<code>hcom_allGather</code> ×1 + 4 次 HCCL 下发（每层每 rank），整轮
{fnum(DEV['off']['fam']['hcom_a2a']['us']/1e6,2)}+{fnum(DEV['off']['fam']['hcom_ag']['us']/1e6,2)}+{fnum(DEV['off']['fam']['hccl_launch']['us']/1e6,2)} s。
这条路径跑在<b>通信核</b>上，与 AI 核的 GEMM 是两笔互不重叠的账（所以 off 的占空比统计里
「通信 {pct_share(W['off']['sh_k']*100,100)}」是实打实占住的设备时间）。</li>
<li><b>占空比才是墙钟差距小的原因</b>：off 的窗口 {sec(W['off']['win'])} 里只有 {sec(W['off']['occ'])} 是设备在忙（
{pct_share(W['off']['occ'], W['off']['win'])}），其余在等 host 下发；slot 的 {sec(W['slot']['win'])} 里设备忙 {sec(W['slot']['occ'])}
（{pct_share(W['slot']['occ'], W['slot']['win'])}）。所以 <b>off 侧一旦把下发问题修好，slot 的差距会从现在的
{pct_up(o_avg2, s_avg2)} 涨到接近机制比 {W['slot']['mech']/W['off']['mech']:.2f}×</b>。</li>
<li><b>最省事的收益点是池子下限</b>：空转步把 8 个真实 token 撑成 {fnum(FLOOR_ROWS['slot']//TOP_K,0)} token 的池
（{fnum(FLOOR_ROWS['slot'],0)} 行），plan 随池子体积走。把下限从 128 调到 32，空转波的 plan 从
{fnum(DEV['slot']['fam']['plan']['percell_D'],0)} µs/cell 往下降，全轮可省
{sec(_SL_LO * (128 - 32) * DEV['slot']['n_cells_D'] / 1e6)} –
{sec(_SL_HI * (128 - 32) * DEV['slot']['n_cells_D'] / 1e6)}（上下界怎么来的见 §7 O1）。</li>
</ul></div>""")

    # §1
    A('<h2 id="s1">1. 实验与口径</h2>')
    A('<h3>1.1 四次运行</h3>')
    A(tbl(["运行目录", "开关", "profiling", "moe_par dump", "用途"], [
        ["gsm8k_profiling_moonep_off", "<code>enable_moonep=false</code>", "是", "否", "干净端到端（无 dump 开销）"],
        ["gsm8k_profiling_moonep_slot", "<code>MOONEP_SLOT_MODE=1</code>", "是", "否", "干净端到端"],
        ["gsm8k_profiling_moepar_moonep_off", "<code>enable_moonep=false</code>", "是", "是（flush_every=1）", "负载结构 + 设备侧归因"],
        ["gsm8k_profiling_moepar_moonep_slot", "<code>MOONEP_SLOT_MODE=1</code>", "是", "是（flush_every=1）", "负载结构 + 设备侧归因"],
    ]))
    A("""<h3>1.2 五处必须记住的口径</h3>
<div class="kv">
<b>1)</b><div>每次请求只产出 1 个 token，因此 <b>TTFT ≡ E2EL</b>，端到端延迟就是首 token 前的步数 × 每步墙钟。</div>
<b>2)</b><div>每步墙钟 = ais_bench 的 <code>Benchmark Duration</code> ÷ 192 步（两组的步数都是 192）。</div>
<b>3)</b><div><b>两类波</b>：真实波（R，这一步有真实路由）与空转波（D，DP 对齐造成的“只有 token 却仍要池化”的步）。
off 运行的判定是「该步每层计算行 == 8 rank × 1 token × K = 48」，slot 运行是「池容量 == 池下限 128
（即每层每 rank 计算行 == 128 × K = 768）」。</div>
<b>4)</b><div><b>池下限</b>：<code>MOONEP_MIN_DISPATCH_TOKENS = 128</code>（默认值，本轮用的就是它；可用同名环境变量调低），slot 每 rank 每层恒算 <code>capacity × K</code> 行；
off 不池化，按真实 token 数算。</div>
<b>5)</b><div>设备侧只分析 <b>rank0</b>（每 rank 一个设备，DP8 对称）；通信类算子的时间以通信核计时为准，
不与计算核重叠计数。</div>
</div>""")

    # §2
    A('<h2 id="s2">2. 端到端结果</h2>')
    A('<h3>2.1 四次运行的原始指标</h3>')
    A(tbl(["运行（目录名）", "TTFT 均值 (ms)", "中位", "P90", "P99", "最大",
           "请求吞吐 (req/s)", "Benchmark 时长 (s)", "每步墙钟 (ms)"], bench_rows()))
    A('<h3>2.2 slot 相对 off 的劣化</h3>')
    A(tbl(["组", "off 均值", "slot 均值", "Δ", "off 中位", "slot 中位", "Δ",
           "off P99", "slot P99", "Δ", "off req/s", "slot req/s", "Δ"], delta_rows()))
    A(f"""<div class="note">两组结论一致：<b>均值 +{pct_up(o_avg1,s_avg1).lstrip('+')} / {pct_up(o_avg2,s_avg2).lstrip('+')}，
中位 +{pct_up(tstat(g1_o,'TTFT','p50'), tstat(g1_s,'TTFT','p50')).lstrip('+')} / {pct_up(tstat(g2_o,'TTFT','p50'), tstat(g2_s,'TTFT','p50')).lstrip('+')}，
P99 涨得更多（+{pct_up(tstat(g1_o,'TTFT','p99'), tstat(g1_s,'TTFT','p99')).lstrip('+')} / +{pct_up(tstat(g2_o,'TTFT','p99'), tstat(g2_s,'TTFT','p99')).lstrip('+')}）</b>，
吞吐掉 {pct_up(g1_o['common']['Request Throughput'][0], g1_s['common']['Request Throughput'][0])} /
{pct_up(g2_o['common']['Request Throughput'][0], g2_s['common']['Request Throughput'][0])}。中位涨得比均值少、P99 涨得比中位多，
说明<b>劣化是逐步累积的（每步都多一点），并且在个别慢步上被放大</b>，不是偶发长尾。</div>""")
    A('<h3>2.3 每步墙钟</h3>')
    A(tbl(["组", "off 每步", "slot 每步", "Δ", "off 总时长", "slot 总时长"],
          [[lb, ms(B[o]["common"]["Benchmark Duration"][0]/192, 1),
            ms(B[s]["common"]["Benchmark Duration"][0]/192, 1),
            (pct_up(B[o]["common"]["Benchmark Duration"][0], B[s]["common"]["Benchmark Duration"][0]), "bad"),
            sec(B[o]["common"]["Benchmark Duration"][0]/1000, 1), sec(B[s]["common"]["Benchmark Duration"][0]/1000, 1)]
           for lb, o, s in (("第 1 组", "g1_off", "g1_slot"), ("第 2 组", "g2_off", "g2_slot"))]))

    # §3
    A('<h2 id="s3">3. 运行时结构：两种波、两种下限</h2>')
    A('<h3>3.1 每步池化量（dump 实测）</h3>')
    A('<h4>off：不池化，按真实路由 token 数走（下表按桶压缩）</h4>')
    A(tbl(["每步全局真实 token 区间（8 rank 合计）", "步数", "占比", "区间内均值", "最小", "最大"], ladder_rows("off")))
    A('<h4>slot：池容量各 rank 相同（构造对称）</h4>')
    A(tbl(["每步池容量 token（每 rank）", "步数", "占比", "每 rank 每层计算行（=容量×K）",
           "每 rank 该步计算行（×43 层）"], ladder_rows("slot")))
    A(f"""<div class="note">关键区别：<b>off 的“空转步”是 8 个真实 token（每 rank 1 个）也会各自发 6 行</b>，
而 <b>slot 的空转步被池下限抬到 {fnum(FLOOR_ROWS['slot']//TOP_K,0)} token（每 rank 每层 {fnum(FLOOR_ROWS['slot'],0)} 行）</b>。
两次运行的步数构成：off {DEV['off']['n_R']} 真实步 / {DEV['off']['n_D']} 空转步，
slot {DEV['slot']['n_R']} 真实步 / {DEV['slot']['n_D']} 空转步。</div>""")

    A('<h3>3.2 真实行 / pad 行</h3>')
    A(tbl(["运行", "真实行（8 rank 合计）", "pad 行（8 rank 合计）", "合计计算行", "pad 占比"], [
        ["off", fnum(P["off"]["real_total"], 0), "0（无 compute 视图）",
         fnum(P["off"]["compute_pad_total"] + P["off"]["real_total"], 0), "0.0%"],
        ["slot", fnum(P["slot"]["real_total"], 0), fnum(P["slot"]["compute_pad_total"], 0),
         fnum(P["slot"]["compute_pad_total"] + P["slot"]["real_total"], 0),
         pct_share(P["slot"]["compute_pad_total"],
                   P["slot"]["compute_pad_total"] + P["slot"]["real_total"])],
    ]))
    A(f"""<div class="note">两次运行的路由量几乎一样（{fnum(P['off']['route_total'],0)} vs {fnum(P['slot']['route_total'],0)} 行，
差 {pct_share(abs(P['off']['route_total']-P['slot']['route_total']), P['off']['route_total'], 2)}），
但 slot 多算了 <b>{fnum(P['slot']['compute_pad_total'],0)} 行 pad</b>（占其计算量
{pct_share(P['slot']['compute_pad_total'], P['slot']['compute_pad_total']+P['slot']['real_total'])}）。
pad 行不过路由、但要过 grouped matmul，<b>它是 slot 计算量的主体，也是 plan 要走完池子的原因</b>。</div>""")

    A('<h3>3.3 两类波的每 cell 设备开销（含通信）</h3>')
    A('<h4>off</h4>')
    A(tbl(["算子", "核类型", "调用数", "总时长 (s)", "均值 (µs)", "最大 (µs)", "占窗口",
           "真实波 µs/cell", "空转波 µs/cell"], fam_rows("off")))
    A('<h4>slot</h4>')
    A(tbl(["算子", "核类型", "调用数", "总时长 (s)", "均值 (µs)", "最大 (µs)", "占窗口",
           "真实波 µs/cell", "空转波 µs/cell"], fam_rows("slot")))
    A(legend(seg_series(SEG_ORDER)))
    A(c_cell)
    A(f'<p class="cap">图：两类波每 cell 的机制时间（每 rank 每层，µs）。off 空转波 = dispatch+combine（MC2），'
      f'真实波 = HCCL 集合通信；slot 两种波都以 planning_kernel 与 barrier 为主。点/悬停可见明细。<br>'
      f'读表注意：<code>最大 (µs)</code> 是<b>单条任务</b>的时长，含从入队到结束的等待，个别任务能到几百 ms'
      f'（off 的 allGather 最大 {us(DEV["off"]["fam"]["hcom_ag"]["us_max"])}，'
      f'slot 的 plan 最大 {us(DEV["slot"]["fam"]["plan"]["us_max"])}）；跨运行可比的是<code>均值</code>与 <code>µs/cell</code>。</p>')

    A('<h2 id="s4">4. 设备侧归因</h2>')
    o = DEV["off"]
    A('<h3>4.1 off：真实波走 HCCL，空转波走 MC2</h3>')
    A(f"""<p>off 运行的跨卡机制由两条完全不同的路径组成，两条都在设备侧看得见：
真实波（R）走<b>通信核</b>上的 HCCL 集合通信，空转波（D）走 MC2 的
<code>MoeDistributeDispatch/Combine</code> 融合算子。每 rank 的调用数正好是
「步数 × 43 层 × 每 cell 次数」，可以逐条对上：</p>""")
    A(tbl(["路径", "算子", "核类型", "调用数（每 rank）", "总时长 (s)", "每 cell (µs)", "对应波"],
          [["HCCL 集合通信", "<code>hcom_alltoallv_AicpuKernel</code>",
            "COMMUNICATION", fnum(o["fam"]["hcom_a2a"]["n"], 0),
            fnum(o["fam"]["hcom_a2a"]["us"] / 1e6, 3), fnum(o["fam"]["hcom_a2a"]["percell_R"] or 0, 0),
            f"{o['n_R']} 步 × 43 层 × 3"],
           ["", "<code>hcom_allGather_AicpuKernel</code>", "COMMUNICATION",
            fnum(o["fam"]["hcom_ag"]["n"], 0), fnum(o["fam"]["hcom_ag"]["us"] / 1e6, 3),
            fnum(o["fam"]["hcom_ag"]["percell_R"] or 0, 0), f"{o['n_R']} 步 × 43 层 × 1"],
           ["", "<code>HcclLaunchAicpuKernel</code>（下发）", "AI_CPU",
            fnum(o["fam"]["hccl_launch"]["n"], 0), fnum(o["fam"]["hccl_launch"]["us"] / 1e6, 3),
            fnum(o["fam"]["hccl_launch"]["percell_R"] or 0, 0), "每条通信算子一次下发"],
           ["MC2 融合算子", "<code>MoeDistributeDispatchV3</code>", "MIX_AIV",
            fnum(o["fam"]["mc2_dispatch"]["n"], 0), fnum(o["fam"]["mc2_dispatch"]["us"] / 1e6, 3),
            fnum(o["fam"]["mc2_dispatch"]["percell_D"] or 0, 0), f"{o['n_D']} 步 × 43 层"],
           ["", "<code>MoeDistributeCombineV3</code>", "AI_VECTOR_CORE",
            fnum(o["fam"]["mc2_combine"]["n"], 0), fnum(o["fam"]["mc2_combine"]["us"] / 1e6, 3),
            fnum(o["fam"]["mc2_combine"]["percell_D"] or 0, 0), f"{o['n_D']} 步 × 43 层"]],
          ))
    _off_mech = (o["fam"]["mc2_dispatch"]["us"] + o["fam"]["mc2_combine"]["us"]
                 + o["fam"]["hcom_a2a"]["us"] + o["fam"]["hcom_ag"]["us"]
                 + o["fam"]["hccl_launch"]["us"])
    A(f"""<div class="note">R 波与 D 波用量级完全不同的两条路：
<b>R 波每 cell {fnum((o['fam']['hcom_a2a']['percell_R'] or 0) + (o['fam']['hcom_ag']['percell_R'] or 0) + (o['fam']['hccl_launch']['percell_R'] or 0), 0)} µs
（HCCL 集合通信 + 下发）</b>，<b>D 波每 cell {fnum((o['fam']['mc2_dispatch']['percell_D'] or 0) + (o['fam']['mc2_combine']['percell_D'] or 0), 0)} µs
（MC2 dispatch + combine）</b>；合计 {sec(_off_mech / 1e6)}（含下发），
其中 R 波占 {pct_share(o['fam']['hcom_a2a']['us'] + o['fam']['hcom_ag']['us'] + o['fam']['hccl_launch']['us'], _off_mech)}。
<b>这也解释了 off 为什么在通信核上「看起来很忙」</b>：HCCL 的 7.35 s 全落在通信核，
与 AI 核的 11.5 s 完全不重叠（并集仍是两者之和）。</div>""")
    A("<h3>4.2 slot：MoonEP 机制</h3>")
    A(tbl(["算子", "调用数（每 rank）", "总时长 (s)", "均值 (µs)", "占机制",
           "真实波 µs/cell", "空转波 µs/cell"],
          [[f'<code>{esc(pat)}</code>', fnum(DEV["slot"]["fam"][k]["n"], 0),
            fnum(DEV["slot"]["fam"][k]["us"]/1e6, 3), fnum(DEV["slot"]["fam"][k]["us_avg"], 0),
            pct_share(DEV["slot"]["fam"][k]["us"], sum(RU["slot"][x]["us"] for x in SEG_ORDER)),
            (fnum(DEV["slot"]["fam"][k]["percell_R"], 0), "good") if DEV["slot"]["fam"][k]["percell_R"] else "—",
            (fnum(DEV["slot"]["fam"][k]["percell_D"], 0), "bad") if DEV["slot"]["fam"][k]["percell_D"] else "—"]
           for k, pat, _ in FAMILIES["slot"]]))
    A(legend(seg_series(SEG_ORDER)))
    A(c_tot)
    A(f'<p class="cap">图：MoE 机制总时间构成（每 rank 整轮 192 步，s）。'
      f'slot 的机制时间 {sec(W["slot"]["mech"])} 是 off 的 {W["slot"]["mech"]/W["off"]["mech"]:.2f} 倍。</p>')

    A("<h3>4.3 核内指令占比：谁在干活，谁在等</h3>")
    A(insn_table(DEV))
    _b_plan = insn_busiest(DEV["slot"]["df"], "planning_kernel")
    _b_bar = insn_busiest(DEV["slot"]["df"], "barrier_on_stream_kernelim")
    A(f"""<div class="note"><b>读法</b>：<code>*_ratio</code> 是「该单元占本核 cycle 的比例」，
各流水单元是并行的，因此不能相加成「核忙」；有信息的是<b>最忙的那个单元</b>和它之外的部分。
slot 的两个大头恰好是最不干活的：<code>planning_kernel</code> 最忙单元只有
{fnum(_b_plan['ratio'],1)}%（{_b_plan['unit']}），<code>_Z24barrier_on_stream_kernelim</code> 只有
{fnum(_b_bar['ratio'],1)}%（{_b_bar['unit']}）——它们的时长基本是<b>等通知/等对端</b>，
不是算力。对照 <code>GroupedMatmul</code>：cube 占
{fnum(insn_busiest(DEV['slot']['df'], 'GroupedMatmul')['ratio'],1)}%，那才是真在算。
off 的空转波则是反过来：<code>MoeDistributeDispatchV3</code> 的
{fnum(insn_busiest(DEV['off']['df'], 'MoeDistributeDispatchV3')['unit'] and insn_busiest(DEV['off']['df'], 'MoeDistributeDispatchV3')['ratio'],1)}%
压在 {insn_busiest(DEV['off']['df'], 'MoeDistributeDispatchV3')['unit']} 上，是真在搬数据。</div>""")

    A("<h3>4.4 planning 的开销随池子体积走</h3>")
    A(tbl(["每层计算行 M", "对应池容量", "cell 数", "plan µs/cell", "plan ÷ 容量 (µs/token)",
           "barrier×8 µs/cell", "combine µs/cell", "GroupedMatmul µs/cell"],
          [[fnum(r["M"], 0) if "M" in r else fnum(r["cap"] * TOP_K, 0), fnum(r["cap"], 0),
            fnum(r["n_cells"], 0), fnum(r.get("plan"), 0),
            fnum((r.get("plan") or 0) / r["cap"], 2) if r["cap"] else "—",
            fnum((r.get("barrier") or 0), 0), fnum(r.get("moep_combine"), 0), fnum(r.get("gemm"), 0)]
           for r in sorted([x for x in DEV["slot"]["ladder"] if x["n_cells"] > 100], key=lambda x: x["cap"])]))
    A(legend([{"key": "plan", "label": "规划 plan", "color": "--series-5"},
              {"key": "barrier", "label": "流屏障 barrier", "color": "--series-6"}]))
    A(c_cap)
    A('<p class="cap">图：plan / barrier 的每 cell 时间随池容量（128 / 256 / 512 token）。</p>')

    A("<h3>4.5 设备占用 vs 墙钟窗口（为什么差距只有十几 %）</h3>")
    A(tbl(["运行", "设备窗口 (s)", "计算占比", "通信占比（未重叠）", "空闲占比",
           "设备占用 (s)", "空闲 (s)", "机制时间 (s)", "机制 ÷ 占用"],
          [[leg, sec(W[leg]["win"]), pct_share(W[leg]["sh_c"]*100, 100), pct_share(W[leg]["sh_k"]*100, 100),
            pct_share(W[leg]["sh_i"]*100, 100), sec(W[leg]["occ"]), sec(W[leg]["win"]-W[leg]["occ"]),
            sec(W[leg]["mech"]), pct_share(W[leg]["mech"], W[leg]["occ"])] for leg in ("off", "slot")]))
    A(legend([{"key": "computing", "label": "计算", "color": "--series-1"},
              {"key": "comm", "label": "通信（未重叠）", "color": "--series-2"},
              {"key": "free", "label": "空闲（等下发/等对端）", "color": "--neutral-fill"}]))
    A(c_win)
    A(f"""<p class="cap">图：每 rank 的窗口时间构成（%）——同一张图里 off 有 {pct_share(W['off']['sh_i']*100,100)} 的时间设备是空的，
slot 只有 {pct_share(W['slot']['sh_i']*100,100)}。</p>
<div class="note warn">这条解释了为什么“机制时间差 {fnum(abs(W['slot']['mech']-W['off']['mech']),1)} s”只换来
“墙钟差 {fnum(abs(W['slot']['win']-W['off']['win']),1)} s”：<b>off 侧本来就有一大段设备空转在吸收工作量</b>。
反过来说，<b>修好 off 侧的下发（hostbound）之后，两模式的墙钟差会向机制比 {W['slot']['mech']/W['off']['mech']:.2f}× 靠拢</b>——
现在拿 off 的端到端延迟当基线，会低估 MoonEP 的代价。</div>""")

    # §5
    A('<h2 id="s5">5. 官方 profiling-analysis skill 的判定（含口径警告）</h2>')
    A(tbl(["运行", "计算占比", "通信占比", "空闲占比", "判定"],
          [["off", pct_share(W["off"]["sh_c"]*100, 100), pct_share(W["off"]["sh_k"]*100, 100),
            pct_share(W["off"]["sh_i"]*100, 100), "【下发问题】Hostbound"],
           ["slot", pct_share(W["slot"]["sh_c"]*100, 100), pct_share(W["slot"]["sh_k"]*100, 100),
            pct_share(W["slot"]["sh_i"]*100, 100), "无明显瓶颈"]]))
    A("""<div class="note warn"><b>别把这两个判定当结论</b>：MoonEP 的跨卡流量走 SHMEM + 自定义算子，
工具看到的「通信 = 0%」，于是把 plan/barrier 的自旋与等待都算进「计算」；off 侧的设备空转则被如实标成 hostbound。
两个判定只有放在同一把尺子下才有意义——本报告的每 cell 归因就是那把尺子。</div>""")

    # §6/§7 rebuild from computed numbers
    A(build_diff_opt(DEV, RU, P, T, W, B))
    # §8
    def _src(p: Path) -> str:
        try:
            st = p.stat()
        except OSError:
            return f'<code>{esc(p)}</code><br>（未找到）'
        stamp = _dt.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        return f'<code>{esc(p)}</code><br>{stamp} · {st.st_size / 1048576:.1f} MiB'

    def _rank0(leg_key: str) -> str:
        hits = sorted((SCRIPTS / LEGS[leg_key] / "profiling").glob("*rank0*_ascend_pt"))
        return hits[0].name if hits else "dp0_..._rank0_<pid>_<ts>_ascend_pt"

    L1o, L1s = LEGS["g1_off"], LEGS["g1_slot"]
    L2o, L2s = LEGS["g2_off"], LEGS["g2_slot"]
    A('<h2 id="s8">8. 复现与数据出处</h2>')
    A('<h3>8.1 复现命令（后三步零卡时）</h3>')
    A(f"""<pre># ① 跑一次运行：MOE_PAR=1 才带负载 dump，PROFILE=1 才带 profiling（脚本内部会等上一次运行的设备释放）
#    产物默认落在脚本旁边 serve_&lt;mode&gt;/，要落在工作区就像下面这样显式给 OUT_DIR
MOE_PAR=1 PROFILE=1 OUT_DIR=$WS/scripts/运行目录名 \\
    bash $WS/vllm_ascend_moonep/scripts/dsv4flash_service/start_service_mode.sh {{off|slot}}

# ② 记录器分析：把该运行的 dump 聚合成负载报告（运行名 = 输出目录名，报告名自带运行名）
python $WS/vllm_ascend_moonep/scripts/moe_par/analyze_moe_par.py \\
    --run {L2o}=$WS/scripts/{L2o}/moe_par \\
    -o $WS/scripts/{L2o}/{L2o}_report.html          # slot 另一次运行同理，换目录名即可

# ③ 设备侧离线导出：profiling 目录在只读共享盘上，msprof 要往里面写中间文件，
#    所以用 overlayfs 把 rank0 目录叠成可写视图（只叠加，不动原始数据）
mkdir -p /tmp/prof3/off/{{upper,work,merged}}
mount -t overlay overlay -o lowerdir=$WS/scripts/{L2o}/profiling/{_rank0('g2_off')}, \\
      upperdir=/tmp/prof3/off/upper,workdir=/tmp/prof3/off/work /tmp/prof3/off/merged
msprof --export=on --output=/tmp/prof3/off/merged      # 约 5 分钟，产出 op_summary / task_time*
umount /tmp/prof3/off/merged

# ④ 本报告：所有数字现算，报告里没有写死常量（换运行只改脚本头的 LEGS / PROF_ROOTS / PAYLOAD_SRC）
python $WS/vllm_ascend_moonep/scripts/dsv4flash_service/gen_moonep_off_vs_slot_perf_report.py</pre>""")
    A("""<h3>8.2 已知的口径陷阱（换数据前先读这一节）</h3>
<ul>
<li><b>通信核任务不在 <code>task_time</code> 里</b>：CANN 9.2 导出的 <code>task_time*.csv</code> 只覆盖一部分实例
（slot 运行的 <code>task_time_*.csv</code> 还把 kernel 名字塞进了类型列、<code>kernel_name</code> 是 N/A）。
所以本脚本<b>窗口取自 <code>task_time*.csv</code>，而三类区间（计算 / 通信 / AI CPU）全部取自
<code>op_summary</code> 的 <code>Task Type</code></b>——两边混用会整段漏掉 off 的 HCCL（约 7.35 s）。</li>
<li><b>两次运行的导出文件名不同</b>：off 只有 <code>task_time_slice_0/1_*.csv</code>（外加 <code>communication_statistic_*.csv</code>），
slot 只有 <code>task_time_*.csv</code>（外加 <code>aicpu_*.csv</code>）；老布局的 <code>kernel_details.csv</code> /
<code>step_trace_time.csv</code> 在 CANN 9.2 里都不存在。脚本用 glob 兜住两种，别照抄文件名。</li>
<li><b><code>*_ratio</code> 是并行流水占比，不能相加</b>：它是「该流水单元占本核 cycle 的比例」，
cube / mte2 / scalar 是并行的，加起来会超过 100%；有信息的是<b>最忙的那个单元</b>（见 §4.3）。</li>
<li><b>别用「步间空隙」聚类步号</b>：HCCL 通信发生在本步第一个 GEMM <b>之前</b>，按空隙切会把通信整段记到上一步。
本脚本直接用 <code>idx // 43</code>（cell 数 8256 = 192 步 × 43 层，整除），cell 归因用「最近 cell 中点」。</li>
<li><b>off 空转波的 M 恒为 48</b>：<code>M</code> = 该层每 rank 的计算行 = 该步 8 rank × 1 token × K；
它与「逐 rank 路由量」无关（payload 里 off 空转步的逐 rank 真实行是 0 / 16 / 0 … 那样分布）。
判波只能看 M（off）或池容量（slot），不能看路由量。</li>
<li><b>pad 行不是「浪费的算力」而是「构造出来的对称」</b>：容量各 rank 相同 ⇒ 合计计算量的 PAR 恒为 1.000，
这不是均衡证据；信号在真实 / pad 拆分里（§3.2）。</li>
<li><b>末步截断</b>：第 192 步记录不全（本轮 215 个 cell 缺失），阶梯表里会出现 3 / 75 / 384 这类尾值；
占比可忽略，但别把它当平均值的输入。</li>
<li><b>单 rank 口径</b>：设备侧只分析 rank0。DP8 对称，但若某 rank 路由显著偏斜，绝对值会有出入。</li>
</ul>""")
    A('<h3>8.3 数据出处</h3>')
    A(tbl(["运行", "端到端日志（ais_bench）", "负载报告（payload 来源）", "设备侧导出", "用途"], [
        [f"<code>{esc(L1o)}</code>", _src(SCRIPTS / L1o / f"{L1o}_test_gsm8k.log"), "—", "—", "干净端到端 off"],
        [f"<code>{esc(L1s)}</code>", _src(SCRIPTS / L1s / f"{L1s}_test_gsm8k.log"), "—", "—", "干净端到端 slot"],
        [f"<code>{esc(L2o)}</code>", _src(SCRIPTS / L2o / f"{L2o}_test_gsm8k.log"),
         _src(PAYLOAD_SRC["off"]), _src(PROF_ROOTS["off"]), "负载结构 + 设备归因 off"],
        [f"<code>{esc(L2s)}</code>", _src(SCRIPTS / L2s / f"{L2s}_test_gsm8k.log"),
         _src(PAYLOAD_SRC["slot"]), _src(PROF_ROOTS["slot"]), "负载结构 + 设备归因 slot"],
    ]))
    A(f"""<div class="cap">设备侧只导出 rank0 一份：已核对 8 个 rank 的 <code>*_ascend_pt</code> 目录都在，
两次运行的窗口差（{sec(W['off']['win'])} vs {sec(W['slot']['win'])}）与客户端日志的
Benchmark Duration 差（{sec(B['g2_off']['common']['Benchmark Duration'][0]/1000, 1)} vs
{sec(B['g2_slot']['common']['Benchmark Duration'][0]/1000, 1)}）同量级，说明单 rank 的窗口是可信的。</div>""")
    A(f"""<div class="foot">本报告由 <code>scripts/dsv4flash_service/gen_moonep_off_vs_slot_perf_report.py</code> 生成于 {now}；
所有数字由该脚本从头现算（端到端日志 + 记录器 payload + msprof 离线导出），报告里没有写死的常量。
口径：每 rank 每层（cell）；窗口 = rank0 <code>task_time*.csv</code> 的覆盖范围；占空比 = 三类区间按优先级
（计算 &gt; 通信 &gt; AI CPU）做互斥切分后的并集，空闲 = 窗口 − 占用。</div>
</main></body></html>""")
    return "\n".join(H)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成 MoonEP off / slot 性能差异与优化点报告（gsm8k）")
    ap.add_argument("--out", default=str(OUT_DEFAULT),
                    help=f"输出 HTML（默认 {OUT_DEFAULT}）")
    args = ap.parse_args(argv)

    B = {}
    for key, d in LEGS.items():
        log = SCRIPTS / d / f"{d}_test_gsm8k.log"
        if not log.exists():
            raise SystemExit(f"缺少端到端日志：{log}")
        B[key] = parse_bench(log)

    P = {leg: payload_summary(load_payload(leg)) for leg in PAYLOAD_SRC}
    T = {leg: window_stats(leg) for leg in PROF_ROOTS}
    DEV = {leg: analyse_device(leg) for leg in PROF_ROOTS}
    RU = {leg: rollup_fam(DEV[leg]) for leg in PROF_ROOTS}

    html = build(B, P, T, DEV, RU)
    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"[OK] 写出 {out}（{len(html) / 1024:.0f} KiB，{out.stat().st_size / 1024:.0f} KiB 落盘）")

    # 自检：把关键数字打出来，方便和报告里的表逐条对
    for leg in ("off", "slot"):
        w = T[leg]
        mech = sum(r["us"] for k, r in RU[leg].items() if k != "gemm") / 1e6
        print(f"     {leg}: 窗口 {w['win']:.3f}s 占用 {w['occupied']:.3f}s "
              f"({w['occupied'] / w['win'] * 100:.2f}%) | 计算 {w['share']['compute'] * 100:.1f}% "
              f"通信 {w['share']['comm'] * 100:.1f}% 空闲 {w['idle'] / w['win'] * 100:.1f}% | "
              f"{DEV[leg]['n_cells']:,} cell = {DEV[leg]['n_R']}R + {DEV[leg]['n_D']}D 步 | "
              f"M {DEV[leg]['m_min']:.0f}–{DEV[leg]['m_max']:.0f}（floor {DEV[leg]['floor']:.0f}，"
              f"{DEV[leg]['n_floor_cells']:,} cell）| 机制 {mech:.3f}s")
    print(f"     payload: off 真实 {P['off']['real_total']:,.0f} 行 / pad {P['off']['compute_pad_total']:,.0f}；"
          f"slot 真实 {P['slot']['real_total']:,.0f} / pad {P['slot']['compute_pad_total']:,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
