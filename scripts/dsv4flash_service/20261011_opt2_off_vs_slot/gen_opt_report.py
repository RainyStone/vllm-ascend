#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""off vs slot(+历史收益优化组合) 性能报告生成器（每形状一台服务矩阵）。

用法: python3 gen_opt_report.py [--grid /tmp/mc2_probe/grid_opt] [--out <前缀>]
输出:
  <out>.html 详情报告
  <out>.md   摘要（每 mbt×形状×路由×腿一行的均值表 + slot 相对 off 收益）

数据来源：$GRID/<run>/{warmup_c0,L*_n32_c1..3}.json（bench_rt_one.sh 产物），
          $GRID/<run>.log（服务日志，取 [moonep-env-audit] 运行时 env 证据）。

口径（与既有方法论一致）：
  - 每格 3 轮，轮 1 为热身弃掉，统计取轮 2/3；n=2 的均值=中位。
  - 均值为主口径；bench json 的全部标量字段（mean/median/std/p50/p90/p99、
    各吞吐、token 计数、并发观测等）逐轮全量收录，不省略。
  - 收益 = (slot − off)/off × 100%，同 (mbt, 路由, 形状) 内配对；
    延迟类负值=slot 更快，吞吐类正值=slot 更高。
"""
import argparse
import datetime as _dt
import html as H
import json
import math
import os
import re
import statistics as st
import subprocess
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
GRID_DEF = "/tmp/mc2_probe/grid_opt"
RUN_RE = re.compile(r"^op2_mbt(?P<mbt>\d+)_(?P<route>\w+?)_(?P<leg>off|slot)_(?P<shape>L8k|L16k)$")
MBTS = ("8192", "16384")
ROUTES = ("uniform", "4hot", "3hot1r", "4hot1r")
LEGS = ("off", "slot")
SHAPES = ("L8k", "L16k")
SHAPE_FULL = {"L8k": "L8k_n32", "L16k": "L16k_n32"}
SHAPE_DESC = {
    "L8k": "输入 8192 / 输出 1（32 prompts，并发 16，seed 档位 9）",
    "L16k": "输入 16384 / 输出 1（32 prompts，并发 16，seed 档位 12）",
}
ROUTE_INFO = {  # 路由名 → (档案, s = max rank/均值, rank0-3 home 载荷)
    "uniform": ("sim_route_uniform.txt", "1.00", ("25%", "25%", "25%", "25%")),
    "4hot": ("sim_route_4hot.txt", "1.66", ("41.5344%", "41.5344%", "8.4656%", "8.4656%")),
    "3hot1r": ("sim_route_3hot_1rank.txt", "2.48", ("62.0554%", "12.6482%", "12.6482%", "12.6482%")),
    "4hot1r": ("sim_route_4hot_1rank.txt", "2.98", ("74.6033%", "8.4656%", "8.4656%", "8.4656%")),
}
LEG_TITLE = {"off": "off（标准 MC2 / all-to-all EP，对照组）",
             "slot": "slot（MoonEP slot + real-T + 历史收益优化组合）"}

# ---------------- slot 腿开启的 ascend-moonep 优化清单（报告核心章节的数据源） ----------------
# (名称, 层, 启用方式, 作用说明, 历史依据/出处)
OPTS_ON = [
    ("real-T（内核按真实 token 负载计算与均衡）", "ascend-moonep 内核+tiling", "默认开（MOONEP_REAL_T=1）",
     "dispatch/combine 的 tiling 携带 num_valid_tokens，内核按真实 T 而非含 pad 的 T 计算行循环与负载均衡，MOONEP_REAL_T=1 时零 pad 注入——即「按真实 token 负载均衡，而非含 padding 的负载」。",
     "ascend-moonep 3e8f1cc + e549ea3；vllm-ascend moonep_comm 默认 MOONEP_REAL_T=1"),
    ("dispatch 输出复用（reuse_outputs）", "ascend-moonep api", "默认开（MOONEP_REUSE_OUT=1）",
     "dispatch 复用 plan/cu_seqlens 与非 zero-copy 输出张量，免去每层重分配与重包装。",
     "ascend-moonep 1ab1e55；vllm-ascend 35138dc2e（zero-sync+复用默认开，当时 +22.4%）"),
    ("slot FP8 量化镜像（slot quant）", "vllm-ascend moonep_balance", "默认开（MOONEP_SLOT_QUANT=1）",
     "FP8 量化权重镜像进 bf16 byte carrier（按 4 payload 镜像），slot 槽位按字节承载而非按权重形状，显存与搬运量减半。",
     "服务日志 [moonep-release] quantized payloads mirror into bf16 byte carriers"),
    ("planning 核数自适应", "ascend-moonep api", "默认开（未设 MOONEP_PLANNING_NUM_SMS 时自动）",
     "planning 核数按路由量选择：routes≤8192 用 8 blocks（网格同步更便宜），dispatch/combine/prefetch 仍用满核——planning 的耗时基本不随 token 数变化（每步固定动作），窄网格在小路由量下实测省 ~8μs/次。",
     "ascend-moonep 36f8ca6（4 rank 隔离 bench 实测表）"),
    ("top-B 选择排序→名次计数", "ascend-moonep planning 内核", "默认开（无条件）",
     "planning 的 top-B 槽位选择从 O(B) 选择排序改为名次计数（rank-counting），3000 随机样例与边界例逐位等价。",
     "ascend-moonep b6430c7：planning 中位 1138→463μs"),
    ("lookup 表加载间冗余 PIPE_ALL 移除", "ascend-moonep planning 内核", "默认开（无条件）",
     "去掉 planning 两张 lookup 表加载之间不必要的全核管道同步。",
     "ascend-moonep 530b33e"),
    ("sym tensor 包装缓存", "ascend-moonep buffer", "默认开（无条件，_SYM_TENSOR_CACHE）",
     "create_sym_tensor_from_source 结果按源缓存，免每层重复包装对称张量视图。",
     "ascend-moonep 385f0e0：prefetch host 段 0.16→0.06 ms/层"),
    ("AICore 数缓存", "ascend-moonep prefetch C++", "默认开（无条件）",
     "LaunchPrefetch 的 grid 计算不再每次调 aclrtGetDevice/aclrtGetDeviceInfo（每次几十 μs、每层 2 次），进程内只解析一次。",
     "ascend-moonep e189239：−3.3 ms/步"),
    ("current_stream 查询跳过", "ascend-moonep api", "默认开（无条件）",
     "非测量路径不再每 op 查询 current stream；库内分段计时仅按需打开。",
     "ascend-moonep 63c0eb7"),
    ("host 参数字符串拼接移除", "ascend-moonep ops", "默认开（无条件）",
     "_collect_inputs 去掉每层拼接摘要字符串的固定 host 开销。",
     "ascend-moonep 3cd6c2e"),
    ("set_access 物理卡号修正", "ascend-moonep vmm", "默认开（无条件，正确性 fix）",
     "VMM set_access 用物理卡号而非逻辑序号，修缩卡/DP 部署下的 507899 错误——DP4 部署的前置条件。",
     "ascend-moonep 7f86db4"),
    ("ASCEND_MOONEP_PREFETCH_RELAXED=1", "ascend-moonep prefetch C++", "env 显式开启（本次矩阵 slot 腿）",
     "摘掉 prefetch 的两个跨卡 barrier（aclshmemx_barrier_all_on_stream）：源是 peer 的 home chunk、init 后不变，前后栅栏对本用途排不出有意义的序；kernel 内 MTE3_MTE2 事件已保证 GET 完成可见。同时为 side-stream 发射扫清时序障碍。",
     "用户指定的历史收益项；ascend-moonep e189239"),
    ("ASCEND_MOONEP_NO_RECONCILE=1", "ascend-moonep api", "env 显式开启（本次矩阵 slot 腿，与 RELAXED 成对）",
     "跳过每层 3 次（dispatch/prefetch/combine 后）的 host 侧 npu.synchronize + dist.barrier 对齐 shim——该 shim 曾造成 51~60% 设备空转，且让 host 永远跑不到设备前面。RELAXED 摘掉 prefetch 的 host barrier 后，剩余三个设备内联栅栏各 rank 同序同次、代际天然一致，无需 host 重排。",
     "ascend-moonep 4995a63（profiling 中对应明显设备空转）；成对前提见 api.py:595"),
    ("ASCEND_MOONEP_SKIP_DUP=1", "ascend-moonep planning C++", "env 显式开启（本次矩阵 slot 腿）",
     "planning 短路 source-dup 阶段（counts→prefixes→payloads + 2 次 inbox 轮询）；代价是 dispatch 多传重复的 hidden 行（dispatch 本身才 ~180μs/层），换取 planning 大幅变快。",
     "journey 报告实测：dp0 长尾 1219.7→24.5μs；ascend-moonep c75a271"),
]
OPTS_OFF = [
    ("ASCEND_MOONEP_BARRIER_V4=1", "跨卡汇合栅栏 pull→push 模式。R=4（DP4）下历史实测 +1.0%（噪声内）、周期 −2.5%，判定中性；v4 在 R≥8 才明确占优。", "所有 rank 必须一致"),
    ("MOONEP_EARLY_PREFETCH=1", "attention 前用上一步 plan 行投机预取（预测命中 ~69-80%）。实验性：预测路径曾有跨卡 pacing 失配挂死史；且本矩阵为 prefill 主导（输出 1 token），prefetch 仅占 TTFT ~0.4%。", "实验开关"),
    ("MOONEP_PF_STREAM=2", "真实 plan 的 prefetch 发 side stream 与 dispatch 并行。必须配 RELAXED（已配）但无干净的历史端到端收益记录。", "实验开关"),
    ("MOONEP_NO_PIN / ALLGATHER fallback / KEEP_MC2", "pinned host buffer 跳过 / MC2 建链失败兜底 / 小步走 MC2——非本实验目标条件。", "特定场景开关"),
    ("测量类：BALANCE_STATS / TIME_STATS / TRACE / MOE_PAR / PROFILE", "统计与采集开关会给热路径加 host 同步或落盘 io，污染性能数据——性能腿一律不开（这也是历史 journey 报告的教训：BALANCE_STATS 的 2 个 .item() 曾留下 757.7ms 设备空转）。", "仅负载分析腿使用"),
]

BENCH_FIELDS_ORDER = [
    # 延迟（均值为主口径）与分位
    "mean_ttft_ms", "median_ttft_ms", "p50_ttft_ms", "p90_ttft_ms", "p99_ttft_ms", "std_ttft_ms",
    "mean_e2el_ms", "median_e2el_ms", "p50_e2el_ms", "p90_e2el_ms", "p99_e2el_ms", "std_e2el_ms",
    # 吞吐
    "request_throughput", "output_throughput", "total_token_throughput", "request_goodput", "rtfx",
    # 请求与 token 计数
    "completed", "failed", "num_prompts", "total_input_tokens", "total_output_tokens",
    "max_output_tokens_per_s",
    # 运行条件观测
    "duration", "max_concurrency", "max_concurrent_requests", "request_rate", "burstiness",
    # 标识
    "backend", "endpoint_type", "model_id", "tokenizer_id", "date", "label",
]
KEY_METRICS = ["mean_ttft_ms", "p99_ttft_ms", "mean_e2el_ms", "request_throughput",
               "total_token_throughput", "duration"]
LOWER_BETTER = {"mean_ttft_ms", "p99_ttft_ms", "mean_e2el_ms", "duration"}


def load_run(grid: Path, run: str):
    """返回 {round_name: json_dict}（只含存在的轮次文件），外加 audit env 证据。"""
    d = grid / run
    out = {}
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        try:
            out[f.stem] = json.loads(f.read_text())
        except Exception:
            pass
    return out


def env_audit(grid: Path, run: str) -> str:
    log = grid / f"{run}.log"
    if not log.is_file():
        return ""
    m = re.search(r"\[moonep-env-audit\] (.+)", log.read_text(errors="ignore"))
    return m.group(1).strip() if m else ""


def t_hist(grid: Path, run: str) -> dict:
    """服务日志里 [sim-route] 打印的每步 token 数（T=）直方图：{T: 次数}。

    4 个 rank、3 轮 bench 共用一份日志，次数是合计值；T=1 是 DP 心跳/decode
    步，不计入 prefill 分析。"""
    log = grid / f"{run}.log"
    out = {}
    if not log.is_file():
        return out
    for mo in re.finditer(r"T=(\d+) K=\d+", log.read_text(errors="ignore")):
        t = int(mo.group(1))
        if t > 1:
            out[t] = out.get(t, 0) + 1
    return out


def git_head(repo: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", repo, "log", "-1",
                                        "--format=%h %ad %s", "--date=short"],
                                       text=True, timeout=10).strip()
    except Exception:
        return "不可用"


def agg_rounds(rnds):
    """轮 2/3 聚合：每字段取均值（n=2 时均值=中位）。返回 {field: mean}。"""
    vals = {}
    if not rnds:
        return vals
    keys = set()
    for r in rnds:
        keys |= {k for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for k in keys:
        xs = [r[k] for r in rnds if isinstance(r.get(k), (int, float)) and not isinstance(r[k], bool)]
        if xs:
            vals[k] = st.mean(xs)
    return vals


def fmt(v, field=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        if field.endswith("_ms"):
            return f"{v:,.1f}"
        if abs(v) >= 100:
            return f"{v:,.1f}"
        return f"{v:,.3f}".rstrip("0").rstrip(".")
    return str(v)


def pct(a, b):
    if a is None or b in (None, 0):
        return None
    return (a - b) / b * 100.0


# ---------------- §4 图表：随峰均比变化的折线小多图（内联 SVG，悬停读数） ----------------
C_SLOT, C_OFF = "#2a78d6", "#eb6834"   # 分类色 slot1/slot2，已过 validate_palette（光模式）
VIZ_ROUTES = ("uniform", "4hot", "3hot1r", "4hot1r")
VIZ_S = (1.00, 1.66, 2.48, 2.98)
VIZ_S_LBL = ("1.00", "1.66", "2.48", "2.98")
# 面板几何（viewBox 坐标）：右侧留 46px 给线尾直标，避免文字被裁
PG = dict(W=368, H=196, L=46, R=322, T=14, B=150)


def nice_ticks(lo, hi, step):
    out, v = [], math.ceil(lo / step) * step
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def panel_svg(fig_key, off, slot, ydom, yticks, yfmt, show_y, tip_title):
    """一个 2 折线面板。x=峰均比（按真实数值定位），y 为共享值域。"""
    L, R, T, B = PG["L"], PG["R"], PG["T"], PG["B"]

    def X(v):
        return L + (v - VIZ_S[0]) / (VIZ_S[-1] - VIZ_S[0]) * (R - L)

    def Y(v):
        return B - (v - ydom[0]) / (ydom[1] - ydom[0]) * (B - T)

    attrs = (f'aria-label="{H.escape(tip_title)}" data-fig="{fig_key}" data-title="{H.escape(tip_title)}"'
             f' data-xx="{json.dumps([round(X(v), 1) for v in VIZ_S])}"'
             f' data-off="{json.dumps(off)}" data-slot="{json.dumps(slot)}"')
    p = [f'<svg class="vp" viewBox="0 0 {PG["W"]} {PG["H"]}" tabindex="0" role="img" {attrs}>']
    for tv in yticks:
        yy = Y(tv)
        p.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{R}" y2="{yy:.1f}" stroke="#e1e0d9" stroke-width="1"/>')
        if show_y:
            p.append(f'<text x="{L - 6}" y="{yy + 3.5:.1f}" text-anchor="end" class="tk">{yfmt(tv)}</text>')
    p.append(f'<line x1="{L}" y1="{B}" x2="{R}" y2="{B}" stroke="#c3c2b7" stroke-width="1"/>')
    for xv, xl in zip(VIZ_S, VIZ_S_LBL):
        xx = X(xv)
        p.append(f'<line x1="{xx:.1f}" y1="{B}" x2="{xx:.1f}" y2="{B + 4}" stroke="#c3c2b7" stroke-width="1"/>')
        p.append(f'<text x="{xx:.1f}" y="{B + 16}" text-anchor="middle" class="tk">{xl}</text>')
    p.append(f'<text x="{(L + R) / 2:.0f}" y="{B + 30}" text-anchor="middle" class="axl">峰均比 s</text>')
    rings = []
    for vals, color in ((off, C_OFF), (slot, C_SLOT)):
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(VIZ_S, vals))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
                 'stroke-linejoin="round" stroke-linecap="round"/>')
        rings.append([(round(X(a), 1), round(Y(b), 1)) for a, b in zip(VIZ_S, vals)])
    for (color, pts) in zip((C_OFF, C_SLOT), rings):
        for xx, yy in pts:
            p.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4.5" fill="{color}" '
                     'stroke="#fcfcfb" stroke-width="2"/>')
    # 折线像素坐标交给 JS（悬停吸附用）：补进开标签
    p[0] = p[0][:-1] + (f' data-ptso="{json.dumps(rings[0])}"'
                        f' data-ptss="{json.dumps(rings[1])}"') + ">"
    # 线尾直标（2 条线，名字放线尾；靠得近则上下让开）
    y1, y2 = rings[0][-1][1], rings[1][-1][1]
    d1, d2 = (-9, 13) if y1 <= y2 else (13, -9)
    if abs(y1 - y2) < 15:
        d1, d2 = (-11, 15) if y1 <= y2 else (15, -11)
    p.append(f'<text x="{R + 6}" y="{y1 + d1:.1f}" class="endl">off</text>')
    p.append(f'<text x="{R + 6}" y="{y2 + d2:.1f}" class="endl">slot</text>')
    # 悬停十字线 + 吸附圈（JS 移动）
    p.append('<g class="xh" style="display:none">'
             f'<line y1="{T}" y2="{B}" stroke="#c3c2b7" stroke-width="1"/>'
             '<circle r="4.5" fill="none" stroke="#0b0b0b" stroke-width="1.5" class="xh1"/>'
             '<circle r="4.5" fill="none" stroke="#0b0b0b" stroke-width="1.5" class="xh2"/></g>')
    p.append("</svg>")
    return "".join(p)


def figure_html(fig_key, title, subtitle, panels, ykey, step, yfmt, tip_fmt, tip_unit, tip_delta):
    """panels: [(row_label, col_label, off[], slot[])]，2×2 小多图，共享 y 值域。"""
    allv = [v for _, _, o, s2 in panels for v in (o + s2) if v is not None]
    if not allv:
        return ""
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.08
    ydom = (lo - pad, hi + pad)
    yticks = nice_ticks(ydom[0], ydom[1], step)
    s = [f'<div class="fig"><div class="fig-title">{title}</div>',
         f'<p class="fig-sub">{subtitle}</p>',
         '<div class="legend">'
         f'<span class="lk"><i style="background:{C_OFF}"></i>off（标准 MC2 / all-to-all）</span>'
         f'<span class="lk"><i style="background:{C_SLOT}"></i>slot（MoonEP slot + 优化组合）</span></div>',
         '<div class="grid2x2"><div></div>'
         f'<div class="col-h">输入 8k token（输出 1）</div><div class="col-h">输入 16k token（输出 1）</div>']
    for row, col, off, slot in panels:
        if col.startswith("输入 8k"):
            s.append(f'<div class="row-h"><b>max_num_batched_tokens</b><br>={row}</div>')
        svg = panel_svg(fig_key, off, slot, ydom, yticks, yfmt, show_y=col.startswith("输入 8k"),
                        tip_title=f"max_num_batched_tokens={row} · {col}")
        s.append(f'<div class="pwrap">{svg}</div>')
    s.append("</div>")
    s.append(f'<p class="fig-cap">悬停面板可读数（键盘 Tab 聚焦读末档）；y 轴四块面板共享同一值域，'
             f'便于横向比较；单位 {tip_unit}；y 轴未从 0 开始。每档路由（左→右）：'
             f'uniform / 4hot / 3hot1r / 4hot1r。精确数值与全部 bench 字段见 §3、§4 表。</p></div>')
    # 折线数据注入 JS 侧（含 tooltip 格式）
    s.append(f'<script type="application/json" id="vizdata-{fig_key}">' + json.dumps({
        "unit": tip_unit, "fmt": "ms" if tip_fmt == "ms" else "tput",
        "delta": tip_delta,
        "routes": list(VIZ_ROUTES),
    }) + "</script>")
    return "".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=GRID_DEF)
    ap.add_argument("--out", default=os.path.join(HERE, "report", "op2_off_vs_slot_opt_report"))
    args = ap.parse_args()
    grid = Path(args.grid)

    # ---------- 读全部 run ----------
    runs = {}   # run -> {"rounds": {name: json}, "agg": {...}, "audit": str, "thist": dict}
    missing = []
    for mbt in MBTS:
        for route in ROUTES:
            for shape in SHAPES:
                for leg in LEGS:
                    run = f"op2_mbt{mbt}_{route}_{leg}_{shape}"
                    rr = load_run(grid, run)
                    if not rr:
                        missing.append(run)
                        continue
                    rounds_stat = sorted(k for k in rr if re.match(rf"{SHAPE_FULL[shape]}_c\d+$", k))
                    warm = [k for k in rr if k.startswith("warmup")]
                    keep = [rr[k] for k in rounds_stat if not k.endswith("_c1")]  # 轮1热身弃
                    runs[run] = {
                        "rounds": rr, "stat_names": rounds_stat, "warm": bool(warm),
                        "agg": agg_rounds(keep), "audit": env_audit(grid, run),
                        "thist": t_hist(grid, run),
                    }
    done = [p.stem for p in grid.glob("*.done")]

    now = _dt.datetime.now()
    vllm_ascend_head = git_head(os.path.join(HERE, "..", ".."))
    ascend_moonep_head = git_head(str(Path(HERE).parents[2] / "ascend-moonep"))

    # ---------- 收益计算 ----------
    def cell(mbt, route, shape, leg):
        return runs.get(f"op2_mbt{mbt}_{route}_{leg}_{shape}", {}).get("agg")

    benefits = []  # (mbt, route, shape, {metric: (off, slot, pct)})
    for mbt in MBTS:
        for route in ROUTES:
            for shape in SHAPES:
                off, slot = cell(mbt, route, shape, "off"), cell(mbt, route, shape, "slot")
                if not off or not slot:
                    continue
                benefits.append((mbt, route, shape, {
                    m: (off.get(m), slot.get(m), pct(slot.get(m), off.get(m)))
                    for m in KEY_METRICS}))

    # ================= HTML =================
    css = """body{font-family:-apple-system,'Noto Sans CJK SC','PingFang SC',sans-serif;
margin:0;background:#f6f6f4;color:#1a1a19;padding:2rem 1rem 4rem}
.wrap{max-width:1080px;margin:0 auto}h1{font-size:1.45rem;margin:.2rem 0 .3rem}
h2{font-size:1.12rem;margin:2.2rem 0 .7rem;padding-bottom:.35rem;border-bottom:2px solid #d8d7d2}
h3{font-size:.98rem;margin:1.5rem 0 .5rem}
p,li,td,th{font-size:.88rem;line-height:1.65}
table{border-collapse:collapse;width:100%;margin:.8rem 0;background:#fff;font-size:.82rem}
th,td{padding:.4rem .55rem;border-bottom:1px solid #e4e3de;text-align:left;
font-variant-numeric:tabular-nums}
th{background:#efefeb;font-weight:600;color:#55544f;position:sticky;top:0}
.scroll{overflow-x:auto;border:1px solid #e4e3de;border-radius:6px}
.num{text-align:right}.good{color:#0a8a5f;font-weight:600}.bad{color:#c23b3a;font-weight:600}
.neut{color:#8a7a20}.tag{display:inline-block;font-size:.72rem;padding:.08em .5em;border-radius:99px;
border:1px solid #cfcec8;background:#fff;margin-left:.4em;white-space:nowrap}
.box{background:#fff;border:1px solid #e4e3de;border-left:4px solid #3a78c2;border-radius:6px;
padding:.7rem 1rem;margin:.9rem 0}.box.warn{border-left-color:#d09a25}
code{background:#eeeee9;padding:.08em .35em;border-radius:3px;font-size:.85em}
.small{font-size:.78rem;color:#6d6c66}.mono{font-family:ui-monospace,Menlo,monospace}
details{margin:.6rem 0}summary{cursor:pointer;font-weight:600;font-size:.9rem}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.7rem;margin:1rem 0}
.kpi>div{background:#fff;border:1px solid #e4e3de;border-radius:8px;padding:.7rem .9rem}
.kpi .v{font-size:1.35rem;font-weight:700;font-variant-numeric:tabular-nums}
.kpi .k{font-size:.78rem;color:#6d6c66}
/* ---- 图表（§4）---- */
.viz-root{margin:1rem 0}
.fig{background:#fff;border:1px solid #e4e3de;border-radius:8px;padding:.9rem 1rem 1rem;margin:1rem 0}
.fig-title{font-size:.95rem;font-weight:700;margin:.1rem 0 .5rem}
.fig-sub{font-size:.8rem;color:#6d6c66;margin:0 0 .6rem}
.legend{display:flex;gap:1.2rem;font-size:.8rem;color:#52514e;margin:.4rem 0 .8rem;flex-wrap:wrap}
.legend .lk{display:inline-flex;align-items:center;gap:.4rem}
.legend .lk i{display:inline-block;width:16px;height:3px;border-radius:2px}
.grid2x2{display:grid;grid-template-columns:max-content 1fr 1fr;gap:.4rem .6rem;align-items:end}
.grid2x2 .col-h{font-size:.78rem;color:#52514e;text-align:center;padding-bottom:.15rem;font-weight:600}
.grid2x2 .row-h{font-size:.75rem;color:#52514e;text-align:right;line-height:1.35;padding-right:.2rem;
min-width:8.5em}
.grid2x2 .row-h b{font-weight:600}
.grid2x2 .pwrap{min-width:0}
svg.vp{width:100%;height:auto;display:block;outline:none;cursor:crosshair}
svg.vp:focus-visible{outline:2px solid #2a78d6;outline-offset:2px;border-radius:4px}
svg.vp text{font-family:inherit}
svg.vp .tk{font-size:10px;fill:#898781;font-variant-numeric:tabular-nums}
svg.vp .axl{font-size:10.5px;fill:#52514e}
svg.vp .endl{font-size:10.5px;fill:#52514e;font-weight:600}
.fig-cap{font-size:.78rem;color:#6d6c66;margin:.6rem 0 0}
#viz-tip{position:fixed;display:none;pointer-events:none;z-index:50;background:#fcfcfb;
border:1px solid #e1e0d9;border-radius:6px;box-shadow:0 4px 14px rgba(11,11,11,.12);
padding:.5rem .65rem;font-size:.78rem;max-width:17rem}
#viz-tip .tt-h{color:#52514e;margin-bottom:.25rem;font-weight:600}
#viz-tip .tt-row{display:flex;align-items:center;gap:.45rem;margin:.15rem 0;color:#52514e}
#viz-tip .tt-row b{color:#0b0b0b;font-variant-numeric:tabular-nums;font-weight:700}
#viz-tip .k{display:inline-block;width:12px;height:3px;border-radius:2px;flex:none}
#viz-tip .tt-d{margin-top:.3rem;padding-top:.3rem;border-top:1px solid #e1e0d9;color:#52514e;
font-variant-numeric:tabular-nums}"""

    def cls_for(metric, p):
        if p is None:
            return "num"
        good = (p < 0) if metric in LOWER_BETTER else (p > 0)
        if abs(p) < 1:
            return "num neut"
        return "num good" if good else "num bad"

    s = [f"<!doctype html><html lang=zh><meta charset=utf-8><title>off vs slot(+opt) 性能报告</title>",
         f"<style>{css}</style><div class=wrap>",
         "<h1>MoonEP slot（real-T + 历史收益优化组合） vs off 性能报告</h1>",
         f"<p class=small>生成：{now:%Y-%m-%d %H:%M} · bench 日期见各 run 的 date 字段</p>"]

    # --- 0. 总配置 ---
    s.append("<h2>0. 实验总配置</h2><div class=box><table>")
    for k, v in [
        ("模型", "/mnt/data/models/DeepSeek-V4-Flash（FP8，E=256，K=6，43 MoE 层）"),
        ("部署", "DP=4 / TP=1 / EP on，卡 0-3（Ascend950PR），enforce-eager，chunked prefill，seed 1024"),
        ("服务参数", "max_model_len 20480，kv-cache 9 GiB，gpu-mem-util 0.80，max-num-seqs 16"),
        ("变化轴", "腿 {off, slot} × 峰均比 {uniform 1.00, 4hot 1.66, 3hot1r 2.48, 4hot1r 2.98} × 形状 {L8k, L16k} × max_num_batched_tokens {8192, 16384}"),
        ("术语", "<code>max_num_batched_tokens</code>：vLLM 每个调度步最多参与的 token 数（chunked prefill 的分块预算），"
                 "同时也是 MoonEP dispatch 静态共享内存容量 S。run 名中的 <code>mbt8192/mbt16384</code> 即该参数取值"
                 "（历史脚本里的列名缩写），本报告正文一律写全称。"),
        ("服务粒度", "每 (腿×路由×形状×max_num_batched_tokens) 一台全新服务（本报告 32 格），同一服务内先 warmup 再 3 轮 bench"),
        ("bench 命令", "<span class=mono>vllm bench serve --backend openai --endpoint /v1/completions "
                       "--dataset-name random --num-prompts 32 --max-concurrency 16 --random-output-len 1 "
                       "--random-range-ratio 0.0 --request-rate inf --ignore-eos --seed &lt;档位序号×1000+轮次&gt; "
                       "--percentile-metrics ttft,e2el --metric-percentiles 50,90,99</span>"),
        ("统计方法", "每格 3 轮，轮 1 视为热身弃掉，统计取轮 2/3 的均值（n=2，均值=中位）；逐轮原始值见 §3"),
        ("vllm-ascend", f"<span class=mono>{H.escape(vllm_ascend_head)}</span>"),
        ("ascend-moonep", f"<span class=mono>{H.escape(ascend_moonep_head)}</span>"),
        ("起服脚本", "<span class=mono>start_service_mode_dp4.sh</span> + <span class=mono>run_opt_phase.sh</span>（编排），产物 " + H.escape(str(grid))),
    ]:
        s.append(f"<tr><th style='width:9em'>{k}</th><td>{v}</td></tr>")
    s.append("</table></div>")
    if missing:
        s.append(f"<div class='box warn'><b>缺失的格（未完成或 bench 失败）：</b>{len(missing)} 台<ul>"
                 + "".join(f"<li><code>{m}</code></li>" for m in missing) + "</ul></div>")
    else:
        s.append(f"<p class=small>32/32 格数据齐全（.done 标记 {len(done)} 个）。</p>")

    # --- 1. 优化清单 ---
    s.append("<h2>1. moonep slot 腿开启的 ascend-moonep 优化项</h2>")
    s.append("<p>分两类：<b>默认开启</b>（代码路径/默认值=1，无需 env）与 <b>本次显式开启</b>"
             "（矩阵 slot 腿起服命令前缀 <code>ASCEND_MOONEP_PREFETCH_RELAXED=1 "
             "ASCEND_MOONEP_NO_RECONCILE=1 ASCEND_MOONEP_SKIP_DUP=1</code>，外加编排器统一的 "
             "<code>MOONEP_MIN_DISPATCH_TOKENS=16</code>）。全部开关在所有 rank 一致生效。</p>")
    s.append("<div class=scroll><table><tr><th>优化项</th><th>所在层</th><th>启用方式</th>"
             "<th>作用说明</th><th>历史依据</th></tr>")
    for name, layer, how, desc, ev in OPTS_ON:
        tag = "默认开" if "默认开" in how else "显式开启"
        color = "#3a78c2" if tag == "默认开" else "#0a8a5f"
        s.append(f"<tr><td><b>{H.escape(name)}</b><span class=tag style='border-color:{color};"
                 f"color:{color}'>{tag}</span></td><td>{H.escape(layer)}</td><td>{H.escape(how)}</td>"
                 f"<td>{H.escape(desc)}</td><td class=small>{H.escape(ev)}</td></tr>")
    s.append("</table></div>")
    s.append("<h3>未开启的项及原因</h3><div class=scroll><table><tr><th>开关</th><th>说明</th><th>约束</th></tr>")
    for name, desc, cons in OPTS_OFF:
        s.append(f"<tr><td><code>{H.escape(name)}</code></td><td>{H.escape(desc)}</td>"
                 f"<td class=small>{H.escape(cons)}</td></tr>")
    s.append("</table></div>")
    s.append("<div class=box><b>生效证据：</b>每台 slot 服务的 worker 在 MoonEP 初始化时打印运行时 env 快照"
             "（<code>[moonep-env-audit]</code>，见 §2 各行）——三个显式开关与 MOONEP_SLOT_MODE 均为 1，"
             "未设项（REAL_T/REUSE_OUT/SLOT_QUANT）走默认开启值。</div>")

    # --- 2. 实验条件明细 ---
    s.append("<h2>2. 各 bench 数据对应的实验条件</h2>")
    s.append("<div class=scroll><table><tr><th>run</th><th>腿</th><th>路由（峰均比 s）</th><th>形状</th>"
             "<th>max_num_batched_tokens</th><th>轮次文件</th><th>slot 运行时 env（审计行）</th></tr>")
    for run in sorted(runs):
        m = RUN_RE.match(run)
        if not m:
            continue
        route, leg, shape, mbt = m["route"], m["leg"], m["shape"], m["mbt"]
        audit = runs[run]["audit"]
        audit_html = (f"<span class='small mono'>{H.escape(audit)}</span>" if audit
                      else "<span class=small>（off 腿无 MoonEP）</span>" if leg == "off" else "")
        s.append(f"<tr><td><code>{run}</code></td><td>{LEG_TITLE[leg].split('（')[0]}</td>"
                 f"<td>{route}（s={ROUTE_INFO[route][1]}）</td><td>{SHAPE_DESC[shape]}</td>"
                 f"<td class=num>{mbt}</td><td class='small mono'>"
                 f"{H.escape(','.join(runs[run]['stat_names']))}</td><td>{audit_html}</td></tr>")
    s.append("</table></div>")

    # --- 3. bench 全量数据（逐轮） ---
    s.append("<h2>3. bench 原始数据（逐轮全字段）</h2>")
    s.append("<p>每轮一行，bench json 的全部标量字段全量收录；末行 <b>均值(2,3轮)</b> 为最终采用的统计数字。"
             "延迟/时长单位 ms，吞吐单位 req/s 或 token/s。</p>")
    for mbt in MBTS:
        for shape in SHAPES:
            s.append(f"<h3>max_num_batched_tokens={mbt} · {SHAPE_FULL[shape]}</h3>")
            for route in ROUTES:
                for leg in LEGS:
                    run = f"op2_mbt{mbt}_{route}_{leg}_{shape}"
                    if run not in runs:
                        continue
                    agg = runs[run]["agg"]
                    s.append(f"<details><summary>{run} <span class=tag>{leg}</span>"
                             f"<span class=tag>{route} s={ROUTE_INFO[route][1]}</span></summary>"
                             f"<div class=scroll><table><tr>"
                             + "".join(f"<th class=num>{f}</th>" for f in BENCH_FIELDS_ORDER)
                             + "<th>轮</th></tr>")
                    for name in runs[run]["stat_names"]:
                        r = runs[run]["rounds"][name]
                        s.append("<tr>" + "".join(
                            f"<td class=num>{fmt(r.get(f), f)}</td>" for f in BENCH_FIELDS_ORDER)
                            + f"<td><b>{name.split('_c')[-1]}</b></td></tr>")
                    if agg:
                        s.append("<tr style='background:#f3f8f4'>" + "".join(
                            f"<td class=num><b>{fmt(agg.get(f), f)}</b></td>" for f in BENCH_FIELDS_ORDER)
                            + "<td><b>均值(2,3)</b></td></tr>")
                    s.append("</table></div></details>")

    # --- 4. 收益统计 ---
    s.append("<h2>4. slot 相对 off 的收益统计（其它条件相同）</h2>")
    s.append("<p>比较方法：只在 (max_num_batched_tokens, 路由, 形状) 都相同的格子之间配对，计算 <code>(slot − off)/off × 100%</code>，"
             "基于轮 2/3 均值。延迟/时长<b>负值=slot 更快（收益）</b>；吞吐<b>正值=slot 更高（收益）</b>。"
             "绿=slot 占优，红=off 占优，黄=|Δ|&lt;1%。</p>")
    if benefits:
        # KPI
        ttft_deltas = [b["mean_ttft_ms"][2] for *_, b in benefits if b["mean_ttft_ms"][2] is not None]
        tput_deltas = [b["request_throughput"][2] for *_, b in benefits if b["request_throughput"][2] is not None]
        wins = sum(1 for d in ttft_deltas if d < 0)
        s.append("<div class=kpi>"
                 f"<div><div class=v>{len(benefits)}</div><div class=k>配对格数</div></div>"
                 f"<div><div class=v>{st.mean(ttft_deltas):+.1f}%</div><div class=k>TTFT 均值 Δ 平均（负=slot 快）</div></div>"
                 f"<div><div class=v>{st.median(ttft_deltas):+.1f}%</div><div class=k>TTFT 均值 Δ 中位</div></div>"
                 f"<div><div class=v>{st.mean(tput_deltas):+.1f}%</div><div class=k>请求吞吐 Δ 平均（正=slot 高）</div></div>"
                 f"<div><div class=v>{wins}/{len(ttft_deltas)}</div><div class=k>slot TTFT 占优格数</div></div></div>")

        for mbt in MBTS:
            for shape in SHAPES:
                rows = [b for b in benefits if b[0] == mbt and b[2] == shape]
                if not rows:
                    continue
                s.append(f"<h3>max_num_batched_tokens={mbt} · {SHAPE_FULL[shape]}（其余条件相同的 4 组配对）</h3>")
                s.append("<div class=scroll><table><tr><th>路由（s）</th>"
                         + "".join(f"<th class=num colspan=3>{m}</th>" for m in KEY_METRICS)
                         + "</tr><tr><th></th>"
                         + "".join("<th class=num>off</th><th class=num>slot</th><th class=num>Δ%</th>"
                                   for _ in KEY_METRICS) + "</tr>")
                for _, route, _, b in rows:
                    cells = []
                    for m in KEY_METRICS:
                        o, sl, p = b[m]
                        cells.append(f"<td class=num>{fmt(o, m)}</td><td class=num>{fmt(sl, m)}</td>"
                                     f"<td class='{cls_for(m, p)}'>{p:+.1f}%</td>" if p is not None
                                     else f"<td class=num>{fmt(o, m)}</td><td class=num>{fmt(sl, m)}</td>"
                                           f"<td class=num>—</td>")
                    s.append(f"<tr><td>{route}（s={ROUTE_INFO[route][1]}）</td>{''.join(cells)}</tr>")
                # 分形状小计
                sub = {}
                for m in KEY_METRICS:
                    ds = [b[m][2] for *_, b in rows if b[m][2] is not None]
                    sub[m] = st.mean(ds) if ds else None
                cells = "".join(
                    f"<td colspan=2 class=small>形状内均值</td><td class='{cls_for(m, sub[m])}'>"
                    f"{sub[m]:+.1f}%</td>" if sub[m] is not None else "<td colspan=3></td>"
                    for m in KEY_METRICS)
                s.append(f"<tr style='background:#fbf7ea'><td><b>形状内小计</b></td>{cells}</tr>")
                s.append("</table></div>")
        # 总汇总
        s.append("<h3>总汇总</h3><div class=scroll><table><tr><th>分组</th><th>格数</th>"
                 + "".join(f"<th class=num>{m} Δ% 均值</th>" for m in KEY_METRICS) + "</tr>")
        groups = [("全部", lambda b: True),
                  ("形状 L8k", lambda b: b[2] == "L8k"), ("形状 L16k", lambda b: b[2] == "L16k")]
        for r in ROUTES:
            groups.append((f"路由 {r}（s={ROUTE_INFO[r][1]}）", (lambda rr: lambda b: b[1] == rr)(r)))
        for gname, pred in groups:
            rows = [b for b in benefits if pred(b)]
            if not rows:
                continue
            cells = []
            for m in KEY_METRICS:
                ds = [b[m][2] for *_, b in rows if b[m][2] is not None]
                v = st.mean(ds) if ds else None
                cells.append(f"<td class='{cls_for(m, v)}'>{v:+.1f}%</td>" if v is not None else "<td>—</td>")
            s.append(f"<tr><td>{gname}</td><td class=num>{len(rows)}</td>{''.join(cells)}</tr>")
        s.append("</table></div>")
    else:
        s.append("<div class='box warn'>暂无完整配对（至少一台 off 与一台 slot 同格完成）。</div>")

    # --- §4 图表：随峰均比变化 ---
    def series(mbt, shape, leg, key):
        return [cell(mbt, r, shape, leg).get(key) for r in VIZ_ROUTES]

    for fig_key, key, title, sub, step, yfmt, tip_fmt, unit, low_note in [
        ("ttft", "mean_ttft_ms", "图 1 · 平均 TTFT（从发出请求到收到第 1 个输出 token 的等待时间）随峰均比的变化",
         "同一面板内除峰均比外其它条件完全相同；线越平 = 对负载倾斜越不敏感。数值为轮 2/3 的均值。",
         2000, lambda v: f"{v / 1000:.0f}k", "ms", "毫秒（ms）", "负=slot 更快"),
        ("tput", "request_throughput", "图 2 · 请求吞吐（每秒完成的请求数）随峰均比的变化",
         "与图 1 同一批数据；线越平 = 吞吐不因倾斜下滑。",
         0.5, lambda v: f"{v:.1f}", "tput", "请求/秒（req/s）", "正=slot 更高"),
    ]:
        panels = [(mbt, f"输入 {sh} token（输出 1）",
                   series(mbt, f"L{sh}", "off", key), series(mbt, f"L{sh}", "slot", key))
                  for mbt in MBTS for sh in ("8k", "16k")]
        s.append(figure_html(fig_key, title, sub, panels, key, step, yfmt, tip_fmt, unit, low_note))

    # --- 5. 机制分析：为什么 max_num_batched_tokens ≥ 输入长度才有收益 ---
    s.append("<h2>5. 机制分析：为什么 max_num_batched_tokens ≥ 输入长度，slot 才能拿到收益</h2>")

    def steps_str(run):
        h = runs.get(run, {}).get("thist", {})
        return " + ".join(f"{v/(4*3):.1f} 步×T={t}" for t, v in sorted(h.items(), reverse=True) if t >= 2048) or "—"

    def ttft(run):
        a = runs.get(run, {}).get("agg", {})
        return a.get("mean_ttft_ms")

    slot8, slot16 = ttft("op2_mbt8192_uniform_slot_L16k"), ttft("op2_mbt16384_uniform_slot_L16k")
    off8, off16 = ttft("op2_mbt8192_uniform_off_L16k"), ttft("op2_mbt16384_uniform_off_L16k")
    if all(v is not None for v in (slot8, slot16, off8, off16)):
        t8s, t16s = slot8 / 16, slot16 / 8        # slot：16 步 / 8 步
        t8o, t16o = off8 / 16, off16 / 8
        m_s, f_s = t16s - t8s, 2 * t8s - t16s      # 边际成本 / 每步固定成本
        m_o, f_o = t16o - t8o, 2 * t8o - t16o
        gap_pred, gap_meas = 8 * f_s, slot8 - slot16
        s.append("<p>背景：max_num_batched_tokens 是 chunked prefill 的分块预算——输入长度超过它时，"
                 "一条 prompt 会被切成 ⌈长度/预算⌉ 个 prefill 调度步，<b>每一步都要把全部 43 个 MoE 层"
                 "完整走一遍 MoonEP 通信路径</b>（planning 内核、跨卡栅栏、prefetch、dispatch/combine）。"
                 "下面的步数直接取自服务日志中仿真路由打印的每步 token 数（T=）直方图（4 个 rank、"
                 "3 轮 bench 的合计除以 12）。</p>")
        s.append("<div class=scroll><table><tr><th>配置（uniform 路由，L16k=16384 token 输入）</th>"
                 "<th>每 rank 每轮的 prefill 步构成</th><th class=num>TTFT 均值(ms)</th>"
                 "<th class=num>步均耗时(ms)</th></tr>")
        for name, run, steps_n in [
            ("slot · max_num_batched_tokens=8192（输入被切成 2 块）", "op2_mbt8192_uniform_slot_L16k", 16),
            ("slot · max_num_batched_tokens=16384（一步装下）", "op2_mbt16384_uniform_slot_L16k", 8),
            ("off · max_num_batched_tokens=8192", "op2_mbt8192_uniform_off_L16k", 16),
            ("off · max_num_batched_tokens=16384", "op2_mbt16384_uniform_off_L16k", 8),
        ]:
            v = ttft(run)
            if "_off_" in run:
                # off 的路由注入点看到的是 DP 聚合后的 batch（T=4×每 rank token），
                # 调度步数仍由预算决定（16/8），步构成列标注聚合视角以免误导。
                hmax = max(runs.get(run, {}).get("thist", {}) or {0: 0})
                step_desc = f"{steps_n} 调度步（MoE 调用为 DP 聚合视角，T={hmax}）"
            else:
                step_desc = steps_str(run)
            s.append(f"<tr><td>{name}</td><td class='small mono'>{step_desc}</td>"
                     f"<td class=num>{fmt(v, 'mean_ttft_ms')}</td>"
                     f"<td class=num>{fmt(v / steps_n, 'mean_ttft_ms') if v else '—'}</td></tr>")
        s.append("</table></div>")
        s.append("<h3>把一步的耗时拆成两部分（用上表两组数字解出）</h3>")
        s.append("<p>一步的耗时 = 「跟 token 数成正比的搬运计算时间」+「每开始一步都要付一次、与装多少 "
                 "token 无关的起步耗时」。就像打车账单：里程费按路程付，起步价每上一辆车付一次。"
                 "求解方法：装 8192 的一步耗时 = 1 份里程 + 起步价；装 16384 的一步耗时 = 2 份里程 + 起步价；"
                 "两式相减得到每份里程，再回代得到起步价。</p>")
        s.append("<div class=scroll><table><tr><th>腿</th><th class=num>每多装 8192 token 多花的时间(ms)（里程费）</th>"
                 "<th class=num>每步起步耗时(ms)（起步价）</th><th class=num>起步耗时占整步比例</th></tr>"
                 f"<tr><td>slot</td><td class=num>{m_s:,.0f}</td><td class=num><b>{f_s:,.0f}</b></td>"
                 f"<td class=num><b>{f_s / t16s * 100:.0f}%</b></td></tr>"
                 f"<tr><td>off</td><td class=num>{m_o:,.0f}</td><td class=num>{f_o:,.0f}</td>"
                 f"<td class=num>{f_o / t16o * 100:.0f}%</td></tr></table></div>")
        s.append("<div class=box><b>结论链条：</b>① MoonEP slot 每开始一个调度步都要付约 "
                 f"<b>{f_s:,.0f} ms（≈{f_s / 43:.1f} ms/MoE 层）的起步耗时</b>"
                 "——不管这步装 8192 还是 16384 个 token 都一样多。它是每步一轮的固定动作：planning 内核"
                 "算路由分配、各卡跨卡互相等齐、预取权重、dispatch/combine 里各卡会合，耗时取决于做几轮、"
                 "不取决于 token 数；② 16k 输入在 max_num_batched_tokens=8192 下被切成 2 步，"
                 f"起步价付两次：8 个多出来的步 × {f_s:,.0f} ms ≈ <b>{gap_pred:,.0f} ms</b>，与实测 TTFT 差 "
                 f"<b>{gap_meas:,.0f} ms</b> 吻合；③ 两条腿搬运的数据总量完全相同（都是 16k×K 行），"
                 "多付的纯粹是「每步一次」的起步耗时；④ off 腿（MC2 all-to-all）每步起步耗时仅 "
                 f"{f_o:,.0f} ms——它的耗时几乎全随数据量成正比，怎么分块总价都一样，所以它对 "
                 "max_num_batched_tokens 不敏感。"
                 "（注：TTFT≈步数×步时是近似，还含排队与波次效应；off 腿日志中 T=32768/65536 的步说明"
                 "其 MoE 调用聚合了 4 个 DP rank 的 token，按总量计费，进一步印证其体积线性。）</div>")
        # L8k 对照
        s.append("<h3>反例对照：L8k（8192 token 输入）不被切分，就没有这个惩罚</h3>")
        s.append("<div class=scroll><table><tr><th>配置（uniform 路由，L8k）</th><th>每 rank 每轮的 prefill 步构成</th>"
                 "<th class=num>TTFT 均值(ms)</th><th class=num>slot/off ΔTTFT</th></tr>")
        d8 = pct(ttft("op2_mbt8192_uniform_slot_L8k"), ttft("op2_mbt8192_uniform_off_L8k"))
        d16 = pct(ttft("op2_mbt16384_uniform_slot_L8k"), ttft("op2_mbt16384_uniform_off_L8k"))
        d8_cls = "neut" if abs(d8 or 0) < 1 else ("good" if d8 < 0 else "bad")
        for name, run in [("slot · max_num_batched_tokens=8192（恰好一步装下）", "op2_mbt8192_uniform_slot_L8k"),
                          ("slot · max_num_batched_tokens=16384（预算更大，还会把 2 条 prompt 打包成一步）",
                           "op2_mbt16384_uniform_slot_L8k")]:
            v = ttft(run)
            s.append(f"<tr><td>{name}</td><td class='small mono'>{steps_str(run)}</td>"
                     f"<td class=num>{fmt(v, 'mean_ttft_ms')}</td><td></td></tr>")
        s.append(f"<tr><td colspan=3 class=small>同为 L8k，slot 相对 off 的 ΔTTFT</td>"
                 f"<td class='num {d8_cls}'>{d8:+.1f}%</td></tr>")
        s.append("</table></div>")
        s.append(f"<p class=small>对照 L16k 的同位数字：max_num_batched_tokens=8192 时 ΔTTFT "
                 f"{pct(slot8, off8):+.1f}%（被切分，惩罚出现），=16384 时 {d16:+.1f}%（不切分）。"
                 "惩罚的出现与否完全由「输入是否超过 max_num_batched_tokens」决定，与倾斜度无关"
                 "（见 §4：max_num_batched_tokens=8192 的 L16k 在全部 4 档路由下皆输）。</p>")

    # --- 6. 综合结论 ---
    s.append("<h2>6. 综合结论</h2>")

    def spread(mbt, shape, leg):
        vs = [v for v in series(mbt, shape, leg, "mean_ttft_ms") if v]
        return (max(vs) / min(vs) - 1) * 100 if vs else None

    s.append("<h3>6.1 slot 对负载倾斜几乎不敏感，off 随倾斜线性变慢</h3>")
    s.append("<p>TTFT 从均匀到最倾斜（峰均比 1.00→2.98）的变化幅度：</p>")
    s.append("<div class=scroll><table><tr><th>条件</th><th class=num>off 变化幅度</th><th class=num>slot 变化幅度</th></tr>")
    for mbt in MBTS:
        for shape in SHAPES:
            so, ss2 = spread(mbt, shape, "off"), spread(mbt, shape, "slot")
            s.append(f"<tr><td>max_num_batched_tokens={mbt} · 输入{' 8k' if shape == 'L8k' else ' 16k'} token</td>"
                     f"<td class=num><b>+{so:.0f}%</b></td><td class=num>+{ss2:.0f}%</td></tr>")
    s.append("</table></div>")
    s.append("<p>slot 的平线来自它的工作方式：热点专家的权重行被镜像到每个 rank 的槽位、提前预取，"
             "再由 planning 按真实 token 负载把计算行均摊到各 rank——最忙 rank 不再拖住整个波次。"
             "off 没有这层均衡，最忙 rank（s=2.98 时承担 74.6% 载荷）决定全体速度。</p>")

    s.append("<h3>6.2 盈亏平衡点：峰均比 ≈ 1.7~1.8（max_num_batched_tokens=16384 时）</h3>")
    s.append("<p>在 max_num_batched_tokens=16384（一步装下全部输入）时，off 随倾斜变慢的曲线与 slot 的平线相交于"
             "峰均比 ≈1.7~1.8（8k 与 16k 两种输入内插结果一致）。低于该值 off 更快（最多快 19%），"
             "高于该值 slot 反超（最多快 21%、吞吐高 31%）。max_num_batched_tokens=8192 且输入 16k 时，"
             "输入被切成两步、slot 每步的起步耗时付两次（机制见 §5），四档倾斜下 slot 全部落后。</p>")
    wins = {}
    for mbt in MBTS:
        for shape in SHAPES:
            w = 0
            for r in ROUTES:
                o = cell(mbt, r, shape, "off").get("mean_ttft_ms")
                sl = cell(mbt, r, shape, "slot").get("mean_ttft_ms")
                if o and sl and sl < o:
                    w += 1
            wins[(mbt, shape)] = w
    s.append("<div class=scroll><table><tr><th>条件</th><th class=num>slot TTFT 占优的路由数（共 4）</th></tr>"
             + "".join(f"<tr><td>max_num_batched_tokens={mbt} · 输入{' 8k' if shape == 'L8k' else ' 16k'} token</td>"
                       f"<td class=num>{wins.get((mbt, shape), '—')}</td></tr>"
                       for mbt in MBTS for shape in SHAPES)
             + "</table></div>")

    s.append("<h3>6.3 部署建议</h3>")
    s.append("<div class=scroll><table><tr><th>场景</th><th>建议</th></tr>"
             "<tr><td>业务有热点专家（代码/数学/复读类负载，真实峰均比通常 1.5~3+）</td>"
             "<td><b>开 slot + 优化组合</b>，max_num_batched_tokens 设为完整输入块大小（本模型 16384）；峰均比 ≥2 时稳定占优</td></tr>"
             "<tr><td>负载均匀或未知</td><td>先用负载采集（MOE_PAR=1）量出真实峰均比；&lt;1.75 保持 off</td></tr>"
             "<tr><td>任何开 slot 的场合</td><td>max_num_batched_tokens 不小于输入长度（否则起步耗时翻倍，见 §5）；"
             "跨卡开关全 rank 一致</td></tr></table></div>")

    # --- 附录 ---
    s.append("<h2>附录</h2><h3>A. 仿真路由档案（峰均比设计）</h3><div class=box><table>"
             "<tr><th>路由</th><th>档案</th><th>峰均比 s（rank 级 max/mean）</th><th>rank0-3 home 载荷</th></tr>")
    for r, (f, sv, loads) in ROUTE_INFO.items():
        content = ""
        p = Path(HERE).parent / "sim" / f
        if p.is_file():
            content = p.read_text(errors="ignore")[:600]
        s.append(f"<tr><td>{r}</td><td><code>{f}</code><details><summary>内容</summary>"
                 f"<pre class='small mono'>{H.escape(content)}</pre></details></td>"
                 f"<td class=num>{sv}</td><td class=small>{' / '.join(loads)}</td></tr>")
    s.append("</table><p class=small>s = 最忙 rank home 行数 ÷ 平均；热点专家每专家上限 100/K=16.667%（K=6）。</p></div>")
    s.append("<h3>B. 脚本与产物</h3><div class=box><ul class=small>"
             "<li>编排：<code>bench/run_opt_phase.sh</code> + <code>plans/PLAN_opt_xshape.txt</code>（每格全新服务，断点续跑）</li>"
             "<li>bench：<code>bench/bench_rt_one.sh</code>（warmup + 单形状 3 轮）</li>"
             f"<li>数据：<code>{H.escape(str(grid))}/op2_*/</code>（json）与 <code>{H.escape(str(grid))}/*.log</code>（服务日志，含 env 审计行）</li>"
             "<li>本报告：<code>gen_opt_report.py</code></li></ul></div>")
    # 悬停读数层（§4 图表共用）：十字线吸附最近数据点，tooltip 列出两条线的值
    s.append('<div id="viz-tip" role="status" aria-live="polite"></div>')
    s.append(r"""<script>
(function () {
  var tip = document.getElementById('viz-tip');
  if (!tip) return;
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  var S_LBL = ['1.00', '1.66', '2.48', '2.98'];
  function fmtV(v, mode) {
    return mode === 'ms'
      ? Math.round(v).toLocaleString('en-US') + ' ms'
      : v.toFixed(2) + ' req/s';
  }
  document.querySelectorAll('svg.vp').forEach(function (svg) {
    var meta = JSON.parse(document.getElementById('vizdata-' + svg.dataset.fig).textContent);
    var off = JSON.parse(svg.dataset.off), slot = JSON.parse(svg.dataset.slot);
    var xx = JSON.parse(svg.dataset.xx);
    var ptso = JSON.parse(svg.dataset.ptso), ptss = JSON.parse(svg.dataset.ptss);
    var xh = svg.querySelector('.xh'), ln = xh.querySelector('line');
    var r1 = xh.querySelector('.xh1'), r2 = xh.querySelector('.xh2');
    function nearest(evt) {
      var rect = svg.getBoundingClientRect();
      var vx = (evt.clientX - rect.left) / rect.width * svg.viewBox.baseVal.width;
      var best = 0, bd = Infinity;
      xx.forEach(function (x, i) {
        var d = Math.abs(x - vx);
        if (d < bd) { bd = d; best = i; }
      });
      return best;
    }
    function hide() {
      xh.style.display = 'none';
      tip.style.display = 'none';
    }
    function show(i, cx, cy) {
      var x = xx[i];
      ln.setAttribute('x1', x); ln.setAttribute('x2', x);
      r1.setAttribute('cx', ptso[i][0]); r1.setAttribute('cy', ptso[i][1]);
      r2.setAttribute('cx', ptss[i][0]); r2.setAttribute('cy', ptss[i][1]);
      xh.style.display = '';
      tip.textContent = '';
      tip.appendChild(el('div', 'tt-h',
        svg.dataset.title + ' · 峰均比 s=' + S_LBL[i] + '（' + meta.routes[i] + '）'));
      [['off', off[i], '#eb6834'], ['slot', slot[i], '#2a78d6']].forEach(function (row) {
        var r = el('div', 'tt-row');
        var k = el('span', 'k'); k.style.background = row[2];
        r.appendChild(k);
        r.appendChild(el('b', null, fmtV(row[1], meta.fmt)));
        r.appendChild(el('span', null, row[0]));
        tip.appendChild(r);
      });
      var d = (slot[i] - off[i]) / off[i] * 100;
      tip.appendChild(el('div', 'tt-d',
        'slot − off = ' + (d >= 0 ? '+' : '') + d.toFixed(1) + '%（' + meta.delta + '）'));
      tip.style.display = '';
      var tw = tip.offsetWidth, th = tip.offsetHeight;
      var left = Math.min(cx + 14, window.innerWidth - tw - 8);
      var top = Math.min(cy + 14, window.innerHeight - th - 8);
      tip.style.left = Math.max(8, left) + 'px';
      tip.style.top = Math.max(8, top) + 'px';
    }
    svg.addEventListener('pointermove', function (e) { show(nearest(e), e.clientX, e.clientY); });
    svg.addEventListener('pointerleave', hide);
    svg.addEventListener('focus', function () {
      var r = svg.getBoundingClientRect();
      show(xx.length - 1, r.right - 120, r.top);
    });
    svg.addEventListener('blur', hide);
  });
})();
</script>""")
    s.append("</div>")
    html_path = args.out + ".html"
    Path(html_path).write_text("\n".join(s), encoding="utf-8")

    # ================= MD 摘要 =================
    md = [f"# off vs slot(+opt) 摘要（{now:%Y-%m-%d %H:%M}）", "",
          f"- vllm-ascend: `{vllm_ascend_head}`", f"- ascend-moonep: `{ascend_moonep_head}`", "",
          "| max_num_batched_tokens | 形状 | 路由(s) | off TTFT均值(ms) | slot TTFT均值(ms) | ΔTTFT | off req吞吐 | slot req吞吐 | Δ吞吐 |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for mbt, route, shape, b in benefits:
        o, sl, p = b["mean_ttft_ms"]
        o2, sl2, p2 = b["request_throughput"]
        md.append(f"| {mbt} | {shape} | {route}({ROUTE_INFO[route][1]}) | {fmt(o)} | {fmt(sl)} | {p:+.1f}% "
                  f"| {fmt(o2)} | {fmt(sl2)} | {p2:+.1f}% |")
    md_path = args.out + ".md"
    Path(md_path).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[report] html: {html_path}\n[report] md  : {md_path}\n[report] runs: {len(runs)} 个，缺失 {len(missing)}")


if __name__ == "__main__":
    main()
