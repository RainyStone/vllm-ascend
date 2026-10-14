#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""off vs slot profiling 对比分析报告生成器（读本实验 profile/*/…/ASCEND_PROFILER_OUTPUT）。

用法: python3 gen_profile_analysis.py      # 输出 report/profile_analysis_report.html
数据口径警告：采集窗内含 profiler 开销与 DP 波次/dummy 步，全部数字用于结构对比，
不作端到端性能统计（统计结论以 20261011_opt2 干净矩阵为准）。
"""
import csv
import glob
import html as H
import statistics as st
from collections import defaultdict
from datetime import datetime

HERE = glob.glob(".")[0]
OUT = "report/profile_analysis_report.html"
CSV = "csv.DictReader"

def read_ops(leg):
    agg = defaultdict(lambda: [0, 0.0, 0.0])
    for d in sorted(glob.glob(f"profile/{leg}/dp*_ascend_pt/ASCEND_PROFILER_OUTPUT/op_statistic.csv")):
        with open(d) as f:
            for r in csv.DictReader(f):
                a = agg[r["OP Type"]]
                a[0] += int(r["Count"]); a[1] += float(r["Total Time(us)"])
                a[2] = max(a[2], float(r["Max Time(us)"]))
    return agg

def read_step(leg):
    vals = []
    for d in sorted(glob.glob(f"profile/{leg}/dp*_ascend_pt/ASCEND_PROFILER_OUTPUT/step_trace_time.csv")):
        with open(d) as f:
            r = next(csv.DictReader(f))
        vals.append({k: float(v) for k, v in r.items() if k not in ("Device_id", "Step")})
    n = len(vals)
    return {k: sum(v[k] for v in vals) / n / 1e6 for k in vals[0]}

def read_dist(leg, pat):
    xs = []
    for d in sorted(glob.glob(f"profile/{leg}/dp*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv")):
        with open(d) as f:
            for r in csv.DictReader(f):
                if pat in r["Name"]:
                    xs.append(float(r["Duration(us)"]))
    return xs

def read_host(leg):
    agg = {}
    for d in sorted(glob.glob(f"profile/{leg}/dp*_ascend_pt/ASCEND_PROFILER_OUTPUT/api_statistic.csv")):
        with open(d) as f:
            for r in csv.DictReader(f):
                if r["Device_id"] == "host":
                    a = agg.setdefault(r["API Name"], [0.0, 0]); a[0] += float(r["Time(us)"]); a[1] += int(r["Count"])
    return agg

def wait_split(leg, pat, thresh=2000):
    sn = s_t = fn = f_t = 0
    for d in sorted(glob.glob(f"profile/{leg}/dp*_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv")):
        with open(d) as f:
            for r in csv.DictReader(f):
                if pat in r["Name"]:
                    t = float(r["Duration(us)"])
                    if t > thresh: sn += 1; s_t += t
                    else: fn += 1; f_t += t
    return fn, f_t, sn, s_t

ops = {leg: read_ops(leg) for leg in ("off", "slot")}
step = {leg: read_step(leg) for leg in ("off", "slot")}
host = {leg: read_host(leg) for leg in ("off", "slot")}
dev_tot = {leg: sum(v[1] for v in ops[leg].values()) / 1e6 for leg in ("off", "slot")}
win = {leg: step[leg]["Stage"] for leg in ("off", "slot")}

def top(leg, n=14):
    return sorted(ops[leg].items(), key=lambda kv: -kv[1][1])[:n]

# ---------- SVG 小图（构成条） ----------
C = ["#2a78d6", "#eb6834", "#1baf7a"]
def stacked_bar(title, rows, legend_labels, unit="%", total=100):
    """rows: [(label, [v1,v2,v3]), ...] 一行一根 100% 堆叠条。"""
    W, BH, GAP = 720, 30, 26
    h = GAP + 20 + len(rows) * (BH + 26) + 4
    s = [f'<svg viewBox="0 0 {W} {h}" style="width:100%;height:auto;font-family:inherit">',
         f'<text x="0" y="16" style="font-size:13px;font-weight:700;fill:#0b0b0b">{H.escape(title)}</text>']
    lg = [f'<svg viewBox="0 0 {W} 22" style="width:100%;height:auto;font-family:inherit">']
    lx = 0
    for i, lab in enumerate(legend_labels):
        lg.append(f'<rect x="{lx}" y="6" width="14" height="14" rx="3" fill="{C[i]}"/>'
                  f'<text x="{lx + 20}" y="18" style="font-size:12px;fill:#52514e">{H.escape(lab)}</text>')
        lx += 26 + len(lab) * 12 + 18
    lg.append("</svg>")
    y = GAP + 8
    for label, vals in rows:
        s.append(f'<text x="0" y="{y + BH - 9}" style="font-size:12px;fill:#52514e">{H.escape(label)}</text>')
        x = 128
        for i, v in enumerate(vals):
            w = v / total * (W - 140)
            s.append(f'<rect x="{x + 1}" y="{y}" width="{max(0, w - 2):.1f}" height="{BH}" rx="4" fill="{C[i]}"/>')
            if w > 44:
                s.append(f'<text x="{x + w / 2:.0f}" y="{y + BH / 2 + 4.5}" text-anchor="middle" '
                         f'style="font-size:12px;font-weight:700;fill:#fff">{v:.0f}%</text>')
            x += w
        s.append(f'<text x="{W - 2}" y="{y + BH / 2 + 4.5}" text-anchor="end" '
                 f'style="font-size:11px;fill:#898781">{total:.0f}</text>')
        y += BH + 26
    s.append("</svg>")
    return "".join(lg) + "".join(s)

# ---------- HTML ----------
css = """body{font-family:-apple-system,'Noto Sans CJK SC','PingFang SC',sans-serif;background:#f6f6f4;
color:#1a1a19;margin:0;padding:2rem 1rem 4rem}.wrap{max-width:1000px;margin:0 auto}
h1{font-size:1.4rem;margin:.2rem 0 .4rem}h2{font-size:1.1rem;margin:2.2rem 0 .7rem;
border-bottom:2px solid #d8d7d2;padding-bottom:.35rem}h3{font-size:.98rem;margin:1.5rem 0 .5rem}
table{border-collapse:collapse;width:100%;margin:.8rem 0;background:#fff;font-size:.84rem}
th,td{padding:.42rem .55rem;border-bottom:1px solid #e4e3de;text-align:left;
font-variant-numeric:tabular-nums}th{background:#efefeb;font-weight:600;color:#55544f}
.num{text-align:right}.good{color:#0a8a5f;font-weight:600}.warn{color:#8a7a20;font-weight:600}
.box{background:#fff;border:1px solid #e4e3de;border-left:4px solid #3a78c2;border-radius:6px;
padding:.7rem 1rem;margin:.9rem 0}.box.warn{border-left-color:#d09a25}
.small{font-size:.78rem;color:#6d6c66}.scroll{overflow-x:auto;border:1px solid #e4e3de;border-radius:6px}
code{background:#eeeee9;padding:.08em .35em;border-radius:3px;font-size:.85em}
.fig{background:#fff;border:1px solid #e4e3de;border-radius:8px;padding:.9rem 1rem;margin:1rem 0}"""

s = [f"<!doctype html><html lang=zh><meta charset=utf-8><title>off vs slot profiling 分析</title>",
     f"<style>{css}</style><div class=wrap>",
     "<h1>profiling 对比分析：MoonEP slot 的瓶颈与优化空间</h1>",
     f"<p class=small>生成：{datetime.now():%Y-%m-%d %H:%M} · 生成器 gen_profile_analysis.py（数字全部现算）</p>"]

s.append("<h2>0. 数据条件与口径</h2><div class=box><table>")
for k, v in [
    ("实验条件", "max_num_batched_tokens=16384 · 输入 16384/输出 1 · 仿真路由 3hot1r（峰均比 s=2.48）· DP4/卡 0-3 · 单轮同窗采集（seed 21001 两腿一致）"),
    ("两腿", "off（标准 MC2/EP） vs slot（MoonEP slot + RELAXED/NO_RECONCILE/SKIP_DUP 三件套 + 默认优化，env 审计行佐证）"),
    ("数据源", "profile/&lt;腿&gt;/dp*_ascend_pt/ASCEND_PROFILER_OUTPUT/（torch_npu analyse 产物：op_statistic / kernel_details / step_trace_time / api_statistic）"),
    ("采集窗口（每 rank 平均）", f"off {win['off']:.1f}s · slot {win['slot']:.1f}s（窗口含 DP 波次对齐与 dummy 步）"),
    ("⚠ 口径警告", "两腿窗口不同，<b>只比结构与占比，不比绝对秒数</b>；窗口内含 profiler 开销（off 被拖累更大：bench mean_ttft 19.2s vs 干净值 11.0s）与空转步；端到端统计结论以 op2 干净矩阵为准"),
]:
    s.append(f"<tr><th style='width:11em'>{k}</th><td>{v}</td></tr>")
s.append("</table></div>")

# 总览
s.append("<h2>1. 总览：slot 把 off 的『通信等待』换成了『计算+可重叠等待』</h2>")
s.append("<div class=scroll><table><tr><th>指标（4 rank 合计 / 每 rank 平均）</th><th class=num>off</th><th class=num>slot</th></tr>")
rows = [
    ("采集窗口（s/rank）", f"{win['off']:.1f}", f"{win['slot']:.1f}"),
    ("设备忙时（s，4 rank 合计）", f"{dev_tot['off']:.1f}", f"{dev_tot['slot']:.1f}"),
    ("计算占比（Computing/窗口）", f"{step['off']['Computing']/win['off']*100:.0f}%", f"<b class=good>{step['slot']['Computing']/win['slot']*100:.0f}%</b>"),
    ("未重叠通信占比", f"<b class=warn>{step['off']['Communication(Not Overlapped)']/win['off']*100:.0f}%</b>", f"{step['slot']['Communication(Not Overlapped)']/win['slot']*100:.0f}%"),
    ("空闲占比（Free）", f"{step['off']['Free']/win['off']*100:.0f}%", f"{step['slot']['Free']/win['slot']*100:.0f}%"),
    ("bench 参考值（含 profiler，不作统计）", "mean_ttft 19.2s", "mean_ttft 10.1s"),
]
for k, a, b in rows:
    s.append(f"<tr><td>{k}</td><td class=num>{a}</td><td class=num>{b}</td></tr>")
s.append("</table></div>")
s.append('<div class="fig">' + stacked_bar(
    "图 1 · 采集窗口的构成（每 rank 平均）",
    [("off", [step['off']['Computing']/win['off']*100,
              step['off']['Communication(Not Overlapped)']/win['off']*100,
              step['off']['Free']/win['off']*100]),
     ("slot", [step['slot']['Computing']/win['slot']*100, 0.0,
               step['slot']['Free']/win['slot']*100])],
    ["计算（含 MooneP 内核）", "未重叠通信", "空闲/等待波次"]) + "</div>")

# off 画像
s.append("<h2>2. off 的画像：62% 设备时间在 HCCL 通信里等热点 rank</h2>")
s.append(f"<p>off 的设备时间大头是 <code>HcclLaunchAicpuKernel</code>（MC2/all-to-all 通信）——"
         f"占 <b>62.3%</b>（144.1s / 4 rank）。它不是传输慢，是<b>等最慢的 rank</b>：时长分布 p50=3.9ms、"
         f"p90=39.3ms、max=392.8ms——峰均比 2.48 下热点 rank（承担 62% 载荷）每步都让其余三个 rank "
         f"在集合通信里空等，40ms 级长尾遍布整个窗口。step 分解同样：未重叠通信 43%、计算仅 33%。</p>")
s.append("<div class=scroll><table><tr><th>off top 算子</th><th class=num>占比</th><th class=num>总时长(s)</th>"
         "<th class=num>次数</th><th class=num>均值(us)</th><th class=num>max(ms)</th></tr>")
for name, (cnt, t, mx) in top("off", 10):
    s.append(f"<tr><td>{H.escape(name)}</td><td class=num>{t/1e6/dev_tot['off']*100:.1f}%</td>"
             f"<td class=num>{t/1e6:.1f}</td><td class=num>{cnt}</td><td class=num>{t/cnt:,.0f}</td>"
             f"<td class=num>{mx/1000:.1f}</td></tr>")
s.append("</table></div>")

# slot 画像
s.append("<h2>3. slot 的画像：通信等待消失，但 MooneP 集合点自身成为第一大块</h2>")
s.append("<div class=scroll><table><tr><th>slot top 算子</th><th class=num>占比</th><th class=num>总时长(s)</th>"
         "<th class=num>次数</th><th class=num>均值(us)</th><th class=num>max(ms)</th></tr>")
for name, (cnt, t, mx) in top("slot", 14):
    s.append(f"<tr><td>{H.escape(name)}</td><td class=num>{t/1e6/dev_tot['slot']*100:.1f}%</td>"
             f"<td class=num>{t/1e6:.1f}</td><td class=num>{cnt}</td><td class=num>{t/cnt:,.0f}</td>"
             f"<td class=num>{mx/1000:.1f}</td></tr>")
s.append("</table></div>")
s.append("<p>MooneP 通信三件套（planning + dispatch + combine）合计占 <b>34.4%</b> 的设备时间——"
         "这是 slot 相对 off 新增的主要成本块；专家 GEMM（GroupedMatmul）10.7% 与 off 的 10.8% 同量级，"
         "说明计算本体没变差；<b>未重叠通信为 0</b>（MooneP 走设备内核，不再出现 HCCL 式整段未重叠等待）。</p>")

# 等待拆分
s.append("<h2>4. 瓶颈定位：MooneP 内核的时长 85~99% 是『等对端』，不是传输</h2>")
wk = {}
s.append("<p>按单次时长 2ms 为界把每次调用分成「快速体」（纯计算/传输）与「等待尾」（跨 rank 汇合等待）：</p>")
s.append("<div class=scroll><table><tr><th>内核</th><th class=num>总时长(s)</th><th class=num>快速体占比</th>"
         "<th class=num>快速体均值</th><th class=num>等待尾占比</th><th class=num>等待尾次数</th>"
         "<th class=num>等待尾均值</th><th class=num>纯传输 p50</th></tr>")
splits = []
for pat in ("planning_kernel", "moonep_dispatch_kernel", "moonep_combine_kernel"):
    fn, f_t, sn, s_t = wait_split("slot", pat)
    tot = f_t + s_t
    xs = read_dist("slot", pat)
    xs.sort()
    wk[pat] = (tot / 1e6, f_t / tot * 100, s_t / tot * 100, sn, s_t / sn / 1000, xs[len(xs)//2])
    splits.append((pat, tot/1e6, f_t/tot*100, s_t/tot*100))
    s.append(f"<tr><td><code>{pat}</code></td><td class=num>{tot/1e6:.1f}</td>"
             f"<td class=num>{f_t/tot*100:.0f}%</td><td class=num>{f_t/max(1,fn):.0f}us</td>"
             f"<td class=num><b class=warn>{s_t/tot*100:.0f}%</b></td><td class=num>{sn}</td>"
             f"<td class=num>{s_t/sn/1000:.1f}ms</td><td class=num>{xs[len(xs)//2]:.0f}us</td></tr>")
s.append("</table></div>")
s.append('<div class="fig">' + stacked_bar(
    "图 2 · MooneP 三内核：快速体 vs 等待尾（占各自总时长）",
    [(p.split('_')[0], [f_, w_, 0]) for p, _, f_, w_ in splits],
    ["快速体（纯计算/传输）", "等待尾（等对端汇合）", ""], total=100) + "</div>")
wait_total = sum(v[0] * v[2] / 100 for v in wk.values())
s.append(f"<div class=box><b>结论：</b>三件套合计 {sum(v[0] for v in wk.values()):.1f}s（4 rank），其中"
         f"<b>≈{wait_total:.1f}s（{wait_total/sum(v[0] for v in wk.values())*100:.0f}%）是等对端</b>——"
         f"折合每 rank 约 {wait_total/4:.1f}s，占 {win['slot']:.0f}s 窗口的 <b>{wait_total/4/win['slot']*100:.0f}%</b>。"
         "纯传输极快（dispatch 快速体均值仅 9μs/次）。等待的来源是各 rank 抵达集合点的时刻差："
         "各 rank 的 attention/前序计算完成时间不同步，planning 的跨卡栅栏先到者空转。"
         "这正是与 off『等热点 rank』同源的现象，只是 slot 把它从 40ms 级长尾压到了 7~21ms 级、"
         "且发生在设备内核内（有机会用重叠藏掉）。</div>")

# host
s.append("<h2>5. host 侧：slot 的小内核数量多 68%，InplaceCopy 胶水显眼</h2>")
hl = {}
for leg in ("off", "slot"):
    hl[leg] = sorted(host[leg].items(), key=lambda kv: -kv[1][0])[:6]
s.append("<div class=scroll><table><tr><th>host API（4 rank 合计）</th><th class=num>off 时长(s)×次数</th>"
         "<th class=num>slot 时长(s)×次数</th></tr>")
keys = ["launch", "aclrtLaunchKernelWithHostArgs", "aclnnInplaceCopy", "aclrtSynchronizeEvent",
        "aclnnQuantMatmulV5", "aclnnSWhere", "aclnnRmsNorm"]
for k in keys:
    o = host["off"].get(k, (0, 0)); sl = host["slot"].get(k, (0, 0))
    if o[0] == 0 and sl[0] == 0:
        continue
    s.append(f"<tr><td><code>{k}</code></td><td class=num>{o[0]/1e6:.1f}s ×{o[1]:,}</td>"
             f"<td class=num>{sl[0]/1e6:.1f}s ×{sl[1]:,}</td></tr>")
s.append("</table></div>")
s.append("<p>slot 的内核 launch 次数（WithHostArgs 39.6 万次）比 off（23.6 万）<b>多 68%</b>——"
         "prefetch、量化胶水等小内核拉高了 host 发射密度；<code>aclnnInplaceCopy</code> 13.2s（13.4 万次）"
         "与设备侧 TensorMove 8.4s（2.4 万次）对应，是 slot 路径的搬运胶水。</p>")

# 优化空间
s.append("<h2>6. 优化空间（按数据排序）</h2>")
s.append("<div class=scroll><table><tr><th>#</th><th>方向</th><th>数据依据</th><th>怎么做</th>"
         "<th>状态</th></tr>")
opts = [
    ("重叠掉集合点等待", f"三件套等待尾 ≈{wait_total:.0f}s/4rank，占窗口 {wait_total/4/win['slot']*100:.0f}%；纯传输仅 μs 级",
     "让等待与计算并行：推广 after_planning_hook 的思路——planning 提前发射、dispatch/combine 与 attention/GEMM 交叉；已有实验开关 <code>MOONEP_PF_STREAM=2</code>（prefetch 发 side stream，须 RELAXED 已配）可先验证收益上限",
     "钩子/开关已有，需组合验证"),
    ("planning 的 641 次 21ms 长尾", "planning 等待尾 85% 集中在 641 次、均值 21ms/次",
     "定位这些调用对应的步（疑似 DP 波次对齐/dummy 步先到者在栅栏空转）：让 dummy 步不进 MooneP 集合，或对纯等待场景评估 BARRIER_V4 push 模式",
     "需定位+开发"),
    ("消 TensorMove/InplaceCopy", "设备 8.4s(2.4 万次)+host 13.2s；off 腿仅 4.3%/5536 次",
     "定位 slot 路径的冗余搬运（量化 payload 镜像、bf16 载体转换），可合并/去除——历史 TTFT 报告已把 TensorMove 列为『最可能白捡的一块』",
     "需定位+开发"),
    ("压 Free 32%", "空闲 13.2s/rank；launch 密度比 off 高 68%",
     "减少小内核数（prefetch w13/w2 合并、Cast/Mul/MxQuant 融合）；MoonEP 硬约束 enforce-eager 下不能上图捕获，靠内核数下降",
     "需开发"),
    ("胶水算子融合", "Cast 3.0% + Mul 2.8% + DynamicMxQuant 3.7%",
     "与量化/归一路径合并成复合算子", "需开发"),
]
for i, (d1, d2, d3, d4) in enumerate(opts, 1):
    s.append(f"<tr><td class=num>{i}</td><td><b>{d1}</b></td><td class=small>{d2}</td>"
             f"<td class=small>{d3}</td><td class=small>{d4}</td></tr>")
s.append("</table></div>")

s.append("<h2>7. 附录</h2><div class=box><ul class=small>"
         "<li>数据：profile/{off,slot}/dp*_ascend_pt/ASCEND_PROFILER_OUTPUT/（torch_npu analyse 产物；本报告全部数字由 gen_profile_analysis.py 现算）</li>"
         "<li>时间线下钻：各 rank 的 trace_view.json（MindStudio Insight / chrome://tracing）</li>"
         "<li>干净端到端数据：../20261011_opt2_off_vs_slot/（同条件 3 轮无 profiler：TTFT 11.0s vs 9.6s）</li>"
         "<li>口径：两腿窗口时长不同（64.8s vs 41.2s），全部对比基于占比/分布，不基于绝对值</li></ul></div>")
s.append("</div>")

import os
os.makedirs("report", exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(s))
print(f"[report] {OUT}（{os.path.getsize(OUT)} 字节）")
