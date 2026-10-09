#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""real-T 重跑报告生成器：读 /tmp/mc2_probe/grid/rt_* 的 bench json，
输出三腿对照（off / slot[real-T] / slot0[旧行为]）+ 与 H16 旧基线的漂移对照。

用法: python3 gen_rt_report.py [--grid /tmp/mc2_probe/grid] [--out <前缀>]
输出:
  <out>.md   摘要表（每 mbt × 形状 × 路由一行，四变化量列）
  <out>.html 详情报告，与 H16 旧报告（moonep_off_vs_slot_min16_bench_data_report.html）
             同款式：同一套 CSS/class、toolbar、details.cfg 前言块、每路由详情与逐轮明细表，
             另加「与 H16 旧基线对照」块；列数从两腿扩到三腿 + 两个变化量组。
口径与 H16 一致：每形状 3 轮、轮 1 剔除、统计取轮 2/3 中位；
变化量四个：(slot−off)/off、(slot−slot0)/slot0、(off新−off旧)/off旧、(slot新−slot旧)/slot旧。
轮 2/3 相对中位偏差 >15% 标「模态不稳」并在明细表下给出备注。
"""
import argparse
import json
import os
import re
import statistics as st
import html as H

GRID = "/tmp/mc2_probe/grid"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "moonep_off_vs_slot_real_t_bench_data_report")
RUN_RE = "rt_mbt{mbt}_min16_{route}_{mode}"
MBTS = ("8192", "16384")
ROUTES = ("uniform", "4hot", "3hot1r", "4hot1r")
LEGS = ("off", "slot", "slot0")
LEG_TITLE = {"off": "off（MC2）", "slot": "slot（real-T）", "slot0": "slot0（MOONEP_REAL_T=0 消融）"}
LEG_HDR = {"off": "moonep off", "slot": "moonep slot（real-T）", "slot0": "slot0（MOONEP_REAL_T=0）"}
SHAPES = ("L8k_n32", "L16k_n32")
SEED_BASE = {"L8k_n32": 9, "L16k_n32": 12}
ROUTE_INFO = {  # 路由名 → (档案文件, s = max rank / 均值, rank0-3 home 载荷)
    "uniform": ("sim_route_uniform.txt", "1", ("25%", "25%", "25%", "25%")),
    "4hot": ("sim_route_4hot.txt", "1.6614", ("41.5344%", "41.5344%", "8.4656%", "8.4656%")),
    "3hot1r": ("sim_route_3hot_1rank.txt", "2.4822", ("62.0554%", "12.6482%", "12.6482%", "12.6482%")),
    "4hot1r": ("sim_route_4hot_1rank.txt", "2.9841", ("74.6033%", "8.4656%", "8.4656%", "8.4656%")),
}

# 明细表字段行：(标签, 单位, bench json 键, 方向[lower=越小越好 / higher=越大越好 / None=不上色], 格式)
FIELDS = [
    ("TTFT p50", "ms", "p50_ms", "lower", "{:.2f}"),
    ("TTFT 平均", "ms", "mean_ttft_ms", "lower", "{:.2f}"),
    ("TTFT 中位", "ms", "median_ttft_ms", "lower", "{:.2f}"),
    ("TTFT 标准差", "ms", "std_ttft_ms", "lower", "{:.2f}"),
    ("TTFT p90", "ms", "p90_ttft_ms", "lower", "{:.2f}"),
    ("TTFT p99", "ms", "p99_ttft_ms", "lower", "{:.2f}"),
    ("请求吞吐", "req/s", "request_throughput", "higher", "{:.4f}"),
    ("输出 token 吞吐", "token/s", "output_throughput", "higher", "{:.2f}"),
    ("总 token 吞吐", "token/s", "total_token_throughput", "higher", "{:.2f}"),
    ("duration", "s", "duration", "lower", "{:.3f}"),
    ("max_concurrent_requests", "个请求", "max_conc", None, "{:.0f}"),
    ("completed", "个请求", "completed", None, "{:.0f}"),
    ("failed", "个请求", "failed", None, "{:.0f}"),
]

# H16 基线（L16k_n32，轮 2/3 中位 TTFT p50，秒）：moonep_off_vs_slot_min16_bench_data_report.html
BASELINE_L16K_P50 = {  # (mbt, route, leg)
    ("8192", "uniform", "off"): 9.338, ("8192", "uniform", "slot"): 14.874,
    ("8192", "4hot", "off"): 10.607, ("8192", "4hot", "slot"): 15.657,
    ("8192", "3hot1r", "off"): 12.212, ("8192", "3hot1r", "slot"): 15.622,
    ("8192", "4hot1r", "off"): 13.439, ("8192", "4hot1r", "slot"): 15.506,
    ("16384", "uniform", "off"): 9.418, ("16384", "uniform", "slot"): 17.895,
    ("16384", "4hot", "off"): 10.672, ("16384", "4hot", "slot"): 20.160,
    ("16384", "3hot1r", "off"): 12.340, ("16384", "3hot1r", "slot"): 20.047,
    ("16384", "4hot1r", "off"): 13.519, ("16384", "4hot1r", "slot"): 19.775,
}

# H16 基线（L8k_n32，同来源；4hot1r 旧 slot 低于旧 off 即「8k 热点路由有收益」那两个格）
BASELINE_L8K_P50 = {  # (mbt, route, leg)
    ("8192", "uniform", "off"): 4.532, ("8192", "uniform", "slot"): 5.027,
    ("8192", "4hot", "off"): 5.125, ("8192", "4hot", "slot"): 5.201,
    ("8192", "3hot1r", "off"): 5.999, ("8192", "3hot1r", "slot"): 5.129,
    ("8192", "4hot1r", "off"): 6.557, ("8192", "4hot1r", "slot"): 5.230,
    ("16384", "uniform", "off"): 4.554, ("16384", "uniform", "slot"): 8.866,
    ("16384", "4hot", "off"): 5.218, ("16384", "4hot", "slot"): 8.812,
    ("16384", "3hot1r", "off"): 6.122, ("16384", "3hot1r", "slot"): 8.790,
    ("16384", "4hot1r", "off"): 6.695, ("16384", "4hot1r", "slot"): 8.633,
}
BASELINE = {"L16k_n32": BASELINE_L16K_P50, "L8k_n32": BASELINE_L8K_P50}

DATA_QUALITY_NOTE = (
    "旧轮 `L16k_n32` uniform 格的旧 slot 基线 17.895s 是孤立低值——该旧运行轮 1 = 20.162s、"
    "轮 2/3 才降到 17.9s；本轮 slot0 同格 20.250s，且其余 15 格「旧 slot vs 本轮 slot0」都在 ±3.2% 内"
    "（代码核对：T=S 时各内核改动均不生效，REAL_T=0 回退分支与旧路径行为等价）。"
    "该格 `slot vs 旧 slot`（+0.4%）与 `slot vs slot0`（−11.2%）两个口径矛盾，"
    "读该格以同轮同环境的 `slot vs slot0` 为准。"
)


def md_code(text):
    """md 反引号 → html <code>"""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", text)


def load_rec(run, shape, r):
    p = os.path.join(GRID, run, f"{shape}_c{r}.json")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        d = json.load(fh)
    rec = {key: d.get(key) for _, _, key, _, _ in FIELDS}
    rec["p50_ms"] = d.get("p50_ttft_ms") or d.get("median_ttft_ms")
    return rec


def rel_spread(vals):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2 or st.fmean(vals) == 0:
        return 0.0
    return max(abs(v - st.fmean(vals)) for v in vals) / st.fmean(vals)


def collect():
    """DATA[mbt][shape][route][leg] = {recs: {1|2|3: rec}, med: {key: 轮 2/3 均值}, unstable: bool}"""
    D = {mbt: {shape: {} for shape in SHAPES} for mbt in MBTS}
    for mbt in MBTS:
        for shape in SHAPES:
            for route in ROUTES:
                cell = {}
                for leg in LEGS:
                    run = RUN_RE.format(mbt=mbt, route=route, mode=leg)
                    recs = {r: load_rec(run, shape, r) for r in (1, 2, 3)}
                    rs = [recs[r] for r in (2, 3) if recs[r]]
                    med = None
                    if rs:
                        keys = [k for k in rs[0] if isinstance(rs[0][k], (int, float))]
                        med = {k: st.fmean(r[k] for r in rs if r.get(k) is not None) for k in keys}
                    unstable = (rel_spread([recs[r].get("p50_ms") for r in (2, 3) if recs[r]]) > 0.15
                                or rel_spread([recs[r].get("duration") for r in (2, 3) if recs[r]]) > 0.15)
                    cell[leg] = {"recs": recs, "med": med, "unstable": unstable}
                D[mbt][shape][route] = cell
    return D


def delta(new, old):
    if new is None or old in (None, 0):
        return None
    return (new - old) / old * 100.0


def pct(v):
    return "—" if v is None else f"{v:+.1f}%"


def fnum(v, unit="", fmt=None):
    if v is None:
        return "—"
    if fmt:
        return fmt.format(v)
    return f"{v:.3f}{unit}" if abs(v) < 100 else f"{v:.1f}{unit}"


# ---------------------------------------------------------------- md 摘要表

def render_md(D):
    md = ["# moonep off vs slot（real-T 优化后）重跑报告", "",
          f"- 数据源：`{GRID}/rt_mbt*`（2 mbt × 4 路由 × 3 腿 × 2 形状 × 3 轮）",
          "- 口径与 H16 轮一致：轮 1 剔除、统计取轮 2/3 中位、seed=形状序号×1000+轮次",
          "  （L8k=9001-9003、L16k=12001-12003，与旧运行同 seed）；`slot0` = `MOONEP_REAL_T=0` 消融腿。",
          "- 四个变化量：`(slot−off)/off`（新结论）、`(slot−slot0)/slot0`（real-T 归因）、"
          "`(off新−off旧)/off旧`（环境漂移）、`(slot新−slot旧)/slot旧`（优化总效果，含环境漂移；"
          "两形状基线均硬编码自 H16 报告）。", ""]
    for mbt in MBTS:
        md += [f"## mbt={mbt}（MOONEP_MIN_DISPATCH_TOKENS=16）", ""]
        for shape in SHAPES:
            md += [f"### {shape}", "",
                   "| 路由 | off p50(s) | slot p50(s) | slot0 p50(s) | slot vs off | slot vs slot0 | off vs 旧 | "
                   "slot vs 旧 slot | off req/s | slot req/s | slot0 req/s | 备注 |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for route in ROUTES:
                cell = D[mbt][shape][route]
                notes = []
                for leg in LEGS:
                    if cell[leg]["med"] is None:
                        notes.append(f"{leg} 缺数据")
                    elif cell[leg]["unstable"]:
                        notes.append(f"{leg} 模态不稳")
                g = lambda leg, k: (cell[leg]["med"] or {}).get(k)
                base = BASELINE[shape].get((mbt, route, "off"))
                base_slot = BASELINE[shape].get((mbt, route, "slot"))
                row = [route]
                for leg in LEGS:
                    v = g(leg, "p50_ms")
                    row.append(fnum(v / 1000 if v else None))
                row += [
                    pct(delta(g("slot", "p50_ms"), g("off", "p50_ms"))),
                    pct(delta(g("slot", "p50_ms"), g("slot0", "p50_ms"))),
                    pct(delta(g("off", "p50_ms"), base * 1000 if base else None)) if base else "—",
                    pct(delta(g("slot", "p50_ms"), base_slot * 1000 if base_slot else None)) if base_slot else "—",
                    fnum(g("off", "request_throughput")),
                    fnum(g("slot", "request_throughput")),
                    fnum(g("slot0", "request_throughput")),
                    "；".join(notes) if notes else "",
                ]
                md.append("| " + " | ".join(str(x) for x in row) + " |")
            md.append("")

        # 旧基线对照（H16，两形状）
        md += ["#### 旧基线（H16，轮 2/3 中位 p50，秒）", "",
               "| 形状 | 路由 | 旧 off | 旧 slot |", "|---|---|---|---|"]
        for shape in SHAPES:
            for route in ROUTES:
                md.append(f"| {shape} | {route} | {BASELINE[shape].get((mbt, route, 'off'), '—')} | "
                          f"{BASELINE[shape].get((mbt, route, 'slot'), '—')} |")
        md.append("")
        if mbt == "16384":
            md += ["注（数据质量）：" + DATA_QUALITY_NOTE, ""]
    return "\n".join(md)


# ---------------------------------------------------------------- html（H16 同款式）

CSS = """<style>
  :root { --line:#d8dee4; --head:#f3f5f7; --dim:#8b949e; --good:#0a7d33; --bad:#c0392b; }
  body { font-family: -apple-system, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
         margin: 0 auto; max-width: 1900px; padding: 24px 28px 80px; color:#1f2328; line-height:1.6; }
  h1 { font-size: 26px; margin: 0 0 8px; }
  h2 { font-size: 21px; margin-top: 28px; }
  code { background:#f3f5f7; padding:1px 4px; border-radius:3px; font-size:13px; }
  .lead { background:#f6f8fa; border:1px solid var(--line); border-radius:6px; padding:12px 16px; margin:14px 0; }
  .lead p { margin:6px 0; }
  .toolbar { position: sticky; top:0; background:#fff; padding:8px 0; border-bottom:1px solid var(--line);
             z-index:5; margin-bottom:10px; }
  button { font: inherit; padding:4px 12px; margin-right:8px; border:1px solid var(--line);
           background:#fff; border-radius:5px; cursor:pointer; }
  button:hover { background:#f3f5f7; }
  details { margin: 6px 0; }
  details.cfg > summary { font-size:19px; font-weight:600; padding:8px 10px; background:#eef2f6;
                          border-left:5px solid #4a6fa5; border-radius:4px; cursor:pointer; }
  details.skew > summary { font-size:16px; font-weight:600; padding:6px 10px; background:#f7f9fb;
                           border-left:5px solid #9db3cc; border-radius:4px; cursor:pointer; margin-top:8px; }
  details.shape > summary { font-size:14px; padding:4px 10px; border-left:3px solid #cfd8e3;
                            cursor:pointer; margin-top:4px; }
  details.shape[open] > summary { font-weight:600; }
  .meta { color:#4a5568; font-size:13px; margin:6px 0 10px 14px; }
  .meta b { color:#1f2328; }
  .note { color:#7a4d00; font-size:13px; margin:6px 0 10px 14px; }
  table { border-collapse: collapse; margin: 8px 0 16px 14px; font-size:13px;
          font-variant-numeric: tabular-nums; }
  th, td { border:1px solid var(--line); padding:3px 8px; text-align:left; white-space:nowrap; }
  th { background: var(--head); font-weight:600; }
  td.num, th.num { text-align:right; }
  th.grp { text-align:center; background:#e9eef4; }
  tr.warm td { color: var(--dim); }
  .better { color: var(--good); }
  .worse { color: var(--bad); }
  .legend span { margin-right:16px; }
  .small { font-size:12px; color:#57606a; }
  td.metric { font-family: ui-monospace, Menlo, Consolas, monospace; font-size:12px; }
</style>"""

TOOLBAR = ('<div class="toolbar"><button onclick="document.querySelectorAll(\'details\').forEach(d=>d.open=true)">'
           '全部展开</button><button onclick="document.querySelectorAll(\'details\').forEach(d=>d.open=false)">'
           '全部折叠</button><button onclick="document.querySelectorAll(\'details.shape\').forEach(d=>d.open=true)">'
           '展开 bench 输入</button><span class="legend small"><span><b class="better">绿色</b> = slot 更好</span>'
           '<span><b class="worse">红色</b> = slot 更差</span><span>灰字 = 轮次 1（热身，不参与中位）</span></span></div>')


def cls_color(d, direction):
    """按「变小更好/变大更好」给变化值上色；None/0/方向未知不上色"""
    if d is None or direction is None or d == 0:
        return "num"
    good = d < 0 if direction == "lower" else d > 0
    return "num better" if good else "num worse"


def summary_table(cells):
    """cells: {shape: D[mbt][shape][route]}，输出每形状一行的摘要表"""
    cols = [("TTFT p50", "ms", "p50_ms", "lower", "{:.2f}"),
            ("请求吞吐", "req/s", "request_throughput", "higher", "{:.4f}"),
            ("总 token 吞吐", "token/s", "total_token_throughput", "higher", "{:.2f}")]
    out = ["<table><tr><th rowspan=2>bench 输入</th>"]
    for title, unit, _, _, _ in cols:
        out.append(f'<th class="grp" colspan=5>{title}（{unit}，轮次 2、3 中位）</th>')
    out.append("</tr><tr>")
    for _ in cols:
        out.append('<th class="num">off</th><th class="num">slot</th><th class="num">slot0</th>'
                   '<th class="num">slot vs off</th><th class="num">slot vs slot0</th>')
    out.append("</tr>")
    for shape in SHAPES:
        cell = cells[shape]
        out.append(f"<tr><td>{shape}</td>")
        for _, _, key, direction, fmt in cols:
            m = {leg: (cell[leg]["med"] or {}) for leg in LEGS}
            for leg in LEGS:
                out.append(f'<td class="num">{fnum(m[leg].get(key), fmt=fmt)}</td>')
            out.append(f'<td class="{cls_color(delta(m["slot"].get(key), m["off"].get(key)), direction)}">'
                       f'{pct(delta(m["slot"].get(key), m["off"].get(key)))}</td>')
            out.append(f'<td class="{cls_color(delta(m["slot"].get(key), m["slot0"].get(key)), direction)}">'
                       f'{pct(delta(m["slot"].get(key), m["slot0"].get(key)))}</td>')
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def detail_table(cell, shape):
    out = ['<table><tr><th rowspan=2>字段</th><th rowspan=2>单位</th>']
    for leg in LEGS:
        out.append(f'<th class="grp" colspan=3>{LEG_HDR[leg]}</th>')
    for ref in ("off", "slot0"):
        out.append(f'<th class="grp" colspan=4>变化（slot 相对 {ref}）</th>')
    out.append("</tr><tr>")
    for _ in LEGS:
        out.append('<th class="num">轮次 1</th><th class="num">轮次 2</th><th class="num">轮次 3</th>')
    for _ in ("off", "slot0"):
        out.append('<th class="num">轮次 1</th><th class="num">轮次 2</th><th class="num">轮次 3</th>'
                   '<th class="num">中位</th>')
    out.append("</tr>")
    for label, unit, key, direction, fmt in FIELDS:
        out.append(f'<tr><td class="metric">{label}</td><td class="small">{unit}</td>')
        for leg in LEGS:
            for r in (1, 2, 3):
                rec = cell[leg]["recs"].get(r) or {}
                cls = "num warm" if r == 1 else "num"
                out.append(f'<td class="{cls}">{fnum(rec.get(key), fmt=fmt)}</td>')
        for ref in ("off", "slot0"):
            for r in (1, 2, 3):
                d = delta((cell["slot"]["recs"].get(r) or {}).get(key), (cell[ref]["recs"].get(r) or {}).get(key))
                cls = "num" if r == 1 else cls_color(d, direction)
                out.append(f'<td class="{cls}">{pct(d)}</td>')
            d = delta((cell["slot"]["med"] or {}).get(key), (cell[ref]["med"] or {}).get(key))
            out.append(f'<td class="{cls_color(d, direction)}">{pct(d)}</td>')
        out.append("</tr>")
    out.append("</table>")
    base = SEED_BASE[shape]
    out.append(f'<div class="meta">seed（三条腿每个轮次相同，所以是同一批 prompt）：'
               f'轮次 1 = {base * 1000 + 1}；轮次 2 = {base * 1000 + 2}；轮次 3 = {base * 1000 + 3}</div>')
    bad = [LEG_TITLE[l] for l in LEGS if cell[l]["unstable"]]
    if bad:
        out.append('<div class="note">模态不稳：' + "、".join(bad) + "（轮次 2/3 相差 >15%），该格结论谨慎。</div>")
    return "".join(out)


def cfg_params():
    return ('<details class="cfg" open><summary>两个服务配置的参数、capacity 池、KV 池（实测值取服务日志）</summary>'
            '<table><tr><th>参数</th><th>服务配置 1（<code>max_num_batched_tokens</code>=8192）</th>'
            '<th>服务配置 2（16384）</th></tr>'
            '<tr><td><code>max_num_batched_tokens</code></td><td>8192</td><td>16384</td></tr>'
            '<tr><td><code>MOONEP_MIN_DISPATCH_TOKENS</code></td><td>16</td><td>16</td></tr>'
            '<tr><td><code>max_num_seqs</code></td><td>16</td><td>16</td></tr>'
            '<tr><td><code>max_model_len</code></td><td>20480</td><td>20480</td></tr>'
            '<tr><td><code>kv_cache_memory_bytes</code>（固定 9 GiB）</td><td>9663676416</td><td>9663676416</td></tr>'
            '<tr><td><code>enable_chunked_prefill</code></td><td>True</td><td>True</td></tr>'
            '<tr><td><code>enforce_eager</code></td><td>True</td><td>True</td></tr>'
            '<tr><td><code>data_parallel_size</code> / <code>enable_expert_parallel</code></td><td>4 / True</td>'
            '<td>4 / True</td></tr>'
            '<tr><td>服务 <code>--seed</code></td><td>1024</td><td>1024</td></tr>'
            '<tr><td>实测 capacity 池（slot 运行的服务日志）</td>'
            '<td>16,32,64,128,256,512,1024,2048,4096,8192</td>'
            '<td>16,32,64,128,256,512,1024,2048,4096,8192,16384</td></tr>'
            '<tr><td>实测每 rank <code>GPU KV cache size</code>（tokens）</td><td>29133</td><td>23436</td></tr>'
            '<tr><td>bench 用的卡 / 端口 / 服务拉起</td><td>0-3（同 UB 域）/ 8444 / 每条腿各一次</td><td>同左</td></tr>'
            '</table>'
            '<p class="small">注意：<b>两个配置的 KV 池不一样</b>（配置 1 每 rank 29,133 tokens、配置 2 每 rank 23,436 '
            'tokens），因为 <code>max_num_batched_tokens</code> 影响每步激活的 slot 缓冲占用；两个配置之间不要直接比绝对耗时。'
            '本批服务参数与 H16 轮逐项相同，实测 capacity 池与 KV cache size 也与 H16 报告一致。</p></details>')


def cfg_legs():
    return ('<details class="cfg" open><summary>三条腿的差异（同一份服务脚本，只差 moonep 开关与 <code>MOONEP_REAL_T</code>）</summary>'
            '<table><tr><th>腿</th><th><code>additional_config</code>（enable_moonep）</th>'
            '<th><code>MOONEP_REAL_T</code></th><th>说明</th></tr>'
            '<tr><td>moonep off</td><td>{&quot;enable_moonep&quot;: false}</td><td>不适用</td>'
            '<td>MC2 通信，既有基线</td></tr>'
            '<tr><td>moonep slot（real-T，默认）</td><td>{&quot;enable_moonep&quot;: true}</td><td>1（默认）</td>'
            '<td>本次优化：dispatch/combine/均衡只按真实 T（pad 不参与计算与均衡）</td></tr>'
            '<tr><td>slot0（消融）</td><td>{&quot;enable_moonep&quot;: true}</td><td>0</td>'
            '<td>回退旧路径（pad 全参与），行为与旧 slot 等价——用来把 real-T 改动本身的效果与环境漂移分开</td></tr>'
            '</table>'
            '<p class="small">每条腿各拉起一次服务；同一档路由下 off → slot → slot0 相邻开跑，机器漂移对三条腿一致。'
            '<code>MOONEP_MIN_DISPATCH_TOKENS</code>=16 只影响 capacity 池起点，三条腿相同。</p></details>')


def cfg_routes():
    rows = []
    for route in ROUTES:
        fname, s, payloads = ROUTE_INFO[route]
        rows.append(f'<tr><td>{route}</td><td class="metric">{fname}</td>'
                    + "".join(f'<td class="num">{p}</td>' for p in payloads)
                    + f'<td class="num">{s}</td></tr>')
    return ('<details class="cfg" open><summary>四档路由仿真分布（<code>VLLM_ASCEND_MOE_SIM_ROUTE</code> 档案）</summary>'
            '<table><tr><th>倾斜名</th><th>档案</th><th>rank0 home 载荷</th><th>rank1</th><th>rank2</th>'
            '<th>rank3</th><th>s = max rank / 均值</th></tr>' + "".join(rows) + '</table>'
            '<p class="small">模型 E=256、topk=6、DP4 / EP on → 每个 rank 拥有 64 个专家（rank r 拿 '
            '<code>[r*64, (r+1)*64)</code>）。档案里的数 = 该专家占本轮全部 route（T×topk）的百分比，总和 100；'
            '每 rank 载荷 = 它名下 64 个专家的百分比之和，四者相加 = 100。四档档案在 '
            '<code>scripts/dsv4flash_service/sim_route_*.txt</code>，与 H16 轮用的是同一批档案（s 与载荷逐位相同）。'
            '</p></details>')


def cfg_bench():
    return ('<details class="cfg" open><summary>bench 输入（两个形状）与 seed 规则</summary>'
            '<table><tr><th>形状</th><th>形状集里的下标</th><th><code>num_prompts</code></th>'
            '<th><code>max_concurrency</code></th><th>实际并发</th><th><code>random_input_len</code></th>'
            '<th><code>random_output_len</code></th><th><code>request_rate</code></th><th>seed</th></tr>'
            '<tr><td>L8k_n32</td><td class="num">1</td><td class="num">32</td><td class="num">16</td>'
            '<td class="num">16</td><td class="num">8192</td><td class="num">1</td><td>inf</td>'
            '<td class="metric">9 × 1000 + 轮次</td></tr>'
            '<tr><td>L16k_n32</td><td class="num">2</td><td class="num">32</td><td class="num">16</td>'
            '<td class="num">16</td><td class="num">16384</td><td class="num">1</td><td>inf</td>'
            '<td class="metric">12 × 1000 + 轮次</td></tr></table>'
            '<p class="small">每条运行开始时还会跑一次热身 bench（<code>num_prompts</code>=2、'
            '<code>max_concurrency</code>=2、<code>random_input_len</code>=1024、seed 999、轮次记为 0），'
            '让表缓存与首次 dispatch 先过一遍，它不在下面的表里。seed 与 H16 轮相同（同一批 prompt）。</p></details>')


def cfg_scope():
    return ('<details class="cfg" open><summary>统计口径（怎么读下面的表）</summary><ul>'
            '<li><b>轮次</b>：同一个形状在同一个服务进程上重复跑的第几遍。每轮换一批 prompt'
            '（<code>seed</code> = 形状的 seed 序号 × 1000 + 轮次），换 seed 是为了避免上一轮的 prompt 被 '
            '<code>prefix caching</code> 命中而把 TTFT 压低。</li>'
            '<li><b>轮次 1 当热身剔掉</b>：它是服务刚拉起后的第一遍，历史上偏慢；结论只看轮次 2、3。</li>'
            '<li><b>四个变化量</b>，算式一律 <code>(新 − 参照) / 参照 × 100%</code>：'
            '<code>(slot − off)/off</code> = 本轮主结论（slot 相对 off）；'
            '<code>(slot − slot0)/slot0</code> = real-T 归因（同轮同环境、与旧 slot 等价的回退腿）；'
            '<code>(off 本轮 − off 旧)/off 旧</code> = 环境漂移（旧值取 H16 报告）；'
            '<code>(slot 本轮 − slot 旧)/slot 旧</code> = 优化总效果（含环境漂移）。</li>'
            '<li><b>符号</b>：TTFT / duration 这一类越小越好 → 负号 = 更快 = 好（绿）、正号 = 更慢 = 差（红）；'
            '吞吐这一类越大越好 → 正号 = 更快 = 好（绿）；数值两侧同向比较（对吞吐来说负号 = 吞吐掉 = 红）。</li>'
            '<li><b>变化</b>逐轮各给一个（轮次 1 的值只是留档、不上色，不参与结论），另外每个变化量给一个「中位」列'
            '——先把两边各自取轮次 2、3 中位，再算变化。两种模式没有绝对好坏时'
            '（<code>completed</code> / <code>failed</code> / <code>max_concurrent_requests</code>）不上色。</li>'
            '<li><b>模态不稳</b>：若同一模式保留的两轮之间相差超过 15%，该形状的表下会给出备注。</li>'
            '<li>这批 bench <code>random_output_len</code>=1，量与 E2EL 只差一个输出 token；报告只列 TTFT 家族与吞吐，'
            'E2EL 家族在原始 json 里仍有。也没有逐请求记录（bench 用默认 <code>--save-detailed=False</code>）。</li>'
            '</ul></details>')


def cfg_baseline(D):
    rows = []
    for mbt in MBTS:
        for shape in SHAPES:
            for route in ROUTES:
                cell = D[mbt][shape][route]
                off = (cell["off"]["med"] or {}).get("p50_ms")
                slot = (cell["slot"]["med"] or {}).get("p50_ms")
                bo, bs = BASELINE[shape].get((mbt, route, "off")), BASELINE[shape].get((mbt, route, "slot"))
                d_off, d_slot = delta(off, bo * 1000 if bo else None), delta(slot, bs * 1000 if bs else None)
                rows.append(
                    f'<tr><td>{mbt}</td><td>{shape}</td><td>{route}</td>'
                    f'<td class="num">{fnum(bo, fmt="{:.3f}")}</td>'
                    f'<td class="num">{fnum(off / 1000 if off else None, fmt="{:.3f}")}</td>'
                    f'<td class="{cls_color(d_off, None)}">{pct(d_off)}</td>'
                    f'<td class="num">{fnum(bs, fmt="{:.3f}")}</td>'
                    f'<td class="num">{fnum(slot / 1000 if slot else None, fmt="{:.3f}")}</td>'
                    f'<td class="{cls_color(d_slot, "lower")}">{pct(d_slot)}</td></tr>')
    return ('<details class="cfg" open><summary>与 H16 旧基线对照（轮 2/3 中位 TTFT p50，秒）</summary>'
            '<table><tr><th>mbt</th><th>形状</th><th>路由</th>'
            '<th class="num">旧 off</th><th class="num">本轮 off</th><th class="num">off vs 旧</th>'
            '<th class="num">旧 slot</th><th class="num">本轮 slot</th><th class="num">slot vs 旧 slot</th></tr>'
            + "".join(rows) + '</table>'
            '<p class="small"><code>off vs 旧</code> 是环境漂移（本轮 off 相对 H16 旧 off，L16k ≤2%、L8k −2~−6%，'
            '本轮整体略快，同轮对比列不受影响）；<code>slot vs 旧 slot</code> 是优化总效果（含环境漂移）。'
            '旧值硬编码自 H16 报告 <code>moonep_off_vs_slot_min16_bench_data_report.html</code> 的轮 2/3 中位。</p>'
            f'<div class="note">数据质量：{md_code(DATA_QUALITY_NOTE)}</div></details>')


def render_html(D, md_note_only=False):
    P = ['<html><head><meta charset="utf-8"><title>moonep off vs slot（real-T 优化后）重跑报告</title>', CSS,
         "</head><body>"]
    P.append("<h1>moonep off vs slot（real-T 优化后）：MOONEP_MIN_DISPATCH_TOKENS=16 × "
             "max_num_batched_tokens 8192 / 16384</h1>")
    P.append('<div class="lead">'
             '<p>这批数据是 <b>24 次运行、168 次 vllm bench</b>：两个服务配置（只差 '
             '<code>max_num_batched_tokens</code>）× 四档路由仿真分布（uniform / 4hot / 3hot1r / 4hot1r）× '
             '三条腿（moonep off / moonep slot（real-T）/ slot0）。<code>slot0</code> 是本次改动的消融腿：'
             '<code>MOONEP_REAL_T=0</code>，回退旧路径（pad 全参与计算与均衡、与旧 slot 行为等价），'
             '用来把「real-T 改动本身的效果」与「机器/环境漂移」分开。</p>'
             '<p>bench 输入两个：<code>L8k_n32</code>（<code>num_prompts</code>=32、<code>max_concurrency</code>=16、'
             '<code>random_input_len</code>=8192）与 <code>L16k_n32</code>（32 / 16 / 16384），'
             '<code>random_output_len</code>=1、<code>request_rate</code>=inf；每个形状跑 3 轮。</p>'
             '<p>表格里的数就是 vllm bench 写出的原始字段值；<b>变化</b>是「其它实验条件都相同」的配对结果：'
             '同一份服务配置、同一档路由仿真分布、同一个 bench 输入、同一个轮次（seed 相同 = 同一批 prompt）。'
             '算式一律 <code>(新 − 参照) / 参照 × 100%</code>，本报告给四个变化量：'
             '<code>(slot − off)/off</code>（主结论）、<code>(slot − slot0)/slot0</code>（real-T 归因）、'
             '<code>(off 本轮 − off 旧)/off 旧</code>（环境漂移）、<code>(slot 本轮 − slot 旧)/slot 旧</code>'
             '（优化总效果）；符号怎么读见「统计口径」。</p>'
             '<p class="small">原始数据（vllm bench json 全字段）在 <code>/tmp/mc2_probe/grid/&lt;运行名&gt;/</code>；'
             '本报告由 <code>scripts/dsv4flash_service/gen_rt_report.py</code> 生成；H16 旧报告'
             '（<code>moonep_off_vs_slot_min16_bench_data_report.html</code>）与更早的报告都是原文件，没有被覆盖。</p>'
             "</div>")
    P.append(TOOLBAR)
    P.append("<h2>实验条件</h2>")
    for block in (cfg_params(), cfg_legs(), cfg_routes(), cfg_bench(), cfg_scope(), cfg_baseline(D)):
        P.append(block)

    for mbt in MBTS:
        P.append(f'<h2 id="mbt{mbt}">服务配置 {"1" if mbt == "8192" else "2"}：'
                 f'<code>max_num_batched_tokens</code>={mbt}、<code>MOONEP_MIN_DISPATCH_TOKENS</code>=16</h2>')
        P.append('<p class="small">本配置的 12 次运行：4 档路由仿真分布 × 三腿'
                 '（off / slot / slot0），每档服务拉起一次（三腿相邻开跑）。</p>')
        for route in ROUTES:
            fname, s, payloads = ROUTE_INFO[route]
            P.append(f'<details class="skew" open><summary>路由仿真分布 {route}（档案 <code>{fname}</code>，'
                     f's = {s}）</summary>')
            rn = {leg: RUN_RE.format(mbt=mbt, route=route, mode=leg) for leg in LEGS}
            P.append(f'<div class="meta">moonep off 运行：<b>{rn["off"]}</b>；moonep slot 运行：'
                     f'<b>{rn["slot"]}</b>；slot0 运行：<b>{rn["slot0"]}</b>（start_service_mode_dp4.sh，'
                     f'三腿各拉起一次）；MOONEP_MIN_DISPATCH_TOKENS=16，每 rank home 载荷 rank0 {payloads[0]} / '
                     f'rank1 {payloads[1]} / rank2 {payloads[2]} / rank3 {payloads[3]}</div>')
            cell_by_shape = {shape: D[mbt][shape][route] for shape in SHAPES}
            P.append(summary_table(cell_by_shape))
            for shape in SHAPES:
                P.append(f'<details class="shape"><summary>bench 输入 {shape} 的全部字段'
                         f'（每个轮次的原始值 + 变化）</summary>')
                P.append(detail_table(cell_by_shape[shape], shape))
                P.append("</details>")
            P.append("</details>")

    P.append('<h2>附注</h2><ul>'
             '<li>上面每个数都能在 <code>/tmp/mc2_probe/grid/&lt;运行名&gt;/&lt;形状&gt;_c&lt;轮次&gt;.json</code> '
             '里按 <code>运行名</code> + <code>形状</code> + <code>轮次</code> 找回。</li>'
             '<li>这批数据只有两个 bench 输入，两个服务配置的 KV 池大小不同（见「实验条件」），与 H16 轮相同'
             '（每 rank 29133 / 23436 tokens）。</li>'
             '<li>三条腿的配对口径、四个变化量与符号读法见「统计口径」；与 H16 旧基线的对照及数据质量例外见'
             '「与 H16 旧基线对照」。</li>'
             '<li>本报告只统计这一轮 real-T 重跑的数据；H16 报告（<code>moonep_off_vs_slot_min16_bench_data_report.html</code>）'
             '与更早的 <code>moonep_off_vs_slot_bench_data_report.html</code> 仍是原文件，没有被覆盖或改动。</li>'
             '</ul></body></html>')
    return "\n".join(P)


def main():
    global GRID, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=GRID)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    GRID, OUT = args.grid, args.out

    D = collect()
    with open(OUT + ".md", "w") as fh:
        fh.write(render_md(D))
    with open(OUT + ".html", "w") as fh:
        fh.write(render_html(D))
    print(f"written: {OUT}.md / {OUT}.html")


if __name__ == "__main__":
    main()
