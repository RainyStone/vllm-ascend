"""Two bar charts vs T, in one self-contained HTML page.

  chart 1: TTFT p50 (MoonEP slot vs all-gather) against T
  chart 2: cross-rank load imbalance (max/mean of per-rank dispatched rows)

T = per-rank per-step MoE token count = (N_SEQ / DP) * prompt_len. Each rung
picks MBT so the prefill stays as close to one step as the scheduler allows,
so T is the only variable across rungs.

TTFT comes from /tmp/bench_dp2ep2_lad_<rung>_<route>_*.json; the imbalance from
the MOONEP_BALANCE_STATS lines in /tmp/stat_<rung>.log (rows_all per rank).
No matplotlib on this box, so the charts are hand-rolled SVG.

  python3 scripts/moonep_dev/plot_ladder.py --out /home/tmp/ladder.html
"""
import argparse
import glob
import json
import math
import re

RUNGS = ["A", "B", "C", "D"]
DP = 8
PROMPT_TOKENS = 69          # GSM8K p50 token length


def load_ttft():
    out = {}
    for f in glob.glob("/tmp/bench_dp2ep2_lad_*.json"):
        d = json.load(open(f))
        m = re.match(r"lad_([A-D])_(slot|ag)$", d.get("tag", ""))
        if not m:
            continue
        rung, route = m.group(1), m.group(2)
        steps = ((d.get("rank_res") or [{}])[0] or {}).get("n_prefill_steps")
        out[(rung, route)] = {
            "ttft": (d.get("ttft_ms") or {}).get("p50"),
            "steps": steps,
            "tok": d.get("prefill_tokens", 0),
            "T": (d.get("n_seq", 0) // DP) * PROMPT_TOKENS,
        }
    return out


def load_imbalance():
    out = {}
    for f in glob.glob("/tmp/stat_*.log"):
        m = re.search(r"stat_([A-D])\.log$", f)
        if not m:
            continue
        vals = [int(x) for x in re.findall(r"rows_all=(\d+)", open(f, errors="ignore").read())]
        if vals:
            out[m.group(1)] = max(vals) / (sum(vals) / len(vals))
    return out


def bars(xs, series, ymax, w, h, ylabel, fmt, ymin=0.0, ref=None, ref_label=""):
    """series = [(label, color, [v0..v3])]; xs = x pixel positions."""
    pl, pb, pt, pr = 74, 46, 22, 16
    out = [f'<svg width="{w}" height="{h}" style="background:#fff">']
    span = ymax - ymin
    for i in range(6):
        y = pt + (h - pt - pb) * i / 5
        val = ymax - span * i / 5
        out.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{w-pr}" y2="{y:.0f}" stroke="#eee"/>')
        out.append(f'<text x="{pl-8}" y="{y+4:.0f}" font-size="11" text-anchor="end" '
                   f'fill="#666">{fmt.format(val)}</text>')
    if ref is not None:
        y = pt + (h - pt - pb) * (1 - (ref - ymin) / span)
        out.append(f'<line x1="{pl}" y1="{y:.0f}" x2="{w-pr}" y2="{y:.0f}" '
                   f'stroke="#c00" stroke-dasharray="5,4" stroke-width="1.5"/>')
        out.append(f'<text x="{w-pr-4}" y="{y-5:.0f}" font-size="11" text-anchor="end" '
                   f'fill="#c00">{ref_label}</text>')
    nb = len(series)
    barw = 42 / nb
    for i, x in enumerate(xs):
        for j, (_, color, vals) in enumerate(series):
            v = vals[i]
            if v is None:
                continue
            bh = (h - pt - pb) * (v - ymin) / span
            y = h - pb - bh
            bx = x - 21 + j * barw
            out.append(f'<rect x="{bx:.0f}" y="{y:.0f}" width="{barw-3:.0f}" '
                       f'height="{bh:.0f}" fill="{color}"/>')
            out.append(f'<text x="{bx+barw/2-1.5:.0f}" y="{y-4:.0f}" font-size="10" '
                       f'text-anchor="middle" fill="#333">{fmt.format(v)}</text>')
    out.append(f'<text x="{pl-56}" y="{pt-6}" font-size="11" fill="#666">{ylabel}</text>')
    # legend
    lx = pl + 8
    for label, color, _ in series:
        out.append(f'<rect x="{lx}" y="{pt-16}" width="11" height="11" fill="{color}"/>')
        out.append(f'<text x="{lx+16}" y="{pt-6}" font-size="12" fill="#333">{label}</text>')
        lx += 26 + 7.0 * len(label)
    out.append('</svg>')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/tmp/ladder.html")
    args = ap.parse_args()

    ttft = load_ttft()
    imb = load_imbalance()

    rows = []
    for rung in RUNGS:
        s = ttft.get((rung, "slot"), {})
        a = ttft.get((rung, "ag"), {})
        rows.append({
            "rung": rung, "T": s.get("T") or a.get("T"), "tok": s.get("tok"),
            "steps": f"{s.get('steps')}/{a.get('steps')}",
            "slot": s.get("ttft"), "ag": a.get("ttft"),
            "ratio": (s["ttft"] / a["ttft"]) if (s.get("ttft") and a.get("ttft")) else None,
            "imb": imb.get(rung),
        })
    rows = [r for r in rows if r["T"]]

    # log-spaced x positions so the 64x range stays readable
    lo, hi = math.log10(min(r["T"] for r in rows)), math.log10(max(r["T"] for r in rows))
    W = 760
    pl, pr = 74, 16
    xs = [pl + (W - pl - pr) * ((math.log10(r["T"]) - lo) / (hi - lo or 1)) * 0.82 + 60
          for r in rows]

    tticks = "".join(
        f'<text x="{x:.0f}" y="{h}" font-size="12" text-anchor="middle" fill="#333">'
        f'{r["T"]:.0f}<tspan fill="#999" font-size="10"> ({r["rung"]})</tspan></text>'
        for x, r in zip(xs, rows) for h in [348])

    ttft_max = max([v for r in rows for v in (r["slot"], r["ag"]) if v] or [1]) * 1.18
    c1 = bars(xs, [("MoonEP slot", "#d9534f", [r["slot"] for r in rows]),
                   ("all-gather", "#4a7ebb", [r["ag"] for r in rows])],
              ttft_max, W, 370, "TTFT p50 (ms)", "{:.0f}")
    c1 = c1.replace("</svg>", tticks + "</svg>")

    imb_max = max([r["imb"] for r in rows if r["imb"]] or [1.0])
    ymax2 = 1.0 + (imb_max - 1.0) * 4 if imb_max > 1.0 else 1.5
    c2 = bars(xs, [("跨卡不均衡 max/mean", "#3c9d6a", [r["imb"] for r in rows])],
              ymax2, W, 300, "max / mean", "{:.3f}", ymin=1.0,
              ref=1.0, ref_label="1.000 = 完全均衡")
    c2 = c2.replace("</svg>", "".join(
        f'<text x="{x:.0f}" y="{278}" font-size="12" text-anchor="middle" fill="#333">'
        f'{r["T"]:.0f}<tspan fill="#999" font-size="10"> ({r["rung"]})</tspan></text>'
        for x, r in zip(xs, rows)) + "</svg>")

    note = ("每档只改 T（= 每卡每步 MoE token 数 = N_SEQ/DP × prompt 长度），"
            "其余配置不变；x 轴按 log 间距。<br>"
            "不均衡取 MOONEP_BALANCE_STATS 的 per-rank <code>rows_all</code> 的 max/mean。")

    concl = """
<h3>③ 结论</h3>
<ol>
<li><b>TTFT 随 T 单调恶化。</b> T 从 69 涨到 4416（64×），slot/all-gather 比值从
<b>1.09 涨到 2.47</b>。方向明确：<b>T 越小 MoonEP 越接近打平</b>，但即便在 T=69
（≈ decode 的 64）它仍然慢 <b>8.9%</b>，没有反超。</li>

<li><b>跨卡不均衡与 T 无关。</b> 四档全是 <b>1.009–1.011</b>，缩小 T 完全没有放大
不均衡 —— 这条假设被实验否掉。原因：T=69、K=8 时每卡每层只有 552 条 route
摊到 128 个专家，泊松噪声本身只有 ±4%，而 GSM8K 题目高度同质，实测偏差比随机还小。
<b>⇒ 这套负载上不存在可供 MoonEP 均衡的对象。</b></li>

<li><b>两条路线都是"两趟、等量字节"的跨卡搬运</b>，差别不在搬多少，而在效率：</li>
</ol>

<table>
<tr><th>路线</th><th>进 MoE 前</th><th>出 MoE 后</th><th>每卡跨卡字节量</th><th>耗时随 T 涨 64× 的倍数</th></tr>
<tr><td><b>all-gather</b></td><td>all_gather</td><td>reduce_scatter</td>
    <td>2(R−1)·T·H = <b>14 T·H</b></td><td><b>8.7×（次线性）</b></td></tr>
<tr><td><b>MoonEP slot</b></td><td>dispatch（MTE put）</td><td>combine（MTE get + K-sum）</td>
    <td>2·T·K·(R−1)/R·H = <b>14 T·H</b></td><td><b>19.7×（接近线性）</b></td></tr>
</table>

<p class="note">（R=8、K=8：每个 token 的 K 条 route 里平均 7/8 跨卡 ⇒ 每趟 ≈7·T·H。
字节量相等，所以差异只能来自<b>搬运的形状与实现层级</b>：all-gather/reduce-scatter 搬的是
<b>连续大块</b>（HCCL 深度优化，实测每层切成 ~67 个大分片），而 MoonEP 搬的是
<b>按专家分组的行级 scatter</b>，包小、每次启动开销摊不开 —— 这正对应"线性 vs 次线性"。）</p>

<p><b>T=69 时那 8.9%（6.2 ms）的来源</b>：MoonEP 每层多付 planning（跨卡直方图交换 +
1 次全局栅栏）+ 4 次 <code>barrier_on_stream</code>，约 <b>129 µs/层 × 48 层</b>；
而 all-gather 只有一次小搬运。</p>

<h4>判定与下一步</h4>
<ol>
<li><b>在这类负载（GSM8K、8 卡 EP8、单请求/卡）上 MoonEP 赢不了</b>：可均衡的对象不存在
（不均衡仅 1%），而单位 token 的搬运效率更低。</li>
<li><b>要救的方向是让 dispatch/combine 搬"大块"</b>：例如先按目标 rank 做一次连续化重排，
再整块 put/get，而不是逐 (token, expert) 行发小包。这是库侧 kernel 改造。</li>
<li><b>待查的矛盾</b>：早前文档记录的 MC2 decode 跨卡不均衡是 p50 1.20 / p90 1.50 / max 3.0，
而本次 slot 在 T=69 只有 1.01 —— 相差 20 倍，口径可能根本不同。<b>这个矛盾决定了 MoonEP
到底有没有适用场景，应先查清再决定投入。</b></li>
</ol>
"""

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>MoonEP T ladder</title><style>
body{{font-family:system-ui,sans-serif;margin:28px;color:#222;max-width:860px}}
table{{border-collapse:collapse;margin:14px 0}}
th,td{{border:1px solid #ddd;padding:6px 11px;font-size:13px}}th{{background:#f5f5f5}}
h2{{margin-bottom:4px}}h3{{margin:24px 0 6px}}
.note{{color:#666;font-size:12.5px;line-height:1.6}}
</style></head><body>
<h2>Qwen3-30B-A3B / 8 卡 EP8 / GSM8K / GEN=1 —— TTFT 与不均衡 vs T</h2>
<p class="note">{note}</p>
<h3>① TTFT p50 vs T</h3>{c1}
<h3>② 跨卡负载不均衡 vs T</h3>{c2}
<h3>原始数据</h3>
<table><tr><th>档</th><th>T / 卡</th><th>prefill tok</th><th>步数 slot/ag</th>
<th>TTFT slot</th><th>TTFT all-gather</th><th>slot/ag</th><th>不均衡</th></tr>
{"".join(f'<tr><td><b>{r["rung"]}</b></td><td>{r["T"]:.0f}</td><td>{r["tok"]}</td><td>{r["steps"]}</td>'
         f'<td>{r["slot"]:.1f}</td><td>{r["ag"]:.1f}</td><td>{r["ratio"]:.2f}</td>'
         f'<td>{r["imb"]:.3f}</td></tr>' for r in rows)}
</table>{concl}</body></html>"""

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"wrote {args.out}  ({len(html)} bytes)")
    for r in rows:
        print(f'  {r["rung"]}: T~{r["T"]:.0f}  slot={r["slot"]:.1f} ag={r["ag"]:.1f} '
              f'ratio={r["ratio"]:.2f} imb={r["imb"]:.3f}')


if __name__ == "__main__":
    main()
