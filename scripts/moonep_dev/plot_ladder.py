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

<h3>④ 为什么 MoonEP 是"线性"：dispatch/combine 的搬运粒度</h3>

<p><b>现状：每条 route 一次 put，每次搬一整行 H。</b> 外层循环遍历的是 route
（<code>token_id = i/K</code>, <code>topk_id = i%K</code>），内层每条 route 发一次：</p>

<pre style="background:#f7f7f7;padding:10px;font-size:12.5px;overflow-x:auto">uint64_t hidden_off_bytes = loff * H * sizeof(HiddenT);   // 对方对称内存里该 route 的槽
aclshmemx_mte_put_nbi(
    hidden_symmetric + hidden_off_bytes,   // 目标 VA（aclshmem_ptr 拿到的对端地址）
    hidden_sh + token_id * H,              // 源：本地该 token 的 hidden 行
    row_elements,                          // = H 个元素（Qwen3: 2048 bf16 = 4 KB）
    dest_rank, event_id);</pre>

<table>
<tr><th></th><th>一次搬多少</th><th>每层每卡次数</th><th>每步（48 层）</th></tr>
<tr><td><b>MoonEP dispatch</b></td><td><b>4 KB</b>（1 行）</td><td><b>T×K</b>（T=4416 → 35,328）</td><td><b>~170 万次</b></td></tr>
<tr><td><b>all-gather</b></td><td>数 MB（连续大块）</td><td>~67</td><td>~3,200 次</td></tr>
</table>
<p class="note">⇒ <b>报文数差约 500 倍</b>。这就是 all-gather 次线性（T 涨 64×、耗时涨 8.7×）
而 MoonEP 接近线性（19.7×）的根源：同样的字节量，一个走大块、一个走小包。</p>

<p><b>为什么必须是小包？</b> 因为发往同一个 dest 的行在源端 <code>hidden_sh</code> 里是
<b>散落的</b>（hidden 按 token 顺序排，而每个 token 的 K 条 route 各去不同的 rank）。</p>

<h4>大块化方案（三步）</h4>
<pre style="background:#f7f7f7;padding:10px;font-size:12.5px;overflow-x:auto">① 源端：扫一遍 topk_idx，按 dest 分组，把散落的行连续化到 staging buffer
   —— 一次卡内 DMA（读 T×H + 写 T×H，本地带宽，便宜）
② 网络：每个 dest 一次 mte_put_nbi，搬 [n_rows × H] 的连续块
   —— put 次数 T×K → R−1 = 7 次/层/卡   ★ 500× 的降幅
③ 接收端：目标布局当前是「专家连续」（GEMM 的 group_list 要求），而大块 put
   要求「来源连续」⇒ 接收端补一次 permute 成 expert-grouped</pre>

<p class="note">第 ③ 步不是新发明 —— 这正是 all-gather 路线里
<code>aclnnMoeInitRoutingV3</code> 干的事（实测 265 ms），也是 MC2/DeepEP 的标准 dispatch 结构。
做完之后接收端数据形态与 all-gather 路线一致，<b>两条路线可共用同一套 MLP 接口</b>。</p>

<table>
<tr><th></th><th>报文数/层/卡</th><th>需要源端重排</th></tr>
<tr><td><b>逐行（现状）</b></td><td>T×K = 35,328</td><td>❌ 就地发</td></tr>
<tr><td><b>大块化</b></td><td>R−1 = 7</td><td>✅ 散行先连续化</td></tr>
</table>

<h4>语义澄清：这仍然是 one-sided</h4>
<p>"one-sided" 的定义是<b>通信的完成不需要对端执行配对操作</b>。
源端 compact 是发送方<b>自己设备内</b>的数据重排，<b>根本不涉及通信</b>；
发送方本来就必须参与（是它发起的那个 put）。</p>
<table>
<tr><th>步骤</th><th>谁参与</th><th>是否 one-sided</th></tr>
<tr><td>① 源端 compact</td><td>本地 AIV</td><td>不涉及通信</td></tr>
<tr><td>② 大块 put</td><td>本地 AIV → 对端内存</td><td><b>是</b></td></tr>
<tr><td>③ 接收端 permute</td><td>对端本地 AIV</td><td>不涉及通信</td></tr>
</table>
<p class="note">全程<b>没有对面 CPU 参与</b>：数据靠 <code>aclshmemx_mte_put/get_nbi</code> 直接落到对端
对称内存；完成通知靠 (a) 发送方把 flag 也 one-sided 写进对端、接收方设备侧轮询，或
(b) 设备侧栅栏（<code>aclshmemx_barrier_all_vec</code>，各 rank 的 AIV 进栅栏，不是 CPU）。
对比 HCCL：它也是设备侧，但用 <b>AICPU kernel</b> 做 rank 间协调，且是<b>集合式</b>（所有 rank 必须一起调用）。</p>

<h4>4 次全局栅栏：时机与性质</h4>
<p>profiler 实测 <code>barrier_on_stream_kernel</code> 4 次/层。来源不是"2 次 prefetch 调用"，
而是<b>一次 <code>prefetch_projections</code> 调用里的 2 个投影（w13 / w2）各发一个 kernel</b>
（<code>api.py:734</code> 的 <code>for full_weight in weight_prefetch_args</code> 循环），
而每个 kernel 前后各有 1 个栅栏（<code>csrc/ops/prefetch.cpp:118</code> 与 <code>:134</code>）
⇒ <b>2 投影 × 2 栅栏 = 4 次/层</b>。</p>
<table>
<tr><th>位置</th><th>本意</th><th>数据上必需？</th></tr>
<tr><td>进 prefetch 前</td><td>"别在别人还在用 slot 时覆盖它"</td><td><b>否</b></td></tr>
<tr><td>出 prefetch 后</td><td>"别在权重还没到时就做 GEMM"</td><td><b>否</b></td></tr>
</table>

<p><b>库作者自己就说它们不必要</b>（<code>prefetch.cpp</code> 里 <code>RELAXED</code> 开关的注释原话）：</p>
<ul>
<li>源是 peer 的 <b>home chunk</b>（init 时写好、之后再也不改）⇒ <b>pre-barrier 无物可排</b>；</li>
<li>kernel 自己的 <b>MTE3_MTE2 event</b> 已保证每次 GET 完成 ⇒ <b>post-barrier 对本地可见性也不需要</b>；</li>
<li>每个 rank 必须设同一个 env，否则<b>栅栏代际错位</b>。</li>
</ul>

<p><b>但默认不能关，因为栅栏代际是共享的</b>（<code>api.py:503</code> 原话）：
prefetch 的 host 栅栏与 planning/dispatch 的 device <code>aclshmemx_barrier_all_vec</code>
<b>共用同一个 <code>sync_counter</code></b>；rank 间代际不一致就<b>死锁</b>。</p>

<p>已实测：全集群一起关（<code>ASCEND_MOONEP_PREFETCH_RELAXED=1</code>）⇒
<b>反而慢 2.9 ms</b> ✗ —— 说明它们虽不保护数据，却<b>在当前结构下承担 rank 对齐</b>；
拿掉后偏斜不会消失，只是转移到后面的栅栏上。</p>

<h4>栅栏收敛有两条路，可以叠加</h4>
<table>
<tr><th>路径</th><th>做法</th><th>栅栏数/层</th><th>风险</th></tr>
<tr><td><b>B1</b></td><td>把两个投影的 prefetch <b>合并成一个 kernel</b>
（内核改成一次收多投影、内部循环不同 <code>Hp</code>）</td><td><b>4 → 2</b></td>
<td><b>低</b> —— 不动代际机制</td></tr>
<tr><td><b>B2</b></td><td>把 prefetch 的同步<b>从共享的栅栏代际里解耦</b>
（改用独立 event / flag 轮询）</td><td><b>2 → 0–1</b></td><td>中 —— 要动同步设计</td></tr>
</table>
<p class="note"><b>B1 是低垂果实</b>：一次 launch 的成本 + 一对栅栏，每层都能省。</p>

<h4>改造建议（详见 docs/design_moonep_bulk_transfer_and_sync.md）</h4>
<ol>
<li><b>大块化</b>：put/get 次数 T×K → R−1（源端 compact + 接收端 permute）；</li>
<li><b>B1 合并投影 prefetch</b>：栅栏 4 → 2，低风险，可先行；</li>
<li><b>B2 同步解耦</b>：栅栏 2 → 0–1。不做这步，大块化的收益会被每层的栅栏等待吃掉一半。</li>
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
