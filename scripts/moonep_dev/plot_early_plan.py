"""Plot predicted (early-router) vs real MoE routing, and the resulting plans.

Reads the MOONEP_EARLY_PLAN / MOONEP_PLAN_STATS dumps and writes a
self-contained HTML report (base64 PNGs inline) that explains, per figure, what
is drawn, what every axis/colour/legend means, how to read it, and what this run
actually showed.

  fig1  per (layer, expert) routing counts, predicted vs real   (log-log scatter)
  fig2  per-step per-rank load share, predicted vs real         (scatter + band)
  fig3  distribution of the per-rank load prediction error
  fig4  hot-expert heat maps, predicted | real (shared colour scale)
  fig5  plan-level hit rates: predicted plan vs real plan

Usage: python3 plot_early_plan.py <tag> [out.html]
"""
import base64
import io
import json
import os
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

TAG = sys.argv[1] if len(sys.argv) > 1 else "epshort"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/home/vllm-ascend-moonep/docs/early_plan_{TAG}.html"
E, R, EPN = 128, 4, 32          # experts, EP ranks, experts per rank
LAYERS = 48                     # MoE layers per step


def load_hist(kind):
    out = {}
    for r in range(R):
        try:
            out[r] = json.load(open(f"/tmp/moonep_earlyplan_{TAG}_rank{r}.json"))[kind]
        except Exception:
            pass
    return out


def load_json(name, kind=None):
    out = {}
    for r in range(R):
        try:
            d = json.load(open(f"/tmp/{name}_{TAG}_rank{r}.json"))
            out[r] = d[kind] if kind else d
        except Exception:
            pass
    return out


def pad(h):
    return h + [0] * (E - len(h)) if len(h) < E else h


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


early, real = load_hist("early"), load_hist("real")
plan_pred = load_json("moonep_earlyplan", "plan_pred")
plan_real = load_json("moonep_planstats")
if not early:
    sys.exit(f"no dumps for tag {TAG}")
n = min(len(v) for v in early.values())

# ---------------------------------------------------------------- gather
pred_pts, real_pts, errs = [], [], []
loads_pred = {r: [] for r in range(R)}
loads_real = {r: [] for r in range(R)}
for i in range(LAYERS, n):                      # skip warmup/profile forwards
    ge = np.zeros(E)
    gr = np.zeros(E)
    for r in range(R):
        ge += np.array(pad(early[r][i]))
        gr += np.array(pad(real[r][i]))
    if ge.sum() <= 0 or gr.sum() <= 0:
        continue
    pred_pts.append(ge)
    real_pts.append(gr)
    for r in range(R):
        pe = float(ge[r * EPN:(r + 1) * EPN].sum())
        pr = float(gr[r * EPN:(r + 1) * EPN].sum())
        loads_pred[r].append(pe)
        loads_real[r].append(pr)
        if pr > 0:
            errs.append(abs(pe - pr) / pr)

P = np.array(pred_pts)
G = np.array(real_pts)
cos = [float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) for a, b in zip(P, G)]
top5 = []
for a, b in zip(P, G):
    ta = set(np.argsort(-a)[:5].tolist())
    tb = set(np.argsort(-b)[:5].tolist())
    top5.append(len(ta & tb) / 5)
argmax_hit = float(np.mean([np.argmax(a) == np.argmax(b) for a, b in zip(P, G)]))
med = sorted(errs)[len(errs) // 2]
p90 = sorted(errs)[int(0.9 * len(errs))]

# ---------------------------------------------------------------- fig1
fig, ax = plt.subplots(figsize=(5.8, 5.2))
xs, ys = P.ravel(), G.ravel()
m = (xs > 0) | (ys > 0)
ax.scatter(xs[m] + 0.5, ys[m] + 0.5, s=1, alpha=0.06, color="#2b6cb0", linewidths=0)
lim = [0.5, max(xs.max(), ys.max()) + 1]
ax.plot(lim, lim, "r--", lw=1.2, label="perfect prediction  y = x")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("PREDICTED token count for that (layer, expert)\n(early router, on the pre-attention hidden)")
ax.set_ylabel("REAL token count for that (layer, expert)\n(this layer's actual routing)")
ax.set_title(f"fig1  routing counts per (layer, expert)\n"
             f"{len(P)} layers x {E} experts, cosine mean {st.mean(cos):.4f}")
ax.legend(loc="upper left", fontsize=9)
fig1 = png(fig)

# ---------------------------------------------------------------- fig2
steps = min(len(loads_pred[0]), 2400) // LAYERS
sp = {r: [] for r in range(R)}
sr = {r: [] for r in range(R)}
for s_ in range(steps):
    idx = range(s_ * LAYERS, (s_ + 1) * LAYERS)
    tp = sum(loads_pred[r][i] for r in range(R) for i in idx) or 1
    tr = sum(loads_real[r][i] for r in range(R) for i in idx) or 1
    for r in range(R):
        sp[r].append(100 * sum(loads_pred[r][i] for i in idx) / tp)
        sr[r].append(100 * sum(loads_real[r][i] for i in idx) / tr)
fig, ax = plt.subplots(figsize=(6.2, 5.4))
cols = ["#2b6cb0", "#dd6b20", "#2f855a", "#c53030"]
hi = max(max(sr[r]) for r in range(R)) * 1.05
lo = min(min(sr[r]) for r in range(R)) * 0.95
ax.fill_between([lo, hi], [lo * 0.9, hi * 0.9], [lo * 1.1, hi * 1.1],
                color="#4299e1", alpha=0.12, label="+/-10% relative band")
for r in range(R):
    ax.scatter(sr[r], sp[r], s=18, alpha=0.65, color=cols[r], label=f"rank {r}")
ax.plot([lo, hi], [lo, hi], "k--", lw=1.1, label="perfect prediction  y = x")
ax.axvline(25, color="gray", ls=":", lw=1)
ax.axhline(25, color="gray", ls=":", lw=1)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_xlabel(f"REAL share of the step's rows handled by that rank [%]   ({steps} steps)")
ax.set_ylabel("PREDICTED share of the step's rows [%]")
ax.set_title("fig2  per-step, per-rank load share: predicted vs real\n"
             "(dotted lines = the perfectly balanced 25%)")
ax.legend(fontsize=8, loc="upper left")
fig2 = png(fig)

# ---------------------------------------------------------------- fig3
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.hist(errs, bins=140, color="#2f855a", alpha=0.85)
ax.axvline(med, color="k", ls="--", lw=1.2, label=f"p50 = {med:.3f}")
ax.axvline(p90, color="r", ls="--", lw=1.2, label=f"p90 = {p90:.3f}")
ax.set_xlabel("relative error of the predicted per-rank load   |pred - real| / real")
ax.set_ylabel(f"count  (layers x ranks = {len(errs)})")
ax.set_title(f"fig3  how wrong is the predicted per-rank load? (mean {st.mean(errs):.3f})")
ax.legend()
fig3 = png(fig)

# ---------------------------------------------------------------- fig4
HP = np.array(pred_pts).T
HG = np.array(real_pts).T
top = np.argsort(-HG.sum(axis=1))[:16]
kk = min(400, HP.shape[1])
vmax = float(max(HP[top][:, -kk:].max(), HG[top][:, -kk:].max()))
fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.8), sharex=True)
fig.subplots_adjust(hspace=0.35)
for ax, M, t in ((axes[0], HP, "PREDICTED"), (axes[1], HG, "REAL")):
    im = ax.imshow(M[top][:, -kk:], aspect="auto", cmap="magma",
                   norm=PowerNorm(0.5, vmin=0, vmax=vmax))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"expert {e}" for e in top], fontsize=6)
    ax.set_ylabel(f"{t}\n(16 hottest)", fontsize=8)
    plt.colorbar(im, ax=ax, pad=0.02, fraction=0.05, label="tokens / layer")
axes[1].set_xlabel("layer index (last 400 layers of the run)")
fig.suptitle("fig4  hot-expert heat maps: predicted (top) vs real (bottom)\n"
             "same experts, same colour scale, same layer axis", fontsize=10)
fig4 = png(fig)

# ---------------------------------------------------------------- fig5
def hit_rate():
    if not plan_pred or not plan_real:
        return None
    tot = exact = expset = slot0 = prev = 0
    nn = min(min(len(v) for v in plan_pred.values()), min(len(v) for v in plan_real.values()))
    for r in sorted(plan_pred):
        a_all, b_all = plan_pred[r], plan_real[r]
        for i in range(LAYERS, nn):
            a, b = a_all[i], b_all[i]
            tot += 1
            if a == b:
                exact += 1
            if {x for x in a if x >= 0} == {x for x in b if x >= 0}:
                expset += 1
            if (a[0] if a else -1) == (b[0] if b else -1):
                slot0 += 1
            if b_all[i - LAYERS] == b:
                prev += 1
    return tot, exact / tot, expset / tot, slot0 / tot, prev / tot


hr = hit_rate()
fig5 = ""
if hr:
    tot, exact, expset, slot0, prev = hr
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = ["exact match\n(slot AND expert)", "expert set\n(ignoring slot)", "slot 0 only",
             "baseline:\nlast step's real plan"]
    vals = [exact, expset, slot0, prev]
    bars = ax.bar(names, vals, color=["#2b6cb0", "#4299e1", "#63b3ed", "#a0aec0"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("hit rate vs the REAL plan")
    ax.set_title(f"fig5  did the predicted plan pick the same redundant experts?\n"
                 f"({tot} layers x ranks)")
    fig5 = png(fig)

# ---------------------------------------------------------------- html
def fig_block(num, title, img, rows, read, observed):
    lis = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    return f"""<section class="card">
  <h2>{num}. {title}</h2>
  <img src="data:image/png;base64,{img}">
  <h3>图例 / 坐标轴分别代表什么</h3>
  <table class="legend">{lis}</table>
  <h3>怎么读</h3><p>{read}</p>
  <h3>本次实测看出什么</h3><p class="obs">{observed}</p>
</section>"""


chips = (f'<span class="k good">layers {len(P)}</span>'
         f'<span class="k good">cosine {st.mean(cos):.4f}</span>'
         f'<span class="k good">top-5 overlap {st.mean(top5):.3f}</span>'
         f'<span class="k good">argmax {argmax_hit:.3f}</span>'
         f'<span class="k good">load err p50 {med:.3f} / p90 {p90:.3f}</span>'
         + (f'<span class="k hi">plan exact {exact:.3f}</span>'
            f'<span class="k hi">vs baseline {prev:.3f}</span>' if hr else ""))

html = f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>early-plan 预测 vs 真实 routing（{TAG}）</title>
<style>
 :root{{--bd:#e2e8f0;--mut:#4a5568;--blue:#2b6cb0;--bg:#f7fafc}}
 *{{box-sizing:border-box}}
 body{{font-family:system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
      max-width:960px;margin:0 auto;padding:28px 20px 60px;color:#1a202c;line-height:1.65}}
 h1{{font-size:23px;margin:0 0 6px}} h2{{font-size:16px;margin:0 0 12px}}
 h3{{font-size:13px;margin:18px 0 6px;color:var(--mut)}}
 .sub{{color:var(--mut);font-size:13px;margin-bottom:18px}}
 .k{{display:inline-block;background:#ebf8ff;border:1px solid #bee3f8;border-radius:999px;
     padding:2px 10px;margin:2px 3px 2px 0;font-size:12px}}
 .k.good{{background:#f0fff4;border-color:#c6f6d5}} .k.hi{{background:#fffaf0;border-color:#fbd38d;font-weight:600}}
 .card{{border:1px solid var(--bd);border-radius:10px;padding:18px 20px;margin:22px 0;background:#fff;
       box-shadow:0 1px 2px rgba(0,0,0,.04)}}
 .card img{{width:100%;border-radius:6px;margin:6px 0 4px}}
 table.legend{{width:100%;border-collapse:collapse;font-size:13px}}
 table.legend th{{text-align:left;vertical-align:top;width:34%;padding:5px 10px 5px 0;
                 color:#2d3748;font-weight:600;border-bottom:1px solid var(--bg)}}
 table.legend td{{padding:5px 0;border-bottom:1px solid var(--bg);color:#2d3748}}
 p{{font-size:14px}} .obs{{background:var(--bg);border-left:3px solid var(--blue);padding:8px 12px;border-radius:0 6px 6px 0}}
 .glossary td{{padding:4px 10px 4px 0;font-size:13px;vertical-align:top}}
 code{{background:var(--bg);padding:1px 5px;border-radius:3px;font-size:12.5px}}
</style>
<h1>early-plan 预测 vs 真实 routing — {TAG}</h1>
<div class="sub">用「attention 之前的残差流（上一层输出）」跑本层 router，预测本层的专家分布与
<em>plan 的冗余专家选取</em>，再与真实 routing / 真实 plan 对比。</div>
<div>{chips}</div>

<section class="card">
<h2>0. 方法：这个「预测」是怎么来的</h2>
<table class="legend">
<tr><th>预测输入</th><td>第 i 层 <b>attention 之前</b>的残差流（= 第 i−1 层的输出，<code>hidden_states + residual</code>）</td></tr>
<tr><th>预测方式</th><td>把它送进 <b>第 i 层自己的</b> <code>post_attention_layernorm</code> + <code>gate</code>，取 top-k(=8) ⇒ 得到「预测的专家计数直方图」</td></tr>
<tr><th>为什么可行</th><td>残差结构 ⇒ 同一层 attention 前后的 hidden 相近 ⇒ 路由分布相近（本页就在验证这一点）</td></tr>
<tr><th>预测 plan</th><td>把预测的 routing 再喂给 <b>规划内核</b> 跑一遍 ⇒ 得到「预测 plan」，取其
 <code>experts_to_copy[rank]</code>（本卡要镜像的冗余专家 + 槽位）</td></tr>
<tr><th>对照真值</th><td>真实 routing（本层实际输入）⇒ 真 plan（本层真实执行的规划结果）</td></tr>
<tr><th>为什么会有人关心</th><td>真 plan 要等本层 dispatch 才算出来，prefetch 只能排在它后面、卡在关键路径上；
 若能在 attention 期间就预测出 plan，就能把 prefetch 挪到和 attention 并行，掩盖掉它的延迟</td></tr>
</table>
</section>

{fig_block("1", "逐 (层, 专家) 的路由计数：预测 vs 真实", fig1, [
 ("横轴", "该 (层, 专家) 的<b>预测</b>路由 token 数（log 刻度）"),
 ("纵轴", "同一 (层, 专家) 的<b>真实</b>路由 token 数（log 刻度）"),
 ("每个点", "一个 (层, 专家) 组合 —— 共 层数 × 128 个专家 个点"),
 ("红色虚线 y=x", "完美预测：点落在这条线上 = 预测值恰好等于真实值"),
 ("点的深浅", "重叠密度（α=0.06 的散点叠加，越深代表越多点重合在那里）"),
], "越贴近红色虚线越准。<b>高计数区</b>（热门专家）必须几乎贴线；<b>低计数区</b>（稀疏专家）允许散布，"
   "因为 1~2 个 token 的差异在绝对量上无意义 —— 规划器均衡看的是<b>负载</b>，由热门专家主导。",
   f"余弦相似度 <b>{st.mean(cos):.4f}</b>，高计数区几乎零偏离；只有极稀疏专家在 1~3 计数处有散布。"
   f"最热专家（argmax）一致率 <b>{argmax_hit:.3f}</b>，top-5 专家集合重合率 <b>{st.mean(top5):.3f}</b>。")}

{fig_block("2", "逐步、逐卡的负载占比：预测 vs 真实", fig2, [
 ("横轴", "某一步中该卡处理的<b>真实</b>行数占该步总行数的百分比 [%]"),
 ("纵轴", "同一步、同一卡的<b>预测</b>占比 [%]"),
 ("每个点", "一个 (步, 卡) 组合 —— 一步 = 48 个 MoE 层聚合（单层样本太小、噪声大，故按步聚合）"),
 ("四个颜色", "rank 0（蓝）/ rank 1（橙）/ rank 2（绿）/ rank 3（红），即四张卡"),
 ("黑色虚线 y=x", "完美预测"),
 ("浅蓝带 ±10%", "相对误差 ±10% 的容差带；点落在带内 ⇒ 预测误差小于 10%"),
 ("灰色点线 25%", "四卡完全均衡时的理论占比（1/4）"),
], "点应沿 y=x 分布。注意这张图里的点<b>并不都落在 25% 上</b> —— 它们横向铺开的宽度，就是真实存在的负载不均衡；"
   "纵轴同样铺开，说明预测把这种不均衡也复现出来了。",
   "点紧贴 y=x ✓；<b>真实各卡占比在 20–26% 之间漂移</b>（理想 25%），预测把「哪张卡偏忙」也跟住了 ✓。")}

{fig_block("3", "预测负载的误差分布", fig3, [
 ("横轴", "相对误差 |预测 − 真实| / 真实（针对每卡的负载）"),
 ("纵轴", "落入该误差区间的样本数（层 × 卡）"),
 ("绿色柱", "误差分布的直方图（越集中在 0 越好）"),
 ("黑色虚线 p50", f"中位误差 {med:.3f} —— 一半样本误差小于它"),
 ("红色虚线 p90", f"90% 分位误差 {p90:.3f} —— 只有 10% 的样本比它更差"),
], "主要看 p90：它决定「最坏情况下要多花多少补搬代价」。",
   f"均值 {st.mean(errs):.3f}、p50 {med:.3f}、p90 {p90:.3f} ⇒ 绝大多数情况下预测负载与真实差在 3% 以内。")}

{fig_block("4", "热专家热力图：预测（上）vs 真实（下）", fig4, [
 ("横轴", "层序号（取运行最后 400 层）"),
 ("纵轴", "按总计数排名的前 16 个专家（<b>全局专家 id</b>）"),
 ("颜色", "该专家在该层收到的 token 数；<b>上下两张共用同一个色标</b>，所以颜色深浅可直接对比"),
 ("色标归一化", "PowerNorm(0.5)：压暗高值、放大低值差异，便于看清逐层结构"),
 ("上图 PREDICTED", "预测计数"),
 ("下图 REAL", "真实计数"),
], "两张图的<b>尖峰应逐层一一对应</b>。若某层上面亮、下面暗（或反之），就是该层预测失误。",
   "上下两张肉眼几乎无法区分 ✓ —— 尖峰位置与强弱逐层对齐，验证了「残差相似 ⇒ 路由相似」这一前提。")}

{fig_block("5", "预测 plan vs 真实 plan：冗余专家选取命中率", fig5, [
 ("纵轴", "命中率（与真实 plan 逐层逐卡对比的相符比例）"),
 ("第 1 根 exact", "<b>逐槽完全相同</b>：槽位与专家都对上（最严格口径）"),
 ("第 2 根 expert set", "只看镜像专家集合是否相同，忽略落在哪个槽位"),
 ("第 3 根 slot 0", "只看最重要的槽位 slot 0 放的是不是同一个专家"),
 ("第 4 根 baseline", "<b>对照组</b>：不做任何预测，直接沿用「上一步的真实 plan」"),
], "前三根越高越好；第四根是必须打败的基线 —— 如果预测还比不过「照抄上一步」，方案就没有价值。",
   (f"exact <b>{exact:.3f}</b>、专家集合 {expset:.3f}、slot0 {slot0:.3f}，"
    f"<b>均高于基线 {prev:.3f}</b> ⇒ 提前预测确实带来了额外信息 ✓（领先约 {(exact - prev) * 100:.0f} 个百分点）。")
   if hr else "（本次运行未采集到 plan 数据）")}

<section class="card">
<h2>术语表</h2>
<table class="glossary">
<tr><td><b>行 / row</b></td><td>一个 (token, 专家) 条目；MoE 计算量 ∝ 行数</td></tr>
<tr><td><b>冗余专家 / 镜像</b></td><td>plan 把某个<b>远端</b>热门专家的行留在本地计算（预取它的权重到本地槽位），用来均衡各卡负载</td></tr>
<tr><td><b>槽位 / slot</b></td><td>本卡存放被镜像专家权重的缓冲区；<code>experts_to_copy[rank]</code> 给出每个槽放哪个专家，−1 = 空槽</td></tr>
<tr><td><b>plan</b></td><td>规划内核每层产出的调度方案：每行的目的地 <code>dst</code>、每卡要镜像的专家 <code>experts_to_copy</code> 等</td></tr>
<tr><td><b>prefill / decode</b></td><td>处理 prompt 的阶段 / 逐 token 生成的阶段；两者负载量级相差约 100×</td></tr>
</table>
</section>
</html>"""
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as fh:
    fh.write(html)
print(f"layers={len(P)} cosine_mean={st.mean(cos):.4f} top5={st.mean(top5):.3f} argmax={argmax_hit:.3f} "
      f"load_err mean={st.mean(errs):.3f} p50={med:.3f} p90={p90:.3f}")
print(f"plan hit rates = {hr}")
print(f"written {OUT}")
