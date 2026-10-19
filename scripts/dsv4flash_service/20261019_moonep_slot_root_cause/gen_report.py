#!/usr/bin/env python3
"""Rebuild the offline HTML report from preserved experiment summaries (stdlib only)."""
from pathlib import Path
import html, json, hashlib, sys
from html.parser import HTMLParser
sys.dont_write_bytecode = True
from report_enhancements import chunk_phase_diagram, kv_gap_diagram, fix_diagram, patch_usage_html, diagram_css
from slot_comparison import render_slot_comparison, render_primary_comparison

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
history=json.loads((DATA/"historical_matrix.json").read_text(encoding="utf-8"))
final=json.loads((DATA/"alignment_fix_validation_20261017"/"summary.json").read_text(encoding="utf-8"))
guard=json.loads((DATA/"guarded_validation_20261017"/"comparison_summary.json").read_text(encoding="utf-8"))
E=html.escape
def table(headers, rows, cls=""):
 return '<div class="table-wrap"><table class="'+cls+'"><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def pct(x):
 return f'<span class="{"gain" if x>=0 else "loss"}">{x:+.2f}%</span>'
def hist_table(rows):
 return table(["路由 / 峰均比","输入 L","每轮预算 B","off TTFT","slot TTFT","TTFT 降低","输入吞吐提升"],[
 [E(x["route"])+f' / {x["skew"]:.2f}',f'{x["input_tokens"]//1024}K',f'{x["budget"]//1024}K',
 f'{x["off"]["mean_ttft_ms"]/1000:.3f} s',f'{x["slot"]["mean_ttft_ms"]/1000:.3f} s',
 pct(x["ttft_reduction_pct"]),pct(x["throughput_gain_pct"])] for x in rows])
def link(path,title):
 return f'<a href="../data/{E(path)}">{E(title)}</a>'
def patch_link(title):
 return f'<a href="../patch/tokenizer_window.patch">{E(title)}</a>'
F="alignment_fix_validation_20261017/"
G="guarded_validation_20261017/"
def phase_table(aligned):
 rows=[]
 for r in range(4):
  if aligned: seq=["P1","P2","D","P1","P2","D"]
  else: seq=["P1","P2","D","P1","P2","D"] if r==0 else ["P2","D","P1","P2","D","P1"]
  rows.append([f"rank {r}"]+[f'<span class="phase {"dummy" if q=="D" else "real"}">{q}</span>' for q in seq])
 rows.append(["全局轮类型"]+(['全真实','全真实','全 dummy']*2 if aligned else ['全真实','混合 1 真','混合 3 真']*2))
 return table(["稳态示意","轮 1","轮 2","轮 3","轮 4","轮 5","轮 6"],rows,"phases")
phase_labels={"planning":"planning","weight_prefetch":"权重预取","communication":"dispatch + combine 通信","combine_prepare":"combine 前容量张量处理","gmm":"GMM1 + GMM2","other_model_and_step_gap":"其它模型计算与轮次间隔","other_moe":"其它 MoE 准备、量化、激活"}
stage=table(["关键路径阶段","slot / s","off / s","slot − off / s"],[
 [phase_labels[x["phase"]],f'{x["slot"]:.6f}',f'{x["off"]:.6f}',f'<strong class="{"loss" if x["delta"]>0 else "gain"}">{x["delta"]:+.6f}</strong>'] for x in guard["profile_phase_table"]])
css=r"""
:root{--ink:#172b43;--muted:#53657b;--line:#dbe4ed;--paper:#fff;--bg:#f2f5f9;--navy:#112b49;--blue:#1d5dad;--teal:#086959;--red:#a32d36;--amber:#815611}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.8 "Microsoft YaHei","Noto Sans CJK SC","Segoe UI",sans-serif}
a{color:var(--blue);text-decoration-thickness:1px;text-underline-offset:3px}a:hover{color:#123966}p{margin:0 0 17px}strong{font-weight:700}
.hero{background:var(--navy);color:white;padding:54px max(28px,calc((100vw - 1220px)/2)) 40px}.eyebrow{font-size:13px;letter-spacing:2px;color:#9dc9f2}.hero h1{font-size:38px;line-height:1.35;margin:15px 0 18px;max-width:1050px}.hero p{color:#d9e7f3;max-width:1050px;font-size:18px}.meta{font-size:13px;color:#b6cbdf}
.layout{max-width:1280px;display:grid;grid-template-columns:206px minmax(0,1fr);gap:28px;margin:28px auto;padding:0 24px 70px}
nav{position:sticky;top:24px;align-self:start;font-size:13px;line-height:1.5}nav strong{display:block;margin:8px 10px 12px}nav a{display:block;padding:7px 10px;text-decoration:none;border-left:2px solid transparent;color:var(--muted)}nav a:hover{border-color:var(--blue);background:#e7eef6}nav .download{margin-top:16px;padding-top:14px;border-top:1px solid var(--line)}
main{min-width:0}section{background:var(--paper);padding:30px 32px;margin-bottom:22px;border:1px solid var(--line);border-radius:12px;box-shadow:0 3px 14px #16354c04}
h2{font-size:24px;line-height:1.5;margin:0 0 20px;color:var(--navy)}h2 .num{font-size:13px;display:inline-block;vertical-align:3px;margin-right:12px;color:var(--blue)}h3{font-size:18px;margin:26px 0 12px}h4{font-size:16px;margin:20px 0 10px}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}.kpi{padding:19px;background:#edf6f5;border:1px solid #c8e4df;border-radius:9px}.kpi b{font-size:29px;color:var(--teal);display:block;line-height:1.4}.kpi span{font-size:13px;color:var(--muted)}
.callout{border-left:4px solid var(--blue);background:#eef4fc;padding:16px 19px;border-radius:0 7px 7px 0;margin:20px 0}.callout.warn{border-color:#c18920;background:#fff8e8}.callout.good{border-color:var(--teal);background:#eff8f5}.callout p:last-child{margin-bottom:0}
.small,.caption{font-size:13px;color:var(--muted)}.caption{margin:8px 0 18px}.tag{display:inline-block;border-radius:4px;background:#e7f2ec;color:var(--teal);font-size:12px;padding:1px 7px;margin-right:8px}.tag.pending{background:#fff0d2;color:var(--amber)}
.table-wrap{max-width:100%;overflow-x:auto;margin:16px 0}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.65;font-variant-numeric:tabular-nums}th{background:#eef2f7;text-align:left;font-weight:700;color:#2c4562}td,th{padding:10px 11px;border-bottom:1px solid var(--line);vertical-align:top}tbody tr:hover{background:#fafcff}table th:first-child{border-radius:5px 0 0 0}td code{font-size:12px}.gain{color:var(--teal);font-weight:700}.loss{color:var(--red);font-weight:700}.nowrap{white-space:nowrap}
code,pre{font-family:Consolas,"Cascadia Code",monospace;font-size:.88em}code{background:#edf1f6;padding:2px 4px;border-radius:4px;overflow-wrap:anywhere}pre{background:#132a43;color:#e2edf6;padding:20px;border-radius:8px;white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.65}pre code{background:none;padding:0;color:inherit}
ul,ol{padding-left:23px;margin:10px 0 20px}li{padding-left:2px;margin:7px 0}.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}.flow>div{background:#f4f7fb;border:1px solid var(--line);padding:14px;border-radius:7px;font-size:14px}.flow b{display:block;color:var(--blue);margin-bottom:6px}.flow small{color:var(--muted);font-size:12px}
.phases{table-layout:fixed;min-width:600px;text-align:center}.phases th,.phases td{text-align:center;padding:8px 4px}.phases th:first-child{width:104px}.phase{display:block;padding:6px;border-radius:4px;font-weight:700}.real{background:#dff1ea;color:#0c6556}.dummy{background:#edf0f5;color:#667384}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:20px}.equation{padding:17px;text-align:center;border:1px solid var(--line);border-radius:7px;font-weight:700;background:#f8fafc}
details{margin:20px 0;border:1px solid var(--line);border-radius:7px;padding:12px 15px}summary{cursor:pointer;font-weight:700;color:var(--blue)}footer{font-size:12px;color:var(--muted);padding:10px 8px;overflow-wrap:anywhere}
@media(max-width:1050px){.layout{grid-template-columns:1fr;max-width:980px}nav{position:static;display:flex;flex-wrap:wrap;gap:4px}nav strong{width:100%}nav a{border:1px solid var(--line);border-radius:5px;padding:6px 9px}.download{display:none}.hero{padding:36px 24px}.hero h1{font-size:30px}}
@media(max-width:640px){body{font-size:15px}.layout{padding:0 12px;gap:15px;margin:16px auto}section{padding:22px 17px}.hero h1{font-size:25px}.hero p{font-size:16px}.kpis,.flow,.two-col{grid-template-columns:1fr}.kpi b{font-size:25px}h2{font-size:21px}h3{font-size:17px}td,th{padding:8px}table{min-width:600px}nav{font-size:12px}}
@media print{body{background:white;font-size:10.5pt;line-height:1.65}.hero{padding:18px 0;background:white;color:#111}.hero h1{font-size:24pt}.hero p,.meta,.eyebrow{color:#333}.layout{display:block;max-width:none;margin:0;padding:0}nav{display:none}section{box-shadow:none;border:0;border-radius:0;padding:16px 0;margin:0;break-inside:auto}h2,h3{break-after:avoid}h2{font-size:16pt}table{font-size:9pt;min-width:0!important}thead{display:table-header-group}tr,.callout,.kpi,.flow>div{break-inside:avoid}.table-wrap{overflow:visible}.kpis{grid-template-columns:repeat(3,1fr)}pre{background:#f2f4f6;color:#111}details{display:block}a{color:#183d64}.hero,.kpi,.callout,th,.real,.dummy{-webkit-print-color-adjust:exact;print-color-adjust:exact}@page{size:A4;margin:15mm}}
"""
css += diagram_css
parts=[]
def add(s): parts.append(s)
add("""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="MoonEP slot 在 16K 输入、8K token 预算下的性能回退：分词到达时序、KV 回收与 DP 调度相位的因果链，关键路径归因与对齐修复重复验证。">
<title>MoonEP slot / off 性能根因与修复验证 · 2026-10-19</title><style>"""+css+"""</style></head><body>
<header class="hero"><div class="eyebrow">MOONEP · PERFORMANCE INVESTIGATION · FINAL SYNTHESIS</div>
<h1>为何 16K 切成 8K 后 slot 变慢，<br>高倾斜度下对齐后又能获益？</h1>
<p>从分词请求就绪、DP 准入、KV 回收到 MoE 关键路径，解释同一套负载均衡为何出现相反的端到端结果。</p>
<div class="meta">报告日期 2026-10-19 · 实验截至 2026-10-17 · 90.90.93.34 / xzx_moonep_refactor<br>
DeepSeek-V4-Flash · DP4 / TP1 / EP · 仿真高倾斜路由 · 本会话最终证据汇总</div></header>
<div class="layout"><nav aria-label="报告目录"><strong>阅读导航</strong>
<a href="#conclusion">01 结论与收益边界</a><a href="#matrix">02 历史性能矩阵</a><a href="#batch">03 分词如何影响组批</a><a href="#root">04 固定错位的因果链</a><a href="#off">05 为什么 off 常同相</a><a href="#cost">06 多出的时间花在哪里</a><a href="#padding">07 dummy 与 padding</a><a href="#solution">08 修复与后续方案</a><a href="#fig-fix-alignment">↳ 修复前后原理图</a><a href="#patch-usage">↳ patch 使用步骤</a><a href="#validation">09 性能与正确性验证</a><a href="#off-slot-after-fix">↳ 修复后 off / slot 主对照</a><a href="#slot-before-after">↳ slot 自身变化（补充）</a><a href="#evidence">10 证据、复现与版本</a>
<div class="download"><a href="../patch/tokenizer_window.patch">源码补丁</a><a href="../manifest.json">附件 SHA256 清单</a><a href="../EXPERIMENT.md">报告范围与重建方式</a></div></nav><main>
<section id="conclusion"><h2><span class="num">01</span>核心结论</h2>
<p><strong>16K / 8K 的回退来自“到达时序触发错位 + 异步调度维持错位 + 跨 rank 并行度下降”的组合。</strong>
slot 对热点专家的均衡收益一直存在；当真实请求与 dummy 长期交错，更多全局轮次需要承担大 token 模型计算和通信，额外成本超过 GMM 节省，端到端便慢于 off。</p>
<div class="flow">
<div><b>① 分词分批释放请求</b>默认 2ms 聚合窗口把首批 16 个请求切成 1+15；后 15 个约晚 437ms 就绪。<br><small>本工作负载的触发因素。</small></div>
<div><b>② 首批准入差两轮</b>先到请求启动 DP wave，空 rank 先提交 dummy；后来请求不能替换已提交轮次。<br><small>slot 常见初始偏移 [0,2,2,2]。</small></div>
<div><b>③ 三轮节拍锁住错位</b>9GiB KV 与异步输出回收形成 P1、P2、D；2 mod 3 ≠ 0，混合轮持续出现。<br><small>切分改变了相位能否自然重合。</small></div>
</div>
<p>高倾斜度下，只要跨 rank 真实工作集中在同一批轮次，slot 把热点专家的有效路由行分配给更多计算 rank，降低最慢 rank 的 GMM 工作量。这能补偿规划、权重预取及通信的成本。历史 8K/8K、8K/16K、16K/16K 的收益，与修复后 16K/8K 的收益符合这一机制。</p>
<div class="kpis"><div class="kpi"><b>−5.17%</b><span>16K / 8K 对齐后，平均 TTFT<br>off 9.9917s → slot 9.4755s</span></div><div class="kpi"><b>+5.82%</b><span>输入吞吐提升<br>首次占用告警前 3 组配对均获益</span></div><div class="kpi"><b>16 / 16</b><span>20ms 窗口验证组均初始对齐<br>其中 14 组在首次占用告警前</span></div></div>
<div class="callout warn"><p><strong>适用范围：</strong>上述修复收益限定于本模型、3hot1rank 仿真路由、DP4、9GiB KV、32 请求 / 并发 16、输出 1 token 的突发负载。20ms 窗口是已验证的局部时序缓解方案，不保证任意流量全局对齐。跨模式数值等价尚未被本实验认证，详见第 09 节。</p></div>
</section>""")
add("""<section id="matrix"><h2><span class="num">02</span>先把“哪些配置有收益”说清楚</h2>\n<div class="callout"><p><strong>修复后的主要对比请看 <a href="#off-slot-after-fix">9.1：off 与 slot 同条件主对照</a>：</strong>16K/8K、双方 20ms 且对齐，slot 平均 TTFT 降低 5.17%，输入吞吐提高约 5.82%。本节以下保留历史配置矩阵，不能与修复后主样本合并计算。</p></div>
<p>以下 L 是单请求输入长度，B 是每个 DP rank 的每轮调度 token 预算；8K=8192、16K=16384。
B 是上限，不要求每轮凑满。L≤B 只表示预算允许一次处理完整输入，实际仍受队列、KV 和其它约束影响。</p>
<p>历史数据来自 2026-10-10/11 的 32 台服务矩阵，每台服务取 c2/c3 均值、弃 c1。卡组为 0–3；10 月 17 日根因验证使用卡组 4–7。两批数据分别陈列，不跨卡组拼接收益。表内“TTFT 降低”和“吞吐提升”为正表示 slot 更好。</p>""")
add(hist_table([x for x in history["rows"] if x["route"] in ["3hot1r","4hot1r"]]))
add("""<p class="caption">每行均是同一历史矩阵内 off/slot 对照。输入吞吐按各次 total_input_tokens / duration 计算，再取 c2/c3 均值；TTFT 降低 = 1 − slot/off。历史矩阵未逐项补采本次全部时序证据，不能把某一组相位轨迹外推到全部形状。</p>
<div class="callout"><p><strong>问题集中在 L=16K、B=8K：</strong>3hot1r 下 slot TTFT 从 off 的 10.641s 变成 13.185s，增加 23.91%（约 2.544s）；4hot1r 下也增加 10.31%。同样高倾斜下，其余三个组合均有收益。</p></div>
<h3>高倾斜收益与切分回退，为什么能同时成立？</h3>
<p>off 的专家计算仍主要受热点 owner rank 限制；slot 借助额外 expert slot 与权重预取，让其它 rank 承担热点专家的真实路由行。倾斜越高，消除热点 GMM 长尾的收益通常越大；倾斜低时，额外通信、预取和容量操作可能无法被摊薄。历史 uniform 的四种组合均无收益，说明 slot 不是所有负载下都应启用。</p>
<p>16K 不切分的实测稳态为 P→D，两轮初始偏移是完整周期；16K 切两块后为 P1→P2→D，相同两轮偏移不再是完整周期。于是单纯缩小 token 预算，除了增加分块，还改变了各 rank 真实计算能否同轮发生。</p>
<p><strong>两个独立 8K 请求也不等同于一条 16K 请求的两个 chunk。</strong>后者有同一请求的 KV 生命周期、P1→P2 依赖、较长的第二块上下文，以及最后一块完成后才回收整条请求 KV 的约束；独立 8K 请求有各自的准入与完成边界。8K 历史收益不能直接套成“16K 切两次必然等于 2×8K”，也不能由此断言每个 8K 算子变慢。</p>
<details><summary>查看完整 16 组历史配对矩阵</summary>"""+hist_table(history["rows"])+"""</details>
<p class="small">数据："""+link("historical_matrix.json","原始 benchmark 文件路径、SHA256 与重算指标")+"""。历史报告中“切分使固定开销翻倍”的说法不完整，本报告以调度日志和关键路径证据更新解释。</p></section>""")
add("""<section id="batch"><h2><span class="num">03</span>分词如何影响真正送入模型的 batch</h2>
<div class="equation">HTTP 请求 → 分词微批 → token IDs 就绪 → DP rank 选择 → 引擎 WAITING → schedule() → 本轮模型 batch</div>
<p>调度器只能挑选已经完成前端预处理、且已被本 rank 引擎处理的请求。HTTP 同时到达不等于四个引擎同时拥有可调度请求。旧版 AsyncMicrobatchTokenizer 会先在前端聚合文本，批量分词完成后再释放结果；后续请求仍以独立请求进入引擎。</p>""")
add(table(["参数 / 机制","所在阶段","决定什么","不保证什么"],[
["<code>max_batch_size=32</code>","CPU 分词微批","一批最多聚合多少个兼容分词请求","不等于模型每轮 32 个请求"],
["<code>batch_wait_timeout_s=0.002</code>","CPU 分词微批","首项取出后最多收集多久，满批可提前结束","不等于总分词耗时；还要排队和执行"],
["<code>VLLM_TOKENIZER_BATCH_WAIT_MS</code>","本次新增配置入口","把已有等待窗口配置为毫秒；显式构造参数优先","不是 upstream 原有环境变量，也不是 DP barrier"],
["<code>--max-num-seqs</code>","每个 DP 引擎","请求并发/运行序列上限；本版本限制 RUNNING 数","不要求凑满，不是全 DP 的总请求数"],
["<code>--max-num-batched-tokens</code>","每个 DP 引擎每轮","本轮所有请求可调度 token 的总预算","不保证满预算，也不只按请求数分批"]
]))
add("""<p>本版本调度器先考虑已有 RUNNING 请求（包含未完成 prefill 和 decode），再按策略接纳 WAITING 请求；默认 FCFS，还需满足 token 预算、KV 分配与请求状态等约束。最终 <code>num_scheduled_tokens = {request_id: token_count}</code> 才决定模型这一轮包含哪些请求和多少 token。例如 8K 预算可能完全由一条 16K 请求的第一块占满，尽管 <code>max-num-seqs=16</code>。</p>
<h3>本次观测到的分词时间线</h3>""")
add(table(["事件","本次默认 2ms 窗口样本","作用"],[
["首批 16 请求进入分词队列","到达跨度约 11ms","已经超过 2ms 聚合窗口"],
["第一个分词微批","1 个请求，约 40.2ms 完成","首请求先分发并启动 DP wave"],
["第二个分词微批","15 个请求，约 477.5ms 完成","其它 rank 约晚 437ms 才获得真实工作"],
["改为 20ms 聚合窗口","首 16 个请求合为一批完成","随后的快速分发使四 rank 均赶上首轮真实调度"]
]))
add("""<p class="caption">时间为该样本前端跟踪中的相对时间，反映实际端到端分词完成边界；不能把 437ms 解释成纯“多等待了 2ms”，也不能假定每次分词都是同一耗时。</p>
<p>分词微批按完成批次释放结果，因此小的到达差可被放大为较大的请求就绪差。对模型层而言，分词不直接指定一个 batch 的成员，它通过改变各请求赶上哪一次 <code>schedule()</code> 来影响组批。20ms 方案也没有把 16 个请求强制塞进同一个模型 batch。</p>
<p class="small">后续用固定 token IDs 绕过分词、人工引入到达延迟，仍可复现或消除错位：这证明分词是本负载中实际的时序触发来源，调度准入边界才是决定相位的直接机制。</p></section>""")
add("""<section id="root"><h2><span class="num">04</span>固定错位是怎样形成并持续的</h2>"""+chunk_phase_diagram()+kv_gap_diagram()+"""
<h3>第一步：空 rank 已提交的 dummy 不能由迟到请求替换</h3>
<p>首请求被分给一个 rank 后，FIRST_REQ / START_DP_WAVE 让其它 rank 进入共同执行波次。没有真实请求的 rank 必须参与 DP/EP 集合通信，于是执行 dummy。引擎在外层循环处理新输入；当 dummy 调用已提交且尚未返回时，后来的请求只能等待后续调度轮。</p>
<p>slot 的一个直接实测样本中，rank1/2/3 在约 2ms 提交第一轮 dummy，在约 115ms 提交第二轮；它们的真实请求约 418–421ms 才被前端选中分发，约 1.25s 才进入第一轮真实 forward。因此初始真实 forward 偏移为 <code>[0,2,2,2]</code>。不同运行中的具体毫秒值会变，先提交两轮 dummy 的因果顺序不变。</p>
<h3>第二步：KV 回收顺序制造每请求后的空轮</h3>
<ol>
<li>16K 请求 A 的 P1、P2 分别调度 8192 token；最后一块在途时，A 仍占用 KV。</li>
<li><code>step_with_batch_queue()</code> 优先尝试 schedule / submit，然后消费旧输出。A 已知将达到 <code>max_tokens=1</code>，不再需要额外调度；WAITING 中请求 B 却因剩余 KV 不足而被拒绝。</li>
<li>这一轮 schedule 得到 0 token；随后消费 A 的最后输出，A 完成，KV 被释放。</li>
<li>该分支没有在回收后重新调度，<code>model_executed=False</code> 仍保留。波次尚未结束，DP 引擎执行一轮 dummy。</li>
<li>下一次循环 B 才可获得 KV，形成 <strong>P1 → P2 → D</strong> 的稳定节拍。</li>
</ol>
<div class="callout"><p><strong>D 是调度 bookkeeping 空轮转成的 dummy，不是请求额外生成一个业务 token 的 decode。</strong>本样本在 KV 拒绝时剩余 4335 blocks；下一块自身就需要 6722 blocks，完整 16K 请求占用 13444 / 17779 blocks。即使关闭 full-sequence-fit 检查，也不保证消除空轮。block 计数不是 token 数。</p></div>
<p>初次实机两腿共 54 次“有等待请求但空调度”均核对了 <code>KV 拒绝 → 空调度 → KV 释放 → executed=False → dummy → 下一次真实调度</code> 链条。这一节拍由本次 KV 容量、输出长度和异步队列共同形成，不是 vLLM 任意配置必然每三轮插一次 dummy。</p>
<h3>第三步：切分把周期从 2 变为 3，让偏移长期保留</h3>""")
add(table(["场景","观测到的每 rank 稳态","初始偏移","周期关系","结果"],[
["16K 输入 / 16K 预算","P → D","2 轮","2 mod 2 = 0","启动后同相，少量首尾混合"],
["16K 输入 / 8K 预算，slot 常见","P1 → P2 → D","2 轮","2 mod 3 = 2","真实 / dummy 混合长期存在"],
["16K 输入 / 8K 预算，off 常见","P1 → P2 → D","3 轮","3 mod 3 = 0","稳态同相，首尾仍可能混合"],
["16K 输入 / 8K 预算，对齐后","P1 → P2 → D","0 轮","0 mod 3 = 0","全真实轮与全 dummy 轮集中"]
]))
add("""<h4>错位 slot：稳态每三轮都含真实工作</h4>"""+phase_table(False)+"""<h4>对齐 slot：两个全真实轮 + 一个全 dummy 轮</h4>"""+phase_table(True)+"""
<p class="caption">这是稳态相位示意，P1/P2 属于各 rank 各自的连续请求，不表示不同 rank 共享同一个请求。P1/P2 各 8192 真 token；D 的调度真 token 为 0、执行 dummy 的本地 T 为 1。</p>
<p>两种三轮周期处理的真实 chunk 总量相同，区别在于真实工作是否并行集中。错位让各 rank 的 attention、共享专家等本地计算分散在更多全局轮次，跨 rank MoE 通信又要求保持同一 collective 次序。因此 token 总量相同，也会增加端到端关键路径时间。</p>
<p>在 32 请求、每 rank 8 请求的可比 slot 样本中：错位为 <strong>7 全真实 + 18 混合 = 25 个带真实负载的全局轮次</strong>；对齐后为 <strong>16 全真实 + 7 全 dummy</strong>，只有 16 轮带真实负载。真实调度 token 均为 524288，没有通过少算或缓存命中获得收益。</p>
<p class="small">证据："""+link("initial_scheduler_validation/final_analysis.txt","逐请求 KV / dummy 因果链与源码行号")+"""，"""+link(G+"validation_report.txt","同服务自然对齐样本")+"""。</p></section>""")
add("""<section id="off"><h2><span class="num">05</span>为什么 slot 高频错位，off 常规运行却较少表现出来</h2>
<p><strong>off 没有额外的“请求阶段自动对齐保证”。</strong>两种模式都可能有 dummy，也都受同一前端分词就绪差影响。差别在于 worker 提交、同步与背压的实际耗时不同，迟到请求赶上了不同的轮次边界。</p>""")
add(table(["直接观测","warm slot","warm off"],[
["首个 execute_model 主机方法墙钟耗时","约 494.522ms","约 70–80ms"],
["迟到请求抵达前，空 rank 已进入的关键 dummy","第 2 轮约 96ms 开始，约 1226.5ms 返回","第 2 轮约 134ms 返回，第 3 轮约 135ms 已提交"],
["后续 rank 请求 frontend 到达","约 435–437ms","约 434ms"],
["首个真实 forward","第 3 轮，即偏移 2","第 4 轮，即偏移 3"],
["与 P1/P2/D 三轮周期的关系","非整周期 → 稳态错位","整周期 → 稳态同相"]
]))
add("""<p class="caption">这组时间线以首个 frontend_dispatch 为 t=0。execute_model 耗时是主机方法墙钟时间，包含提交、同步、排队和背压，不能称为纯 CPU 或某一个 NPU 算子的耗时。</p>
<p>所以，常见 400ms 左右的分词分批间隔，在本实现 warm slot 中容易落入“偏移 2”的时间窗口，在 warm off 中容易落入“偏移 3”的窗口。它是实际时间边界与周期的组合，不是某个 rank 被永久指定领先，也不是 off 本身没有错位风险。</p>
<h3>反事实实验：绕过分词，仅改变请求到达时序</h3>
<p>固定输入 token IDs、显式 rank=i%4、greedy、seed=23001；每组 32 请求全部完成，每 rank 8 请求，真实 token 总量相同。</p>""")
add(table(["客户端设定延迟","slot TTFT / 首轮偏移","off TTFT / 首轮偏移"],[
["0ms","9.2681s / [0,0,0,0]","9.8287s / [0,0,0,0]"],
["20ms","未跑","14.4310s / [0,2,2,2]"],
["60ms","13.1660s / [0,2,2,2]","14.3967s / [0,2,2,2]"],
["150ms","13.1478s / [0,2,2,2]","10.8443s / [0,3,3,3]"],
["400ms","13.0549s / [0,2,2,2]","10.6614s / [0,3,3,3]"]
]))
add("""<p class="caption">设定延迟不是精确的前端观测间隔：off 20ms 组实际约 58.425ms，60ms 组 slot/off 实际约 99.594 / 100.889ms。该表用于因果验证，不与后面的主性能配对混算。</p>
<p>当 off 也被触发为偏移 2 时，TTFT 同样退化到约 14.4s；新进程 off 首次 execute_model 约 463ms 的样本也产生过偏移 2。这直接排除了“off 天然不会错位”的解释。</p>
<h3>自然对齐那一次，究竟触发了什么？</h3>
<p>同一个 slot 服务的自然对齐组，前四次 frontend_dispatch 相对首请求仅差 <code>[0,0.912,1.637,2.281]ms</code>，四个 rank 都及时拿到真实请求；下一簇 12 个请求约在 352ms 后。TTFT 从前一错位组 13.1971s 降为 9.2800s，稳态周期从 3.9816s 降为 2.8355s。</p>
<p>这一次没有记录分词微批边界，因此只能确认“首四请求赶上共同首轮”是直接条件，无法倒推 OS / 事件循环为何恰好形成该簇，也没有证据证明 <code>stop_profile</code> 本身修复了调度。后续聚合窗口回退恢复与人工到达实验，才把触发机制变成可重复证据。</p></section>""")
add("""<section id="cost"><h2><span class="num">06</span>slot 多出的约 2.54 秒，实际花在哪里</h2>
<p>历史 clean 样本差为 2.5438s；后续同卡组轻量 profile 的客户端差为 <strong>2.5062s</strong>。以下分账精确对应后者，不能把不同运行的数字硬拼成历史 2.5438s。采样条件为 NPU-only / Level1，无 CPU capture、stack、memory 或 AIC metrics。</p>
<div class="callout"><p><strong>计算口径：</strong>每层取四 rank 对应阶段的最晚完成边界，组成不重叠的全局时间区间；每个区间乘以当时未完成请求数 / 32，得到对平均服务端 TTFT 的贡献。每个时间区间只记一次，不把各 rank 等待相加，也不把算子时长总和当成客户端延迟。</p></div>"""+stage+"""
<p class="caption">单位：秒 / 平均请求；正差代表 slot 更慢。通信项包含实际传输及暴露在关键路径上的协议同步，不是纯网络传输时间。off 的 planning / prefetch 填 0 代表没有这些 MoonEP 专属阶段，并非 off 完全没有路由准备。</p>
<div class="equation">阶段差 +2.537818s − 首尾覆盖差 0.002091s − 客户端/服务端边界差 0.029563s<br>= 客户端 TTFT 差 +2.506164s</div>
<h3>如何读这个分账</h3>
<ul>
<li><strong>GMM 已经获益：</strong>slot 节省约 2.167s。不能说“slot 均衡算法把专家 GMM 算得更慢”。</li>
<li><strong>主要不利项：</strong>dispatch/combine 多约 1.981s，combine 前容量张量处理多约 0.930s，其它模型计算与轮次间隔多约 1.747s。错位把真实工作分散到更多串行全局轮次，是这些成本暴露更多的关键原因。</li>
<li><strong>planning 不是秒级主因：</strong>四 rank 原始 planning 时长相加约 30.243s，但最后 rank 到齐后的全局规划区间累计约 0.0895s；对平均请求的关键路径贡献只有 0.0268s。长 planning 等待说明某些 rank 提前到集合点，不能再把同一时间其它 rank 的计算重复计费。</li>
<li><strong>预取是次要项：</strong>约 0.254s 的暴露成本，值得优化，但不能解释整个差额。</li>
</ul>
<p>combine 准备中实际 device 工作主要为 Mul 0.377s、Cast 0.363s、SelectV2 0.229s、TensorMove 0.235s，说明这里有真实的整容量读写。其它模型项包括 attention、投影、norm、HC、共享专家、router 与少量轮次边界，不等于全是 CPU 空转。</p>
<p>同形状、同 P1/P2 的 attention 对比中，slot 均值约 3.349 / 3.718ms，off 约 3.421 / 3.829ms。没有证据表明 slot 把同一个 attention kernel 拖慢；总关键路径贡献增加主要与真实工作分散到更多轮次有关。这里也没有通过消融证明 polling 资源竞争的独立份额。</p>
<div class="callout warn"><p><strong>对早期判断的更新：</strong>“planning 等待总量增长”是错位的症状，不能直接当成根因或全部性能差额。只缓存 plan、减少 barrier 次数，并不能据此保证消除本次回退；先修复准入时序并验证相位，证据更直接。</p></div>
<p class="small">数据："""+link(G+"stage_comparison.csv","关键路径阶段 CSV")+""" · """+link(G+"comparison_summary.json","采样边界核算与 clean 对照")+"""。off 在连续运行中曾伴随温度/频率变化变慢；最终收益另用同样冷却条件的三组配对，不由本 profile 表外推。</p></section>""")
add("""<section id="padding"><h2><span class="num">07</span>slot 均衡的是哪些 token？off 有 dummy 为什么仍较快？</h2>
<p>需要区分三个量：<strong>业务真实 token、dummy forward 自身的 1 个输入 token、为容量 / 通信对齐预留的 padding</strong>。本次 slot 启用 <code>MOONEP_REAL_T=1</code>；T 是本地有效输入行数，S 是容量档位，K=6。</p>""")
add(table(["环节","slot：本次 real-T 路径","off：本次大 token / mixed 路径"],[
["参与路由 / 均衡的输入","按真实 T 构建直方图；dummy 本身 T=1，也有 6 条有效 route；S−T 尾部不进入有效负载","实际走 DP ALLGATHER：先 pad 到跨 rank 最大 T，再 allgather；本模拟路由在 MoE 前不 unpad"],
["GMM","按 owned + slot 的实际 group_list 有效行数计算；可接收其它 rank 的真实 token","mixed padding 在本模拟路由下也会进入专家路由 / GMM"],
["容量操作","NO_SYNC 保留 S×K 的容量张量，copy、量化、激活、乘权、转换、where 清尾、搬运仍可能处理尾部","mixed 轮负担大；全 dummy 时实际切换为便宜 MC2 路径"],
["dummy rank 有大 GMM 是否代表算 padding？","不一定：它可作为 owner 或 slot 承担来自其它 rank 的真实专家计算","owner rank 也可能承担其它 rank 的真实请求计算"]
]))
add("""<pre><code>混合轮：本地有效 T = [8192, 1, 1, 1]，K = 6
slot 有效路由行总量 = (8192 + 1 + 1 + 1) × 6 = 49,170
不是按全容量计算的 4 × 8192 × 6 = 196,608

slot 理想 fair-share 基准 = floor(49,170 / 4) = 12,292 行
实际分配还受专家 / slot 约束，不保证精确等分。
单 rank 容量 NvS = 8192 × 6 = 49,152，hidden size H = 4096。</code></pre>
<p>因此，本次 slot 并没有把 8K 容量尾部全部当作“有效专家负载”去均衡；但 padding 的逐元素 / 内存成本仍在。旧 <code>MOONEP_REAL_T=0</code> 路径可能把 padding 带入有效路由，不能与本次路径混为一谈。</p>
<p>对 off，trace 在 <code>[8192,1,1,1]</code> mixed 轮看到 MoeInitRoutingV3 输入 <code>32768×4096</code> / <code>32768×6</code>，GMM 容量 196608 行；其代价接近全真实轮。全 dummy 轮则只有 24 行 GMM 输入，profile 中每层 MoE 约 0.541ms，其它模型部分约 1.074ms，便宜得多。</p>
<p><strong>关键不是 dummy 的总个数，而是它们是否聚集在同一全局轮。</strong>通常 off 的稳态为两轮全真实 + 一轮全 dummy；错位 slot 每轮都有真实 8K 工作。off 可拥有更多 dummy forward 却更快；若人为让 off 也长期 mixed，它同样退化。</p>
<p class="small">以上 backend、padding 和 GMM 行为限于本次 TP1 / DP4、模型、路由和代码版本。不能泛称所有 off 场景都使用 ALLGATHER，或所有 padding 都会进入 GMM。</p></section>""")
add("""<section id="solution"><h2><span class="num">08</span>解决方案：已验证的最小改动，与后续优化</h2>"""+fix_diagram()+"""
<h3><span class="tag">已实施并验证</span>将既有分词聚合窗口暴露为配置</h3>
<p>保留旧默认值 2ms，只新增 <code>VLLM_TOKENIZER_BATCH_WAIT_MS</code> 配置入口；本负载设为 20ms，重启 API 服务，使首批请求在一个微批中完成分词，再快速分配给四个 rank。它改变请求就绪时序，既不改变 NPU 算子，也不改变 MoE 路由或 KV 容量。</p>""")
add(table(["文件","实际变更"],[
["<code>vllm/vllm/envs.py</code>","新增浮点毫秒环境变量，默认 2"],
["<code>vllm/vllm/utils/async_utils.py</code>","构造参数未显式指定时读取环境值并换算为秒；显式值（包括 0）优先；拒绝负值和非有限值"],
["<code>vllm/tests/utils_/test_async_tokenizer_window.py</code>","聚合、单请求、decode、取消、参数优先级和非法值等测试"]
]))
add(patch_usage_html())
add("""<p>修改影响该 wrapper 的 encode 和 decode 微批等待，不能假定每个输出 detokenization 路径都经过它。等待窗口是有上限的，不要求收齐四个 rank 或必须凑满 32 个请求；单请求、3 请求场景均已成功验证。</p>
<div class="callout warn"><p>20ms 相比 2ms，低并发文本请求可能多约 18ms 聚合等待。5ms、10ms 探测也曾对齐，但 5ms 出现 9/8/8/7 的 rank 请求分配及尾部混合；20ms 为本次约 11–14ms 的突发队列跨度留出余量。它是已重复验证的取值，不是数学意义上最小毫秒数。若请求本身相隔几十至几百 ms，仍可能错位。</p></div>
<h3><span class="tag pending">尚未实施 NPU 完整验证</span>更一般的调度与算子方案</h3>""")
add(table(["优先方向","预期解决的问题","需要额外验证"],[
["输出回收后重新评估调度","减少“KV 已释放、本轮仍 dummy”的 bookkeeping 空轮","异步 batch queue 容量、结构化输出、取消/抢占、DP/EP collective 顺序；不能直接跳过某一 rank 的 dummy"],
["适度增加 KV 配额","让下一请求在旧输出回收前也能准入","以实际 hybrid KV block 需求确认；CPU replay 支持机制，但尚无本任务同条件 NPU 收益结论"],
["有界的首波准入协调","降低任意分批请求触发相位偏差的概率","稀疏流量尾延迟、超时、单/少请求、不同 DP rank 状态；本次未实现全局 barrier"],
["融合 combine 容量操作 / 降低无效尾部读写","减少已测得的 Mul / Cast / where / TensorMove 成本","有效行掩码、异步 shape、尾部正确性；不能简单裁剪未知布局"],
["定点优化 dispatch / combine 协议","减少关键路径通信与同步暴露","分别测传输和协议等待，按全局关键路径评估"],
["planning / prefetch 后续消融","优化剩余小项","先证明可复用 plan 的请求和路由语义；当前 planning 仅约 27ms / 平均请求"]
]))
add("""<p>提高 B 到 16K 可避开本例两块 prefill 的三轮节拍，历史结果有收益；但会改变单步开销、内存和调度行为，是有代价的配置选择，不代替根因修复。关闭异步调度、增 KV 的 CPU replay 曾消除空轮，仅作为机制证据，尚不能宣称已获得生产性能收益。</p>
<p class="small">补丁："""+patch_link("tokenizer_window.patch")+""" · """+link(F+"patch_source/async_utils.py","修改后的分词实现快照")+""" · """+link(F+"tokenizer_window_tests.log","测试日志")+"""。本报告仅归档已经验证的补丁，没有重新应用补丁或启动服务。</p></section>""")
add("""<section id="validation"><h2><span class="num">09</span>修复后 slot 相对 off 的性能收益与正确性</h2>"""+render_primary_comparison(table)+"""
<p>20ms 窗口下共 16 组初始对齐，覆盖 slot 两次服务启动、2→20ms 回退恢复与 off 对照；其中 14 组在各自服务第一次占用告警前。完整分析阶段累计 39 组、1104 请求、0 失败，其中包含诊断、正确性和告警后补充样本，不全部作为独占性能样本。</p>
"""+render_slot_comparison(table)+"""
<h3>9.3 正确性：已通过的部分与未证明的部分</h3>""")
add(table(["检查","结果","结论范围"],[
["同模式、同 32 个文本输入，默认 vs 20ms vs 重启版本","greedy 生成文本 32/32 相同","窗口补丁前后输出一致"],
["固定 token IDs，改变到达延迟","slot 0/60/150/400ms 各 32/32 相同；off 各延迟组亦一致","对齐 / 错位本身未改变这些同模式输出"],
["多 token 输出","slot 4 请求 × 8 输出 token，对齐 / 错位 / 重启版本逐 token 一致","补充覆盖多 token 场景"],
["实际模型 tokenizer 编码","32 个长输入，逐条 vs 20ms 聚合，共 532766 token IDs 全部相同","分词内容未改变；不同于性能组固定 524288 token"],
["单测与静态检查","8 项测试、Ruff / format、git diff --check 通过","补丁行为与格式检查"],
["slot vs off 跨模式","仅 24/32 输出 token 相同，8 个不同","不能声称跨模式数值等价"]
]))
add("""<div class="callout warn"><p><strong>跨模式精度限制需要保留：</strong>仿真路由 <code>moe_sim_route._assign</code> 使用 <code>randperm(T)</code> 生成专家表。slot 真实 T=8192，off ALLGATHER 后 T=32768；同 seed 和同分布不保证同一业务 token 选中相同专家。执行原始路由函数的 CPU 验证中，每 rank 8192 行的完整 6 专家选择仅 18 / 27 / 35 / 30 行跨模式一致。因此这组仿真输出不能作为跨模式数值等价 oracle；该差异在窗口补丁前已存在，同时也不能排除其它数值差异。</p></div>
<p>要验证真实模型质量或 slot/off 数值等价，下一步应使用真实路由，或把模拟路由固定到全局 token 身份，使两模式逐 token 专家选择一致，再进行多请求、多输出长度的逐 token / logit 对照。本报告没有把性能验证代替模型质量验证。</p>
<h3>9.4 测量可信度与限制</h3>
<ul>
<li>每卡保留 16GiB 的占位进程覆盖服务切换，并采样进程、温度与频率；这不是驱动级排他锁。</li>
<li>最终 slot 第二次启动首次外来进程告警为 22:16:21；主三组性能样本均在 22:13:36 前结束，且有前后无告警采样。告警后第 4 对只作为补充，不纳入主均值。</li>
<li>NPU 频率仅采样、没有锁频；部分 off 活跃采样为 1400–1600MHz，部分组为 1650MHz。起始温度控制改善可比性，不能据此把 5.17% 全归到某个单独算子或排除所有硬件状态影响。</li>
<li>历史与本次卡组、采样模式及调度状态分开统计；使用新 seed / cache_salt，按逐轮 token 总量核对缓存混淆。实验已于 10 月 17 日 22:31 清理自有服务和占位，本次仅生成报告。</li>
</ul>
<p class="small">数据："""+link(F+"summary.json","最终配对统计")+""" · """+link(F+"all_groups.csv","全部实验组索引")+""" · """+link(F+"correctness.json","正确性检查")+""" · """+link(F+"sim_route_shape_check.json","仿真路由 shape 检查")+""" · """+link(F+"occupancy_quality.json","样本占用质量")+"""。</p></section>""")
add("""<section id="evidence"><h2><span class="num">10</span>证据索引、版本与复现</h2>
<p>报告按“一个实验一个目录”的服务器约定归档。本目录提供离线可读 HTML、重建脚本、机器统计、补丁、关键原始日志副本与 SHA256 清单。HTML 不依赖外部脚本、字体、图片或 CDN；单文件均小于 1MB。</p>
<pre><code>/home/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/
└── 20261019_moonep_slot_root_cause/
    ├── EXPERIMENT.md
    ├── RUNS.txt
    ├── gen_report.py
    ├── report_enhancements.py
    ├── slot_comparison.py
    ├── patch/
    │   ├── tokenizer_window.patch
    │   ├── README.html
    │   └── README.txt
    ├── extract_history.py
    ├── manifest.json
    ├── data/
    │   ├── historical_matrix.json
    │   ├── alignment_fix_validation_20261017/
    │   ├── guarded_validation_20261017/
    │   └── initial_scheduler_validation/
    └── report/
        ├── assets/  # 三张可单独查看的 SVG 图
        └── moonep_slot_vs_off_root_cause.html</code></pre>""")
add(table(["结论","直接证据"],[
["历史 8K / 16K × 8K / 16K 矩阵",link("historical_matrix.json","64 个 c2/c3 原始 JSON 引用与 SHA256")],
["KV 拒绝到 dummy 的因果链",link("initial_scheduler_validation/final_analysis.txt","初次实机验证报告及原始日志行号")],
["关键路径分账",link(G+"comparison_summary.json","最晚边界与服务端 / 客户端差核算")+"；"+link(G+"stage_comparison.csv","阶段 CSV")],
["分词、到达、worker 三段时间线",link(F+"slot_arrival_20261017_a_causes.json.gz","slot 首次服务 causes")+"；"+link(F+"off_arrival_20261017_a_causes.json.gz","off causes")],
["完整逐轮 request ID / token / dummy 矩阵",link(F+"slot_arrival_20261017_a_arrival.json.gz","slot arrival")+"；"+link(F+"off_arrival_20261017_a_arrival.json.gz","off arrival")],
["最终重复验证与补丁",link(F+"validation_report.txt","最终实验文本记录")+"；"+patch_link("可审阅补丁")+"；"+link(F+"reproduce.txt","复现命令")],
["原始大文件及 kernel 保留位置",link(F+"raw_evidence_manifest.json","原始逐轮 trace 清单")+"；"+link(G+"raw_kernel_manifest.json","原始 kernel 清单")],
["完整样本、运行源码与环境快照",link(F+"all_groups.csv","all_groups.csv")+"；"+link(F+"source_provenance.json","source_provenance.json")],
["最终资源清理",link(F+"cleanup_verification.json","cleanup_verification.json")]
]))
add("""<p>原始完整实验目录为 <code>/home/x00893896/vllm_ascend_moonep_integration/scripts/20261017_scheduler_alignment</code>；
大型 trace / kernel CSV 仍按原始清单留在服务器，未嵌入 HTML。历史性能原始 JSON 位于同级旧实验 <code>20261011_opt2_off_vs_slot/data</code>。</p>
<h3>代码定位</h3>""")
add(table(["组件 / 版本","关键位置与职责"],[
["vLLM <code>0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665</code>","<code>v1/engine/core_client.py</code>：DP rank 选择、FIRST_REQ；<code>v1/engine/core.py:484</code>：先 schedule、再消费输出；DP busy loop：空调度转 dummy"],
["vLLM scheduler / KV","<code>v1/core/sched/scheduler.py:381</code>：输出上限；<code>:761</code>：分配 KV；<code>v1/core/kv_cache_manager.py:354</code>：完整输入空间检查"],
["vLLM 分词","<code>utils/async_utils.py</code>：AsyncMicrobatchTokenizer；<code>envs.py</code>：本次新增配置。日志 / 行号以归档快照为准"],
["vLLM Ascend <code>f4d27236969c0f4ae0754b1e553ff1be9b4a3663</code>","model runner 的 DP 同步、MoE prepare / finalize 与 MoonEP 通信路径"],
["Ascend MoonEP <code>ceabdc2ec2d9559a7757be266456ebb061306066</code>","planning 使用有效 T，按实际 group_list 执行 slot GMM；精确源码摘录见 current_code_evidence.txt"]
]))
add("""<p class="small">这是基准 commit 加工作区补丁的运行环境，不能仅用 commit 号声称完全干净。既有 catlass / shmem 子模块 dirty 状态被保留；本次相关 vLLM 补丁与测试另存于附件。本报告不把该环境变量当作所有 vLLM 版本通用参数，升级版本后须重新确认分词实现与配置入口。</p>
<h3>复现与后续回归的验收口径</h3>
<ol>
<li>重建本 HTML：在本报告根目录运行 <code>python3 gen_report.py</code>，仅使用已归档数据，不启动模型、不占卡。</li>
<li>实验级复现入口见 data 中 <code>reproduce.txt</code> 与 <code>run_arrival_service.sh</code> / <code>run_arrival_bench.sh</code>；维持 DP4 / TP1 / EP、9GiB KV、L16384 / B8192、n32 / c16 / out1 与相同 slot 配置。每次使用独立运行目录和新 cache_salt / seed。</li>
<li>先验证每请求 16384 真 token、总量 524288、每 rank 8 请求、初始偏移及全真实 / mixed / 全 dummy 轮数，再看 TTFT 和吞吐。不要只看 aggregate GPU/NPU 利用率或各 rank 等待总和。</li>
<li>正式性能至少重复配对，并控制日志、采样、缓存、温度与其它占用；恢复 2ms、再设 20ms 进行回退恢复，补充固定 token IDs 的到达延迟对照。</li>
<li>正确性至少含同模式补丁前后、单/少请求、取消、多 token 输出；跨模式另需固定逐 token 路由或使用真实路由评估。</li>
</ol>
<p class="small">附件保留各阶段原始记录，其中“尚待验证”等文字代表当时状态；最终判断以本 HTML 及 alignment_fix_validation_20261017 的最终结果为准。没有用早期未完成项覆盖后续已完成的对照证据。</p></section>
<footer>生成于 2026-10-19（Asia/Shanghai） · 本会话证据汇总 · 主结论：时序触发的调度相位错位掩盖了有效的专家计算均衡收益。<br>
报告重建：gen_report.py · 数据校验：manifest.json · 不含提示词正文或鉴权信息。</footer>
</main></div></body></html>""")
out=ROOT/"report"/"moonep_slot_vs_off_root_cause.html"
out.parent.mkdir(exist_ok=True)
out.write_text("\n".join(parts),encoding="utf-8")
usage_body=patch_usage_html()
usage_html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MoonEP tokenizer patch 使用说明</title><style>'+css+'</style></head><body><main style="max-width:1000px;margin:24px auto;padding:24px;background:white"><h1>MoonEP tokenizer patch 使用说明</h1><p><a href="../report/moonep_slot_vs_off_root_cause.html#solution">返回根因分析报告</a></p>'+usage_body+'</main></body></html>'
(ROOT/"patch"/"README.html").write_text(usage_html,encoding="utf-8")
class UsageText(HTMLParser):
 def __init__(self):
  super().__init__(convert_charrefs=True)
  self.parts=[]
 def handle_data(self,data):
  self.parts.append(data)
 def handle_starttag(self,tag,attrs):
  if tag=="br":self.parts.append("\n")
 def handle_endtag(self,tag):
  if tag in {"p","h3","h4","pre","div"}:self.parts.append("\n\n")
usage_text=UsageText()
usage_text.feed(usage_body)
(ROOT/"patch"/"README.txt").write_text("MoonEP tokenizer patch 使用说明\n\n"+"".join(usage_text.parts),encoding="utf-8")
files=[]
for p in sorted(ROOT.rglob("*")):
 if p.is_file() and p!=ROOT/"manifest.json" and not any(x in {"__pycache__","qa"} for x in p.parts):
  files.append({"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
assert all(x["bytes"]<1_000_000 for x in files)
(ROOT/"manifest.json").write_text(json.dumps({"report_date":"2026-10-19","host":"90.90.93.34","container":"xzx_moonep_refactor","files":files},ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"report":str(out),"bytes":out.stat().st_size,"files":len(files),"maximum_file_bytes":max(x["bytes"] for x in files)},ensure_ascii=False))
