"""Derive slot-before/after tables directly from archived benchmark JSON."""
from pathlib import Path
import json,csv,statistics,hashlib,html
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"alignment_fix_validation_20261017"
def render_slot_comparison(table):
 label="slot_arrival_20261017_a"
 root=DATA/"runs"/label
 index={r["tag"]:r for r in csv.DictReader((DATA/"all_groups.csv").open(encoding="utf-8",newline="")) if r["label"]==label}
 before=["baseline1","baseline2","baseline3_retry"]
 after=["wait20_1","wait20_2","wait20_3","wait20_4"]
 extra=["greedy_base","greedy_fixed"]
 rows={}
 evidence=[]
 for tag in before+after+extra:
  p=root/(tag+".json")
  b=json.loads(p.read_text(encoding="utf-8"))
  m=json.loads((root/(tag+".meta.json")).read_text(encoding="utf-8"))
  r=index[tag]
  assert b["completed"]==32 and b["failed"]==0
  assert b["total_input_tokens"]==524288 and int(r["real_tokens"])==524288
  assert abs(float(r["ttft_ms"])-b["mean_ttft_ms"])<1e-6
  assert sorted(json.loads(r["rank_counts"]).values())==[8,8,8,8]
  expected=2.0 if tag in before+["greedy_base"] else 20.0
  assert m["control"]["tokenizer_wait_ms"]==expected
  rows[tag]=dict(tag=tag,seed=m["seed"],window_ms=expected,ttft_s=b["mean_ttft_ms"]/1000,duration_s=b["duration"],
    input_tps=b["total_input_tokens"]/b["duration"],cycle_s=float(r["cycle_s"]),
    A=int(r["A"]),M=int(r["M"]),D=int(r["D"]),offsets=json.loads(r["offsets"]),
    completed=b["completed"],failed=b["failed"],total_input_tokens=b["total_input_tokens"])
  for f in [p,root/(tag+".meta.json")]:
   evidence.append(dict(path=f.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
 def means(tags):
  return {**{k:statistics.mean(rows[t][k] for t in tags) for k in ["ttft_s","input_tps","cycle_s","duration_s"]},"n":len(tags)}
 pre,post=means(before),means(after)
 changes=dict(ttft_reduction_pct=(1-post["ttft_s"]/pre["ttft_s"])*100,input_tps_gain_pct=(post["input_tps"]/pre["input_tps"]-1)*100,cycle_reduction_pct=(1-post["cycle_s"]/pre["cycle_s"])*100)
 same=dict(ttft_reduction_pct=(1-rows["greedy_fixed"]["ttft_s"]/rows["greedy_base"]["ttft_s"])*100,
  input_tps_gain_pct=(rows["greedy_fixed"]["input_tps"]/rows["greedy_base"]["input_tps"]-1)*100)
 result=dict(label=label,scope="slot; 3hot1r; DP4 TP1 EP; L16384 B8192; KV9GiB; n32 c16 out1",before_tags=before,after_tags=after,before=pre,after=post,changes=changes,rows=rows,same_seed_check=same,evidence=evidence,
  method="Same running slot service; runtime tokenizer instance switched between 2ms and 20ms. Group means are unpaired (different seeds, n=3 vs n=4). The env-based source patch was separately validated after restart.")
 (ROOT/"data"/"slot_before_after.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
 summary=table(["slot 状态","窗口 / 组数","平均 TTFT / s","输入吞吐 / token·s⁻¹","稳态周期 / s","全真实 / 混合 / 全 dummy"],[
 ["方案启用前","2ms / 3 组",f'{pre["ttft_s"]:.4f}',f'{pre["input_tps"]:,.2f}',f'{pre["cycle_s"]:.4f}',"每组 7 / 18 / 0"],
 ["方案启用后","20ms / 4 组",f'{post["ttft_s"]:.4f}',f'{post["input_tps"]:,.2f}',f'{post["cycle_s"]:.4f}',"每组 16 / 0 / 7"],
 ["变化","同一 slot 服务",f'<strong class="gain">−{changes["ttft_reduction_pct"]:.2f}%</strong>',f'<strong class="gain">+{changes["input_tps_gain_pct"]:.2f}%</strong>',f'<strong class="gain">−{changes["cycle_reduction_pct"]:.2f}%</strong>',"混合轮归零"]])
 detail=table(["状态 / 组名","seed","窗口","TTFT / s","整组 bench / s","输入吞吐 / token·s⁻¹","首轮偏移"],[
 [html.escape(t),rows[t]["seed"],f'{rows[t]["window_ms"]:.0f}ms',f'{rows[t]["ttft_s"]:.4f}',f'{rows[t]["duration_s"]:.4f}',f'{rows[t]["input_tps"]:,.2f}',"<code>"+str(rows[t]["offsets"])+"</code>"] for t in before+after])
 paired=table(["同 seed 补充组","2ms 窗口","20ms 窗口","变化"],[
 ["平均 TTFT",f'{rows["greedy_base"]["ttft_s"]:.4f}s',f'{rows["greedy_fixed"]["ttft_s"]:.4f}s',f'<strong class="gain">降低 {same["ttft_reduction_pct"]:.2f}%</strong>'],
 ["输入吞吐",f'{rows["greedy_base"]["input_tps"]:,.2f} token/s',f'{rows["greedy_fixed"]["input_tps"]:,.2f} token/s',f'<strong class="gain">提高 {same["input_tps_gain_pct"]:.2f}%</strong>']])
 return '''<div id="slot-before-after"><h3>9.2 补充：slot 自身修复前后的变化</h3>
<p>固定条件：<strong>3hot1rank 高倾斜路由，L=16384、B=8192，DP4 / TP1 / EP，KV=9GiB，32 请求、并发 16、输出 1 token</strong>。下表全部来自同一个 <code>slot_arrival_20261017_a</code> 服务；每组 32/32 成功、真实 token=524288、每 rank 8 请求，该服务占用监控无告警。</p>
'''+summary+f'''
<div class="callout good"><p><strong>slot 自身的回退被消除：</strong>默认窗口均值 {pre["ttft_s"]:.4f}s → 20ms 窗口 {post["ttft_s"]:.4f}s，平均 TTFT 降低 <strong>{changes["ttft_reduction_pct"]:.2f}%</strong>，输入吞吐提高 <strong>{changes["input_tps_gain_pct"]:.2f}%</strong>。这里比较 slot 与自身；前面第 9.1 节主对照的 5.17% / 5.82% 则比较双方都对齐后的 slot 与 off，分母不同。</p></div>
<p><strong>统计口径：</strong>前 3 组和后 4 组分别取算术均值；输入吞吐先按每组 <code>total_input_tokens / duration</code> 计算再平均。它们使用不同 seed，属于同服务的窗口切换重复验证，不能称为同 seed 的逐组配对。<code>baseline3</code> 的实验包装中断，沿用原有效性筛选，使用完整的 <code>baseline3_retry</code>。3 组默认窗口中，baseline2 的偏移为 [0,0,2,2]，其余为 [0,2,2,2]；均产生 18 个混合轮。</p>
<p><strong>“修复后”的含义：</strong>该组在运行中的分词实例上把已有窗口改为 20ms，用于确认方案的因果效果；交付的 patch 把这个窗口暴露为环境变量。通过环境变量启动的新 slot 服务已另外重复验证：主三组平均 TTFT 9.4755s，与这里的 9.4706s 接近。这不是重新采集的新实验。</p>
<h4>逐组原始 bench 指标</h4>
'''+detail+'''
<p class="caption">表按 2ms / 20ms 分类展示，实际运行包含回退与恢复，并非先跑完全部基线再跑全部修复组。温度 / 频率采样、未锁频等限制仍适用。这里只汇总当前问题对应的 3hot1rank、16K/8K；未声称将历史四种输入 / 预算组合全部重跑了补丁前后对照。</p>
<h4>同 seed 的单组补充检查</h4>
<p><code>greedy_base</code> 与 <code>greedy_fixed</code> 均 seed=22010、temperature=0；32 个请求输出全部一致。它们用于正确性交叉检查，未混入上面的主要性能均值。</p>
'''+paired+'''
<p class="small">数据：<a href="../data/slot_before_after.json">重算结果、统计范围与源文件 SHA256</a> · <a href="../data/alignment_fix_validation_20261017/all_groups.csv">全部逐组相位与 token 对账</a>。</p></div>'''


def render_primary_comparison(table):
 """Recompute the matched, aligned off/slot comparison from raw bench files."""
 source=json.loads((DATA/"summary.json").read_text(encoding="utf-8"))
 index={(r["label"],r["tag"]):r for r in csv.DictReader((DATA/"all_groups.csv").open(encoding="utf-8",newline=""))}
 labels={"off":"off_arrival_20261017_a","slot":"slot_arrival_20261017_b"}
 pairs=[]
 evidence=[]
 for pair in source["primary"]["pairs"]:
  item={"seed":pair["seed"]}
  for mode,label in labels.items():
   tag=pair[mode+"_tag"]
   path=DATA/"runs"/label/(tag+".json")
   meta_path=path.with_name(tag+".meta.json")
   bench=json.loads(path.read_text(encoding="utf-8"))
   meta=json.loads(meta_path.read_text(encoding="utf-8"))
   phase=index[(label,tag)]
   quality=source["occupancy_quality"][label]["groups"][tag]
   assert meta["seed"]==pair["seed"]==int(phase["seed"])
   assert bench["completed"]==32 and bench["failed"]==0
   assert bench["total_input_tokens"]==int(phase["real_tokens"])==524288
   assert bench["input_lens"]==[16384]*32 and bench["output_lens"]==[1]*32
   assert json.loads(phase["offsets"])==[0,0,0,0]
   assert sorted(json.loads(phase["rank_counts"]).values())==[8]*4
   assert [int(phase[k]) for k in ["A","M","D"]]==[16,0,7]
   assert quality["before_first_alert"] and quality["bracketing_samples_present"] and not quality["alert_samples"]
   assert abs(bench["mean_ttft_ms"]-pair[mode+"_ms"])<1e-8
   assert abs(bench["duration"]-pair[mode+"_duration_s"])<1e-8
   if mode=="off":assert meta["control"]["tokenizer_wait_ms"]==20.0
   item[mode]=dict(label=label,tag=tag,ttft_s=bench["mean_ttft_ms"]/1000,duration_s=bench["duration"],
     input_tps=bench["total_input_tokens"]/bench["duration"],cycle_s=float(phase["cycle_s"]),
     completed=32,failed=0,input_tokens=524288,offsets=[0,0,0,0],rounds=dict(A=16,M=0,D=7))
   for f in [path,meta_path]:
    evidence.append(dict(path=f.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
  item["ttft_reduction_pct"]=(1-item["slot"]["ttft_s"]/item["off"]["ttft_s"])*100
  item["input_tps_gain_pct"]=(item["slot"]["input_tps"]/item["off"]["input_tps"]-1)*100
  assert abs(item["ttft_reduction_pct"]-pair["ttft_reduction_pct"])<1e-8
  assert abs(item["input_tps_gain_pct"]-pair["input_throughput_gain_pct"])<1e-8
  pairs.append(item)
 means={mode:{k:statistics.mean(p[mode][k] for p in pairs) for k in ["ttft_s","duration_s","input_tps","cycle_s"]} for mode in labels}
 off,slot=means["off"],means["slot"]
 changes={k:(slot[k]/off[k]-1)*100 for k in off}
 paired_gain=statistics.mean(p["input_tps_gain_pct"] for p in pairs)
 assert abs(-changes["ttft_s"]-source["primary"]["ttft_reduction_pct"])<1e-8
 assert abs(paired_gain-source["primary"]["input_throughput_gain_pct"])<1e-8
 result=dict(scope="After alignment fix: off versus slot; 3hot1rank; L16384 B8192; DP4 TP1 EP; KV9GiB; n32 c16 out1",
   pairs=pairs,means=means,slot_vs_off_change_pct=changes,input_tps_paired_gain_mean_pct=paired_gain,
   method="Both tokenizer windows 20ms and phase aligned. Off uses runtime window control; slot uses patched env-based startup. Means are arithmetic means over 3 matched seeds. Input throughput is total_input_tokens / bench duration. Only before-first-alert samples included.",
   evidence=evidence)
 (ROOT/"data"/"off_slot_after_fix.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
 summary=table(["指标（3 组均值）","moonep off · 20ms","moonep slot · 20ms","slot 相对 off"],[
  ["平均 TTFT ↓",f'{off["ttft_s"]:.4f} s',f'{slot["ttft_s"]:.4f} s',f'<strong class="gain">降低 {-changes["ttft_s"]:.2f}%</strong>'],
  ["输入吞吐 ↑",f'{off["input_tps"]:,.2f} token/s',f'{slot["input_tps"]:,.2f} token/s',f'<strong class="gain">提高 {changes["input_tps"]:.2f}%</strong>'],
  ["整组 bench 完成时长 ↓",f'{off["duration_s"]:.4f} s',f'{slot["duration_s"]:.4f} s',f'<strong class="gain">缩短 {-changes["duration_s"]:.2f}%</strong>'],
  ["稳态请求完成周期 ↓",f'{off["cycle_s"]:.4f} s',f'{slot["cycle_s"]:.4f} s',f'<strong class="gain">缩短 {-changes["cycle_s"]:.2f}%</strong>']])
 details=table(["seed","off TTFT / s","slot TTFT / s","TTFT 降低","off 输入吞吐 / token·s⁻¹","slot 输入吞吐 / token·s⁻¹","吞吐提高"],[
  [p["seed"],f'{p["off"]["ttft_s"]:.4f}',f'{p["slot"]["ttft_s"]:.4f}',f'<span class="gain">{p["ttft_reduction_pct"]:.2f}%</span>',
   f'{p["off"]["input_tps"]:,.2f}',f'{p["slot"]["input_tps"]:,.2f}',f'<span class="gain">+{p["input_tps_gain_pct"]:.2f}%</span>'] for p in pairs])
 return '''<div id="off-slot-after-fix"><h3>9.1 主对照：修复后，slot 相对 off 有多少收益？</h3>
<p><strong>本节直接比较双方已对齐后的 moonep off 与 moonep slot。</strong>固定条件：3hot1rank 高倾斜仿真路由，L=16384、B=8192，DP4 / TP1 / EP，KV=9GiB，32 请求、并发 16、输出 1 token。</p>
<p>双方分词窗口均为 <strong>20ms</strong>：off 在运行实例上设为 20ms，slot 通过交付的环境变量补丁重启后设为 20ms。使用 <strong>3 组 warm 同 seed 配对</strong>，相同日志配置、双方降温至 ≤70°C 后测量；所有主样本在各自服务首次外来占用告警前完成。</p>
'''+summary+f'''
<div class="callout good"><p><strong>结论：在这组高倾斜、16K 输入切 8K 的工作负载下，对齐后的 slot 比 off 快。</strong>三组平均 TTFT 减少 {off["ttft_s"]-slot["ttft_s"]:.4f}s（{-changes["ttft_s"]:.2f}%）；输入吞吐增加 {slot["input_tps"]-off["input_tps"]:,.2f} token/s（约 {changes["input_tps"]:.2f}%）。三组分别比较均获益，TTFT 降低 5.02%～5.38%，吞吐提高 5.67%～6.04%。</p></div>
<p><strong>调度条件已对齐：</strong>两边各组均为初始偏移 <code>[0,0,0,0]</code>、每 rank 8 请求、16 个全真实轮 + 0 个混合轮 + 7 个全 dummy 轮，真实输入 token 总量均为 524288，32/32 请求成功。该对照用于观察消除错位后 slot 的净性能收益。</p>
<h4>三组配对明细：两种模式均已修复 / 对齐</h4>
'''+details+f'''
<p class="caption">TTFT 降低 = 1 − slot/off；吞吐提升 = slot/off − 1。上表绝对值先对各组取算术均值再比较；输入吞吐逐组按 <code>total_input_tokens / duration</code> 重算。均值之比得到 {changes["input_tps"]:.4f}%，原报告使用的逐对提升均值为 {paired_gain:.4f}%，两者四舍五入均为 5.82%。吞吐使用整组完成时长计算，不能用 TTFT 比例的倒数替代。</p>
<p><strong>证据边界：</strong>这是已有三组匹配实验的重新汇总，本次未新跑实验。该收益限于上述仿真高倾斜负载；NPU 频率只采样、未锁频，且跨模式数值等价尚未认证，详见 9.3、9.4。slot 自身从错位到对齐的改善只作为下一小节的补充因果验证。</p>
<p class="small">数据：<a href="../data/off_slot_after_fix.json">主对照重算结果及源文件 SHA256</a> · <a href="../data/alignment_fix_validation_20261017/summary.json">原始配对统计</a> · <a href="../data/alignment_fix_validation_20261017/all_groups.csv">逐组相位与 token 对账</a>。</p></div>'''
