"""Publish each completed independent-service pair; average all available pairs."""
import csv,json,socket,subprocess,sys
from pathlib import Path
from metrics import read_round
ROOT=Path(__file__).resolve().parent
e=Path(sys.argv[1])
cases={};pairs=[];rows=[]
for n in [1,2,3]:
    pair=f"pair{n:02}"
    if not all((e/"cases"/(pair+"_"+m)/"case_result.json").exists() for m in ["off","slot"]):break
    metrics={}
    for mode in ["off","slot"]:
        name=pair+"_"+mode
        case=json.loads((e/"cases"/name/"case_result.json").read_text())
        assert case["valid"] and case["formal_requests"]==316
        r=read_round(case["client_dir"],1)
        expected=sum(x["prompt_tokens"] for x in json.loads((ROOT/"data/selection_manifest.json").read_text())["samples"])
        assert r["input_tokens"]==expected and r["max_concurrency"]==16
        cases[name]=dict(service_dir=case["service_dir"],client_dir=case["client_dir"],measurement=r)
        metrics[mode]={k:r[k] for k in ["mean_ttft_ms","input_tps"]}
        rows.append(dict(pair=pair,mode=mode,**{k:v for k,v in r.items() if k!="predictions"}))
    a,b=metrics["off"],metrics["slot"]
    pairs.append(dict(pair=pair,**metrics,
        ttft_reduction_pct=100*(1-b["mean_ttft_ms"]/a["mean_ttft_ms"]),
        input_tps_gain_pct=100*(b["input_tps"]/a["input_tps"]-1)))
assert pairs
overall={}
for mode in ["off","slot"]:
    overall[mode]={k:sum(p[mode][k] for p in pairs)/len(pairs) for k in ["mean_ttft_ms","input_tps"]}
a,b=overall["off"],overall["slot"]
overall.update(ttft_reduction_pct=100*(1-b["mean_ttft_ms"]/a["mean_ttft_ms"]),
               input_tps_gain_pct=100*(b["input_tps"]/a["input_tps"]-1))
agreements=[]
for p in pairs:
    name=p["pair"]
    a=cases[name+"_off"]["measurement"]["predictions"]
    b=cases[name+"_slot"]["measurement"]["predictions"]
    ids=[k for k in a if a[k]!=b[k]]
    agreements.append(dict(pair=name,different=len(ids),matching=316-len(ids),different_ids=ids))
result=dict(status="completed" if len(pairs)==3 else "partial",
    completed_pairs=len(pairs),valid_formal_requests=632*len(pairs),cases=cases,pairs=pairs,
    overall=overall,first_token_text_agreement=agreements,
    criterion="Each service runs one formal round; arithmetic mean of all completed matched pairs",
    features=dict(slot_cache=0,prefetch_stream=2,udma=False,dispatch_dtype="mxfp8",fused_publish=True))
(e/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
with (e/"rounds.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=["# LongBench-v2 单机DP8交叉性能结果","",
       "MTE；Slot关闭持久专家缓存，开启prefetch副流、MXFP8 dispatch、融合publish。每次独立起服只正式测1轮316条。","",
       f"已完成{len(pairs)}对，{632*len(pairs)}条正式请求。各模式对全部已完成配对数据取算术均值。","",
       "|服务对|模式|平均TTFT(ms)|输入吞吐(token/s)|","|---|---|---:|---:|"]
for p in pairs:
    for mode in ["off","slot"]:
        m=p[mode];lines.append(f'|{p["pair"]}|{mode}|{m["mean_ttft_ms"]:.3f}|{m["input_tps"]:.3f}|')
for mode in ["off","slot"]:
    m=overall[mode];lines.append(f'|均值|{mode}|{m["mean_ttft_ms"]:.3f}|{m["input_tps"]:.3f}|')
lines+=["",f'Slot TTFT降低{overall["ttft_reduction_pct"]:.3f}%；输入吞吐提升{overall["input_tps_gain_pct"]:.3f}%。',
        "","首token文本差异（316条/对）："+", ".join(str(x["different"]) for x in agreements)+"；不等于LongBench任务准确率。",
        "","debug日志、逐请求原始JSONL和SQLite位于各client_dir。"]
(e/"RESULT.md").write_text("\n".join(lines)+"\n")
print(json.dumps(dict(completed_pairs=len(pairs),overall=overall),ensure_ascii=False),flush=True)
