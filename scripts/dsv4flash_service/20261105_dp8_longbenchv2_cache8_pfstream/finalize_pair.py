"""Summarize a single Off/Slot service pair using all three formal rounds."""
import csv,hashlib,json,re,socket,subprocess,sys
from pathlib import Path
from metrics import read_round
ROOT=Path(__file__).resolve().parent
e=Path(sys.argv[1])
cases={}
rows=[]
for mode in ["off","slot"]:
    name="rerun01_"+mode
    case=json.loads((e/"cases"/name/"case_result.json").read_text())
    assert case["valid"] and case["formal_requests"]==948
    rounds=[read_round(case["client_dir"],i) for i in [1,2,3]]
    expected=sum(x["prompt_tokens"] for x in json.loads((ROOT/"data/selection_manifest.json").read_text())["samples"])
    assert all(r["input_tokens"]==expected and r["max_concurrency"]==16 for r in rounds)
    avg={key:sum(r[key] for r in rounds)/3 for key in ["mean_ttft_ms","input_tps"]}
    cases[mode]=dict(service_dir=case["service_dir"],client_dir=case["client_dir"],rounds=rounds,mean_three=avg)
    for r in rounds:rows.append(dict(mode=mode,**{k:v for k,v in r.items() if k!="predictions"}))
off,slot=cases["off"]["mean_three"],cases["slot"]["mean_three"]
comparison=dict(ttft_reduction_pct=100*(1-slot["mean_ttft_ms"]/off["mean_ttft_ms"]),
                input_tps_gain_pct=100*(slot["input_tps"]/off["input_tps"]-1))
agreements=[]
for i in range(3):
    a,b=cases["off"]["rounds"][i]["predictions"],cases["slot"]["rounds"][i]["predictions"]
    different=[k for k in a if a[k]!=b[k]]
    agreements.append(dict(round=i+1,different=len(different),matching=316-len(different),different_ids=different))
npu=subprocess.check_output(["npu-smi","info"],text=True)
assert set(re.findall(r"No running processes found in NPU (\d+)",npu))==set(map(str,range(8)))
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(("127.0.0.1",8444))
(e/"npu_final.txt").write_text(npu)
(e/"CLEANUP_CHECK.json").write_text(json.dumps(dict(eight_npus_released=True,port_8444_bindable=True),indent=2)+"\n")
result=dict(status="completed",cases=cases,comparison=comparison,first_token_text_agreement=agreements,
            valid_formal_requests=1896,rounds=6,criterion="Arithmetic mean of all 3 formal rounds; warmups excluded",
            features=dict(slot_cache=8,prefetch_stream=2,udma=False))
(e/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
with (e/"rounds.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
lines=["# LongBench-v2 单机DP8性能结果","",
       "Slot：每卡每层缓存8专家、cache-aware prefetch副流、MTE。Off和Slot分别一次独立起服，3轮316条，全部3轮取均值。","",
       "|模式|轮次|平均TTFT(ms)|输入吞吐(token/s)|","|---|---:|---:|---:|"]
for mode,c in cases.items():
    for r in c["rounds"]:lines.append(f'|{mode}|{r["round"]}|{r["mean_ttft_ms"]:.3f}|{r["input_tps"]:.3f}|')
    a=c["mean_three"];lines.append(f'|{mode}|三轮均值|{a["mean_ttft_ms"]:.3f}|{a["input_tps"]:.3f}|')
lines+=["",f'Slot TTFT降低：{comparison["ttft_reduction_pct"]:.3f}%；输入吞吐提升：{comparison["input_tps_gain_pct"]:.3f}%。',
        "","首token文本差异（316条/轮）："+", ".join(str(x["different"]) for x in agreements)+"。这不等于LongBench任务准确率。",
        "","全部原始aisbench debug日志、JSONL和SQLite数据见对应client_dir。八卡及8444端口已释放。"]
(e/"RESULT.md").write_text("\n".join(lines)+"\n")
current=json.loads((ROOT/"CURRENT_EXECUTION.json").read_text())
current.update(status="completed",result=str(e/"RESULT.md"))
(ROOT/"CURRENT_EXECUTION.json").write_text(json.dumps(current,indent=2)+"\n")
print(json.dumps({"means":{m:c["mean_three"] for m,c in cases.items()},"comparison":comparison,"formal_requests":1896},ensure_ascii=False))
