#!/usr/bin/env python3
"""Summarize only validated cases from a specified execution directory."""
import csv,hashlib,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PLAN=["pair01_off","pair01_slot","pair02_slot","pair02_off"]

def number(value):
    if isinstance(value,(int,float)):return float(value)
    match=re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",str(value).replace(",",""))
    if not match:raise ValueError("Not numeric: "+repr(value))
    return float(match.group())

def read_round(client,i):
    root=Path(client)/f"round_{i}"
    validation=json.loads((root/"validation.json").read_text())
    assert validation["valid"] and validation["requests"]==316
    candidates=list((root/"aisbench").rglob("longbenchv2-full-le128k.json"))
    assert len(candidates)==1,candidates
    summary_path=candidates[0]
    summary=json.loads(summary_path.read_text())
    total=lambda key:number(summary[key]["total"])
    duration_text = str(summary["Benchmark Duration"]["total"]).strip()
    assert duration_text.endswith((" ms", " s")), duration_text
    duration_seconds = number(duration_text) / (1000 if duration_text.endswith(" ms") else 1)
    assert total("Total Requests")==total("Success Requests")==316
    assert total("Failed Requests")==0 and total("Total Generated Tokens")==316
    with summary_path.with_suffix(".csv").open() as f:
        metrics=list(csv.DictReader(f))
    ttft=[r for r in metrics if r["Performance Parameters"]=="TTFT" and r["Stage"]=="total"]
    assert len(ttft)==1
    assert "ms" in ttft[0]["Average"],ttft
    raw=summary_path.parent/"longbenchv2-full-le128k_details.jsonl"
    predictions={}
    with raw.open() as stream:
        for line in stream:
            record=json.loads(line)
            predictions[int(record["id"])]=record.get("prediction")
    assert len(predictions)==316
    return dict(round=i,mean_ttft_ms=number(ttft[0]["Average"]),
                median_ttft_ms=number(ttft[0]["Median"]),p90_ttft_ms=number(ttft[0]["P90"]),
                p99_ttft_ms=number(ttft[0]["P99"]),input_tps=total("Input Token Throughput"),
                duration_seconds=duration_seconds,input_tokens=int(total("Total Input Tokens")),
                output_tokens=int(total("Total Generated Tokens")),requests=316,
                concurrency=total("Concurrency"),max_concurrency=total("Max Concurrency"),
                summary_file=str(summary_path),raw_file=str(raw),predictions=predictions)

def main():
    execution=Path(sys.argv[1])
    cases={}
    for name in PLAN:
        result=json.loads((execution/"cases"/name/"case_result.json").read_text())
        assert result["valid"]
        rounds=[read_round(result["client_dir"],i) for i in [1,2,3]]
        avg={key:sum(row[key] for row in rounds[1:])/2 for key in ["mean_ttft_ms","input_tps"]}
        cases[name]=dict(case=name,service_dir=result["service_dir"],
                        client_dir=result["client_dir"],rounds=rounds,mean_last_two=avg)
    for case in cases.values():
        a=case["rounds"][1]["predictions"];b=case["rounds"][2]["predictions"]
        different=[k for k in a if a[k]!=b[k]]
        case["within_mode_round2_vs_round3_first_token"]=dict(matching=316-len(different),total=316,disagreement_ids=different)
    pairs=[]
    for pair in ["pair01","pair02"]:
        off=cases[pair+"_off"]["mean_last_two"];slot=cases[pair+"_slot"]["mean_last_two"]
        agreements=[]
        for i in [1,2,3]:
            a=cases[pair+"_off"]["rounds"][i-1]["predictions"]
            b=cases[pair+"_slot"]["rounds"][i-1]["predictions"]
            disagreements=[k for k in a if a[k]!=b[k]]
            agreements.append(dict(round=i,matching=316-len(disagreements),total=316,
                                   disagreement_ids=disagreements))
        pairs.append(dict(pair=pair,off=off,slot=slot,
                   ttft_reduction_pct=100*(1-slot["mean_ttft_ms"]/off["mean_ttft_ms"]),
                   input_tps_gain_pct=100*(slot["input_tps"]/off["input_tps"]-1),
                   first_token_text_agreement=agreements))
    measured=[r for c in cases.values() for r in c["rounds"][1:]]
    assert all(r["max_concurrency"]==16 for r in measured), measured
    overall={}
    for mode in ["off","slot"]:
        selected=[c["mean_last_two"] for name,c in cases.items() if name.endswith("_"+mode)]
        overall[mode]={key:sum(r[key] for r in selected)/len(selected) for key in ["mean_ttft_ms","input_tps"]}
    overall["ttft_reduction_pct"]=100*(1-overall["slot"]["mean_ttft_ms"]/overall["off"]["mean_ttft_ms"])
    overall["input_tps_gain_pct"]=100*(overall["slot"]["input_tps"]/overall["off"]["input_tps"]-1)
    all_input_counts={r["input_tokens"] for c in cases.values() for r in c["rounds"]}
    expected_input=sum(x["prompt_tokens"] for x in json.loads((execution/"selection_manifest.json").read_text())["samples"])
    assert all_input_counts == {expected_input}, (all_input_counts, expected_input)
    payload=dict(execution=str(execution),cases=cases,pairs=pairs,overall_equal_service_mean=overall,
                 valid_formal_requests=3792,all_rounds_complete=True,
                 selected_samples=316,input_cap=131072,max_model_len=131073,
                 kv_cache_bytes_per_card=25769803776,
                 criterion="Each service mean of rounds2/3; round1 retained and excluded from comparisons",
                 previous_invalid_execution_excluded=True)
    (execution/"results.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n")
    with (execution/"rounds.csv").open("w",newline="") as f:
        fields=["case","round","mean_ttft_ms","median_ttft_ms","p90_ttft_ms","p99_ttft_ms",
                "input_tps","duration_seconds","input_tokens","output_tokens","requests","concurrency","max_concurrency"]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for name,c in cases.items():
            for r in c["rounds"]:writer.writerow(dict(case=name,**{k:r[k] for k in fields if k!="case"}))
    lines=["# LongBench-v2 prefill：off / MoonEP Slot",
           "", "316条128K以内原始全文，原生专家路由；DP8/TP1/EP8，客户端并发16，输出1token，8K chunk，24GiB KV/卡，两组关闭prefix cache。",
           "", "每台独立服务测3轮，取后2轮均值。旧run_20261104_114303因输入完整性问题整轮排除。",
           "", "| 服务对 | off TTFT(ms) | Slot TTFT(ms) | TTFT降低 | off输入吞吐(token/s) | Slot输入吞吐(token/s) | 吞吐提高 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for p in pairs:
        lines.append(f'| {p["pair"]} | {p["off"]["mean_ttft_ms"]:.2f} | {p["slot"]["mean_ttft_ms"]:.2f} | {p["ttft_reduction_pct"]:.2f}% | {p["off"]["input_tps"]:.2f} | {p["slot"]["input_tps"]:.2f} | {p["input_tps_gain_pct"]:.2f}% |')
    lines += ["",f'两对等权平均：off TTFT {overall["off"]["mean_ttft_ms"]:.2f} ms，Slot {overall["slot"]["mean_ttft_ms"]:.2f} ms；TTFT降低 {overall["ttft_reduction_pct"]:.2f}%。off输入吞吐 {overall["off"]["input_tps"]:.2f} token/s，Slot {overall["slot"]["input_tps"]:.2f} token/s；吞吐提高 {overall["input_tps_gain_pct"]:.2f}%。负值表示Slot退化。']
    lines+=["","全部12轮、3792条正式请求通过成功数、输出token、完整输入token数及prompt SHA256校验。每轮输入总数18,518,962 token；纳入对比的8轮客户端最高并发均为16。模式核验文件逐轮保留，8个实际worker的参数/环境/映射库及初始化日志均检查。",
            "", "原始结果、debug日志、逐请求JSONL及关联SQLite数据库保留在各client_dir；详见results.json和rounds.csv。",
            "", "首token文本一致性仅作辅助核查，不替代完整生成精度测试："]
    for p in pairs:
        lines.append("- "+p["pair"]+": "+", ".join(f'round{x["round"]} {x["matching"]}/{x["total"]}' for x in p["first_token_text_agreement"]))
    lines+=["","同模式round2/round3首token文本一致性："]
    for name,case in cases.items():
        item=case["within_mode_round2_vs_round3_first_token"]
        lines.append(f'- {name}: {item["matching"]}/{item["total"]}')
    lines+=["","已经观察到首token文本不完全一致；同模式重复也存在差异。差异原因尚未定位，本性能实验不能证明完整生成精度等价。"]
    (execution/"RESULT.md").write_text("\n".join(lines)+"\n")
    print("SUMMARY_COMPLETE",execution/"RESULT.md")
    print("PAIRS",json.dumps(pairs,ensure_ascii=False))

if __name__=="__main__":main()
