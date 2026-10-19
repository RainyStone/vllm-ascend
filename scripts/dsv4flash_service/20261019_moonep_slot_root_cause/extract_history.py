import json,pathlib,statistics,hashlib
root=pathlib.Path("vllm-ascend/scripts/dsv4flash_service/20261011_opt2_off_vs_slot/data")
rows=[]
for route,skew in [("uniform",1.00),("4hot",1.66),("3hot1r",2.48),("4hot1r",2.98)]:
 for length in [8192,16384]:
  for budget in [8192,16384]:
   row=dict(route=route,skew=skew,input_tokens=length,budget=budget)
   for mode in ["off","slot"]:
    d=root/f"op2_mbt{budget}_{route}_{mode}_L{length//1024}k"
    records=[]
    for c in [2,3]:
     p=d/f"L{length//1024}k_n32_c{c}.json"
     v=json.loads(p.read_text())
     records.append(dict(file=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),mean_ttft_ms=v["mean_ttft_ms"],input_throughput=v["total_input_tokens"]/v["duration"],completed=v["completed"],failed=v["failed"],total_input_tokens=v["total_input_tokens"]))
    row[mode]=dict(mean_ttft_ms=statistics.mean(x["mean_ttft_ms"] for x in records),input_throughput=statistics.mean(x["input_throughput"] for x in records),records=records)
   row["ttft_reduction_pct"]=(1-row["slot"]["mean_ttft_ms"]/row["off"]["mean_ttft_ms"])*100
   row["throughput_gain_pct"]=(row["slot"]["input_throughput"]/row["off"]["input_throughput"]-1)*100
   rows.append(row)
print(json.dumps(dict(source_root=str(root.resolve()),statistics="c2/c3 arithmetic mean; c1 excluded; input throughput = total_input_tokens/duration",rows=rows),ensure_ascii=False))
