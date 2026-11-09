import json
from pathlib import Path
from statistics import mean

import summarize_formal as one

ROOT=Path(__file__).resolve().parent
KEYS=("mean_ttft_ms","median_ttft_ms","p95_ttft_ms","input_tps","request_rps")
LABELS=("TTFT 均值 ms","P50 ms","P95 ms","输入 token/s","请求/s")

def read_pair(path):
    one.ROOT=path
    outcomes={}
    for mode in ("off","slot"):
        result=one.case(mode)
        if result:
            outcomes[mode]=result
    return outcomes

def summarize(pairs,names):
    return {
        mode:{k:mean(pairs[p][mode][k] for p in names) for k in KEYS}
        for mode in ("off","slot")
    }

def relative(values):
    return {k:one.delta(values["slot"][k],values["off"][k]) for k in KEYS}

def table_line(label,mode,values):
    return f"| {label} | {mode} | " + " | ".join(f"{values[k]:.{3 if k=='request_rps' else 1}f}" for k in KEYS) + " |"

def relative_line(label,values):
    return f"| {label} | Slot 相对 Off | " + " | ".join(f"{values[k]:+.2f}%" for k in KEYS) + " |"

def main():
    assignment=json.loads((ROOT/"PAIR_HOSTS.json").read_text())
    paths={pair:ROOT/entry["directory"] for pair,entry in assignment.items()}
    pairs={p:read_pair(path) for p,path in paths.items()}
    complete=[p for p,v in pairs.items() if len(v)==2]
    report=[
        "# 118 DP8 TP1 LongBench-v2，4K 正式对照重复结果","",
        "每对 Off → Slot 独立起服，各自校验最长样本并预热 16 条。正式轮 316 条全文、并发 16、输出 1 token。","",
        "| 轮次 | 模式 | "+" | ".join(LABELS)+" |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for p,cases in pairs.items():
        for m,data in cases.items():
            report.append(table_line(f"{p} ({assignment[p]['host']})",m,data))
        if p in complete:
            assert cases["off"]["input_tokens"]==cases["slot"]["input_tokens"]==18518962
            report.append(relative_line(p,relative(cases)))
    aggregate={}
    for label,names in (("新增两对均值",["pair02","pair03"]),("全部三对均值",["pair01","pair02","pair03"])):
        if set(names).issubset(complete):
            avg=summarize(pairs,names)
            d=relative(avg)
            pair_relative_mean={k:mean(relative(pairs[p])[k] for p in names) for k in KEYS}
            aggregate[label]=dict(pairs=names,metrics=avg,slot_vs_off_pct=d,mean_of_paired_pct=pair_relative_mean)
            for m in ("off","slot"):
                report.append(table_line(label,m,avg[m]))
            report.append(relative_line(label,d))
            report.append(relative_line(label+"，各对百分比再平均",pair_relative_mean))
    report+=["","各指标先按模式对各轮做算术平均，再算 Slot 相对 Off 的变化。TTFT 负值表示降低。历史 8K 单对的 Slot 相对 Off 为平均 TTFT −0.55%、输入吞吐 −0.13%。",
             "","## AISBench Debug 原始日志",""]
    for p,cases in pairs.items():
        report.extend(f"- {p} {m} {s['debug_log']}" for m,s in cases.items())
    result=dict(pairs=pairs,complete_pairs=complete,aggregate=aggregate,hosts=assignment)
    (ROOT/"REPEAT_FORMAL_RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    (ROOT/"REPEAT_FORMAL_RESULTS.md").write_text("\n".join(report)+"\n")
    print("\n".join(report[:23]),flush=True)

if __name__=="__main__":
    main()

