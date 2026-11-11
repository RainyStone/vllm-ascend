import csv
import io
import json
import sqlite3
from pathlib import Path

import numpy as np

from metrics import number, read_round

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parents[1]/"20261106_dp8_longbenchv2_mxfp8_publish/data/executions/run_20261106_190256"

def percentile_95(raw):
    db_cache = {}
    observations = []
    with raw.open() as stream:
        for line in stream:
            row = json.loads(line)
            name = row["db_name"]
            if name not in db_cache:
                path = raw.parent/"db_data"/name
                assert path.is_file(), path
                with sqlite3.connect(path.as_uri()+"?mode=ro",uri=True) as db:
                    db_cache[name] = {
                        i:np.load(io.BytesIO(blob),allow_pickle=False)
                        for i,blob in db.execute("SELECT id, arr_blob FROM numpy_store")
                    }
            ts = db_cache[name][int(row["time_points"]["__db_ref__"])]
            assert len(ts) >= 2 and ts[1] >= ts[0], row["id"]
            observations.append((float(ts[1])-float(ts[0]))*1000)
    assert len(observations) == 316
    return float(np.percentile(observations,95)), float(np.mean(observations))

def delta(v, base):
    return 100*(v/base-1)

def case(mode):
    clients = list((ROOT/"formal/data/formal/aisbench").glob(f"*-{mode}-*"))
    validated = [p for p in clients if (p/"round_1/validation.json").is_file()]
    if not validated:
        return None
    client, = validated
    stats = read_round(client, 1)
    del stats["predictions"]
    p95, mean = percentile_95(Path(stats["raw_file"]))
    assert abs(mean-stats["mean_ttft_ms"]) < 0.11, (mean,stats["mean_ttft_ms"])
    stats["p95_ttft_ms"] = p95
    summary = json.loads(Path(stats["summary_file"]).read_text())
    request_key = next((k for k in summary if k.lower() in ("request throughput","request goodput")), None)
    stats["request_rps"] = number(summary[request_key]["total"]) if request_key else stats["requests"]/stats["duration_seconds"]
    stats["client_dir"] = str(client)
    stats["debug_log"] = str(client/"round_1/aisbench_debug.log")
    return stats

def main():
    cases = {mode:case(mode) for mode in ("off","slot")}
    cases = {mode:v for mode,v in cases.items() if v}
    if not cases:
        print("No validated formal round yet")
        return
    result = {"validated_cases":cases}
    report = [
        "# 118 DP8 TP1 LongBench-v2，4K chunk，Off 与 MTE Slot", "",
        "正式轮各独立起服，最长样本预检后各预热 16 条。每轮 316 条原始全文，并发 16，输出 1 token。预检和预热排除。", "",
        "| 模式 | TTFT均值 ms | P50 ms | P95 ms | 输入 token/s | 请求/s | 正式秒数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode,s in cases.items():
        report.append(f"| {mode} | {s['mean_ttft_ms']:.1f} | {s['median_ttft_ms']:.1f} | {s['p95_ttft_ms']:.1f} | {s['input_tps']:.1f} | {s['request_rps']:.3f} | {s['duration_seconds']:.2f} |")
    if len(cases)==2:
        off,slot = cases["off"],cases["slot"]
        assert off["input_tokens"] == slot["input_tokens"] == 18518962
        relative = {k:delta(slot[k],off[k]) for k in ("mean_ttft_ms","median_ttft_ms","p95_ttft_ms","input_tps","request_rps")}
        result["slot_vs_off_pct"] = relative
        report += ["", f"4K Slot 相对 Off，TTFT 均值 {relative['mean_ttft_ms']:+.2f}%，P50 {relative['median_ttft_ms']:+.2f}%，P95 {relative['p95_ttft_ms']:+.2f}%，输入吞吐 {relative['input_tps']:+.2f}%，请求吞吐 {relative['request_rps']:+.2f}%。"]
        previous_text = (PREVIOUS/"RESULT.md").read_text()
        assert "14463.9" in previous_text and "14384.5" in previous_text and "62507.4" in previous_text and "62426.7" in previous_text
        report += ["", "历史 8K 单对的 Slot 相对 Off，TTFT 均值 −0.55%，输入吞吐 −0.13%。本轮 4K 同样只有一对独立起服。"]
        result["historical_8k_slot_vs_off_pct"] = dict(mean_ttft_ms=delta(14384.5,14463.9),input_tps=delta(62426.7,62507.4))
    report += ["","## 正式轮 Debug 日志",""] + [f"- {mode} {s['debug_log']}" for mode,s in cases.items()]
    (ROOT/"FORMAL_RESULTS.json").write_text(json.dumps(result,indent=2)+"\n")
    (ROOT/"FORMAL_RESULTS.md").write_text("\n".join(report)+"\n")
    print("\n".join(report[:14]))
