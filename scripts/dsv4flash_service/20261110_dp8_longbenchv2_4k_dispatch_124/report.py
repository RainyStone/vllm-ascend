"""Summarize only validated rounds, including a same-host historical reference."""
import json
from pathlib import Path

import summarize_formal as previous_parser

ROOT = Path(__file__).resolve().parent
CASES = {"off": ("mxfp8", "off"), "mxfp8_slot": ("mxfp8", "slot"),
         "bf16_copy_slot": ("bf16_copy", "slot"), "bf16_zero_copy_slot": ("bf16_zero_copy", "slot")}
METRICS = ("mean_ttft_ms", "median_ttft_ms", "p95_ttft_ms", "input_tps", "request_rps")


def summarize():
    results = {}
    for label, (profile, mode) in CASES.items():
        previous_parser.ROOT = ROOT / profile
        case = previous_parser.case(mode)
        if case is not None:
            assert case["input_tokens"] == 18518962, (label, case["input_tokens"])
            results[label] = case
    rows = ["# 124 单机 8 卡 LongBench-v2 4K", "",
            "| 组别 | TTFT 均值 ms | P50 ms | P95 ms | 输入 token/s | 请求/s |",
            "|---|---:|---:|---:|---:|---:|"]
    for label, case in results.items():
        rows.append(f"| {label} | {case['mean_ttft_ms']:.1f} | {case['median_ttft_ms']:.1f} | "
                    f"{case['p95_ttft_ms']:.1f} | {case['input_tps']:.1f} | {case['request_rps']:.3f} |")
    historical_path = (ROOT.parent / "20261107_dp8_longbenchv2_4k_mte_118"
                       / "run_20261107_193356_118/REPEAT_FORMAL_RESULTS.json")
    archive = json.loads(historical_path.read_text())
    assert archive["hosts"]["pair03"]["host"] == "141.61.45.124"
    historical = archive["pairs"]["pair03"]
    for mode, label in (("off", "历史 124 Off"), ("slot", "历史 124 MXFP8 Slot")):
        case = historical[mode]
        rows.append(f"| {label} | {case['mean_ttft_ms']:.1f} | {case['median_ttft_ms']:.1f} | "
                    f"{case['p95_ttft_ms']:.1f} | {case['input_tps']:.1f} | {case['request_rps']:.3f} |")
    rows += ["", "| 对比 | TTFT 均值 | P50 | P95 | 输入吞吐 | 请求吞吐 |",
             "|---|---:|---:|---:|---:|---:|"]
    comparisons = {}

    def compare(label, new, base):
        change = {key: previous_parser.delta(new[key], base[key]) for key in METRICS}
        comparisons[label] = change
        rows.append(f"| {label} | " + " | ".join(f"{change[key]:+.2f}%" for key in METRICS) + " |")
        return change

    for label, base in (("mxfp8_slot", "off"), ("bf16_copy_slot", "off"),
                        ("bf16_zero_copy_slot", "off"), ("bf16_zero_copy_slot", "bf16_copy_slot")):
        if label in results and base in results:
            compare(f"{label} 相对 {base}", results[label], results[base])
    old_delta = compare("历史 124 Slot 相对历史 Off", historical["slot"], historical["off"])
    if "off" in results:
        compare("本轮 Off 相对历史 124 Off", results["off"], historical["off"])
    if "mxfp8_slot" in results:
        compare("本轮 MXFP8 Slot 相对历史 124 Slot", results["mxfp8_slot"], historical["slot"])
    new_delta = comparisons.get("mxfp8_slot 相对 off")
    if new_delta is not None:
        ttft = new_delta["mean_ttft_ms"] - old_delta["mean_ttft_ms"]
        tps = new_delta["input_tps"] - old_delta["input_tps"]
        rows += ["", f"以各自 Off 为基线，Slot 的相对变化比历史同机对照多 {ttft:+.2f} 个百分点 TTFT，"
                     f"{tps:+.2f} 个百分点输入吞吐。两轮为独立端到端对照。"]
    rows += ["", "## 正式轮 Debug 日志", ""] + [f"- {label} {case['debug_log']}" for label, case in results.items()]
    (ROOT / "FORMAL_RESULTS.md").write_text("\n".join(rows) + "\n")
    (ROOT / "FORMAL_RESULTS.json").write_text(json.dumps({"host": "141.61.45.124", "cases": results,
                                                         "historical_124": historical,
                                                         "comparisons_pct": comparisons}, indent=2) + "\n")


if __name__ == "__main__":
    summarize()
