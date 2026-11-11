"""Compute equal-weight means across completed independent GSM8K service pairs."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
MODES = ("off", "slot")
METRICS = ("mean_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "input_tps", "request_rps")
COLUMNS = "平均 TTFT（ms） | P50（ms） | P95（ms） | 输入 token/s | 请求/s"


def read_pair(file):
    data = json.loads(file.read_text())
    assert data["host"] == "141.61.45.118"
    assert set(data["formal"]) == set(MODES)
    for case in data["formal"].values():
        assert case["valid"] and case["requests"] == case["output_tokens"] == 1319
        assert case["input_tokens"] == 77756 and case["max_concurrency"] == 16
        assert Path(case["debug_log"]).is_file()
    audit = Path(data["formal"]["slot"]["debug_log"]).with_name("OPTIMIZATION_AUDIT.json")
    proof = json.loads(audit.read_text())
    assert proof["bounded_unpack_after_dispatch_ranks"] == list(range(8))
    return data["formal"]


def compare(slot, off):
    return {name: 100 * (slot[name] / off[name] - 1) for name in METRICS}


def cells(case, percent=False):
    template = "{:+.2f}%" if percent else "{:,.2f}"
    return " | ".join(template.format(case[name]) for name in METRICS)


def main():
    pairs = {"pair01": read_pair(ROOT / "RESULTS.json")}
    for pair in ("pair02", "pair03"):
        path = ROOT / "repeat_formal" / pair / "FORMAL_RESULTS.json"
        if path.is_file():
            pairs[pair] = read_pair(path)
    fingerprints = {case["dataset_sha256"] for cases in pairs.values() for case in cases.values()}
    assert len(fingerprints) == 1, fingerprints
    averages = {mode: {key: mean(cases[mode][key] for cases in pairs.values()) for key in METRICS}
                for mode in MODES}
    delta = compare(averages["slot"], averages["off"])
    pair_deltas = {pair: compare(cases["slot"], cases["off"]) for pair, cases in pairs.items()}
    rows = [
        "# 118 GSM8K 4K 独立起服重复对照", "",
        f"已完成 {len(pairs)} 对。每对 Off → MXFP8 Slot 独立起服，每个服务先预检、预热 16 条，再测完整 1,319 条。"
        "先按模式求各轮算术平均，再用 Slot 均值除以 Off 均值计算变化。P50 和 P95 为各轮分位数的平均。",
        "", f"| 组别 | {COLUMNS} |", "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        rows.append(f"| {mode} 均值 | {cells(averages[mode])} |")
    rows += [f"| Slot 相对 Off | {cells(delta, percent=True)} |",
             "", f"| 各对明细 | {COLUMNS} |", "|---|---:|---:|---:|---:|---:|"]
    for pair, cases in pairs.items():
        for mode in MODES:
            rows.append(f"| {pair} {mode} | {cells(cases[mode])} |")
        rows.append(f"| {pair} Slot 相对 Off | {cells(pair_deltas[pair], percent=True)} |")
    rows += ["", "## 正式轮 AISBench debug", ""]
    for pair, cases in pairs.items():
        rows.append("- " + " · ".join(f"[{pair} {mode}]({cases[mode]['debug_log']})" for mode in MODES))
    rows += ["", f"负载诊断见[首对的独立回放报告]({ROOT / 'RESULTS.md'})。"]
    (ROOT / "REPEAT_FORMAL_RESULTS.md").write_text("\n".join(rows) + "\n")
    (ROOT / "REPEAT_FORMAL_RESULTS.json").write_text(json.dumps(
        {"host": "141.61.45.118", "pairs": pairs, "mean_by_mode": averages,
         "slot_vs_off_pct": delta, "per_pair_slot_vs_off_pct": pair_deltas},
        ensure_ascii=False, indent=2) + "\n")
    print("\n".join(rows[:9]))


if __name__ == "__main__":
    main()
