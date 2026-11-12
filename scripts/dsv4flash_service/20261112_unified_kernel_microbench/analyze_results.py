#!/usr/bin/env python3
"""Summarize all observations without discarding the initial drifted case."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def summarize(pattern):
    files = sorted((ROOT / "raw").glob(pattern))
    runs = [json.loads(p.read_text()) for p in files]
    assert len(runs) == 6
    assert all(d["finished"] and d["correctness_pass"] for d in runs)
    assert all(d["correctness"] == runs[0]["correctness"] for d in runs)
    variants = {kind: [d for d in runs if d["variant"] == kind]
                for kind in ("split", "unified")}
    assert all(len(v) == 3 for v in variants.values())
    rows = []
    for name in runs[0]["cases"]:
        values = {kind: [d["cases"][name]["median_device_us"] for d in selected]
                  for kind, selected in variants.items()}
        wall = {kind: [d["cases"][name]["median_wall_us"] for d in selected]
                for kind, selected in variants.items()}
        a, b = statistics.mean(values["split"]), statistics.mean(values["unified"])
        pair_changes = []
        for old_i, new_i in ((0, 1), (3, 2), (4, 5)):
            x = runs[old_i]["cases"][name]["median_device_us"]
            y = runs[new_i]["cases"][name]["median_device_us"]
            pair_changes.append((y / x - 1) * 100)
        rows.append({
            "case": name, "split_run_medians_us": values["split"],
            "unified_run_medians_us": values["unified"],
            "split_mean_of_run_medians_us": a,
            "unified_mean_of_run_medians_us": b,
            "change_percent": (b / a - 1) * 100,
            "paired_changes_percent": pair_changes,
            "split_wall_mean_of_run_medians_us": statistics.mean(wall["split"]),
            "unified_wall_mean_of_run_medians_us": statistics.mean(wall["unified"]),
        })
    return {"run_files": [str(p.relative_to(ROOT)) for p in files],
            "correctness_checks": sum(len(d["correctness"]) for d in runs),
            "identical_output_digests_and_correctness_metadata": True,
            "cases": rows}


initial = summarize("[0-9][0-9]_*.json")
focus = summarize("focus_[0-9][0-9]_*.json")
old_paths = sorted((ROOT / "artifacts/split").glob("*.so"))
new_path = ROOT / "artifacts/unified/libascend_moonep_kernels.so"
libraries = {
    "split": {p.name: {"bytes": p.stat().st_size,
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in old_paths},
    "unified": {"bytes": new_path.stat().st_size,
                "sha256": hashlib.sha256(new_path.read_bytes()).hexdigest()},
}
report = {
    "machine": "141.61.45.124", "physical_npu": 0,
    "scope": "one-card local synthetic operators only; no SHMEM/HCCL/model",
    "timing_method": "mean of 3 independent process medians; each median uses event-time batches",
    "initial_sampling": {"warmup": 10, "batches": 7, "iterations_per_batch": 50},
    "focus_sampling": {"warmup": 500, "batches": 9, "iterations_per_batch": 100},
    "initial": initial, "focused_confirmation": focus, "libraries": libraries,
    "total_correctness_checks": initial["correctness_checks"] + focus["correctness_checks"],
    "limits": [
        "Other NPUs had unrelated tasks; package power/clock variation was not isolated.",
        "Event spans include gaps between host submissions; these are not per-instruction hardware traces.",
        "The initial MXFP4 prefill-full timing drift is retained, not discarded.",
        "Focused confirmation uses its own identical synthetic inputs on both variants and longer warmup.",
        "Three repeats support an exploratory comparison, not proof of zero performance impact.",
        "No multi-rank communication, real-weight model or Home-GEMM-overlap end-to-end benchmark was run.",
    ],
}
(ROOT / "RESULTS.json").write_text(json.dumps(report, indent=2))
lines = [
    "# 统一设备库算子微基准",
    "",
    "机器141.61.45.124，物理NPU 0；不启动模型、不初始化SHMEM/HCCL，不采负载。",
    "分库/统一库各3次独立进程，顺序A→B→B→A→A→B；每次先预热。",
    "表中耗时为每进程批次中位值再取3进程均值，单位微秒；正百分比表示变慢。",
    "初始采样10次预热、7批×50次；全部初始数据保留。",
    "",
    "| 场景 | 分库 µs | 统一库 µs | 变化 |",
    "|---|---:|---:|---:|",
]
for row in initial["cases"]:
    lines.append("| {} | {:.3f} | {:.3f} | {:+.2f}% |".format(
        row["case"], row["split_mean_of_run_medians_us"],
        row["unified_mean_of_run_medians_us"], row["change_percent"]))
lines += [
    "",
    "初始MXFP4 prefill完整范围的同一进程内存在约250→300微秒的批次漂移。",
    "为验证该点是否稳定劣化，额外做500次预热、9批×100次的交叉三对复核；单独列出，不替换初始结果。",
    "",
    "| 充分预热复核 | 分库 µs | 统一库 µs | 变化 |",
    "|---|---:|---:|---:|",
]
for row in focus["cases"]:
    lines.append("| {} | {:.3f} | {:.3f} | {:+.2f}% |".format(
        row["case"], row["split_mean_of_run_medians_us"],
        row["unified_mean_of_run_medians_us"], row["change_percent"]))
lines += [
    "",
    "正确性：{}项检查通过，跨版本/重复的输出指纹及正确性元数据全部一致。".format(report["total_correctness_checks"]),
    "覆盖wire原始字节、有效行及未写尾部，publish的NaN尾部清零，以及BF16/MXFP4 Home/Slot/完整范围GEMM与原生GMM参考。",
    "库文件总量：分库{}字节，统一库{}字节；_C和SHMEM/CANN依赖继续独立。".format(
        sum(p.stat().st_size for p in old_paths), new_path.stat().st_size),
    "",
    "边界：其他卡存在任务，未隔离整机功耗/频率因素；事件区间可能含主机提交间隙。",
    "当前结果只支持本地算子范围的探索性判断，不能推广为多rank同步正确性或模型端到端性能结论。",
    "原始日志/逐批JSON在raw/，每轮均记录实际加载路径和库哈希；所有初始结果均保留。",
]
(ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
print("Correctness checks:", report["total_correctness_checks"])
print("Focused confirmation:", json.dumps(focus["cases"], indent=2))
