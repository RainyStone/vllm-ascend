#!/usr/bin/env python3
"""Finish reporting automatically after the independent full-matrix controller completes."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
session = Path(sys.argv[1])
previous = ROOT / "data/executions/run_20261104_122031_literal"
deadline = time.monotonic() + 8 * 3600
while True:
    state_path = session / "remaining_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("event") in ("stopped_on_error", "cleanup_failure"):
        raise RuntimeError(f"Matrix stopped: {state}")
    if (session / "result_export/EXPORT_SHA256.json").exists():
        break
    if time.monotonic() > deadline:
        raise TimeoutError("Matrix reporting wait exceeded eight hours")
    time.sleep(10)

subprocess.run([sys.executable, str(ROOT / "compare_first_token_text.py"),
                str(session), str(previous)], check=True)
new = json.loads((session / "results.json").read_text())
old = json.loads((previous / "results.json").read_text())
assert new["valid_formal_requests"] == 3792 and new["all_rounds_complete"]
n = new["overall_equal_service_mean"]
o = old["overall_equal_service_mean"]
comparison = {
    "new_execution": str(session), "previous_execution": str(previous),
    "previous": o, "fixed": n,
    "fixed_slot_vs_old_slot": {
        "ttft_reduction_pct": (1 - n["slot"]["mean_ttft_ms"] / o["slot"]["mean_ttft_ms"]) * 100,
        "input_tps_gain_pct": (n["slot"]["input_tps"] / o["slot"]["input_tps"] - 1) * 100,
    },
    "off_baseline_change": {
        "ttft_change_pct": (n["off"]["mean_ttft_ms"] / o["off"]["mean_ttft_ms"] - 1) * 100,
        "input_tps_change_pct": (n["off"]["input_tps"] / o["off"]["input_tps"] - 1) * 100,
    },
}
(session / "FIX_COMPARISON.json").write_text(json.dumps(comparison, indent=2))
text = json.loads((session / "FIRST_TOKEN_COMPARISON.json").read_text())
report = [
    "修复版 LongBench-v2 重跑完成：3792条正式请求，四次独立起服、12轮均通过校验。",
    "每台服务性能取第2、3轮，两台服务等权平均；模型和服务参数沿用已确认方案。",
    f"Off: TTFT={n['off']['mean_ttft_ms']:.3f} ms, input TPS={n['off']['input_tps']:.3f}",
    f"Slot: TTFT={n['slot']['mean_ttft_ms']:.3f} ms, input TPS={n['slot']['input_tps']:.3f}",
    f"Slot相对Off: TTFT降低{n['ttft_reduction_pct']:.3f}%，输入TPS变化{n['input_tps_gain_pct']:.3f}%。",
    f"各重复首token文本稳定样本：off={text['stable_off']}/316，slot={text['stable_slot']}/316。",
    f"两模式均稳定样本={text['stable_both']}；其中不同={len(text['stable_both_different_ids'])}。",
    "只比较一个输出token的文本，不是LongBench任务准确率，也不能证明token ID相同。",
    "本轮正式性能不含诊断统计开销；逐rank负载与固定请求诊断如需补采，独立计时。",
    "原始debug日志、JSONL、SQLite路径见result_export/RAW_LOG_INDEX.csv。",
]
(session / "POST_RUN_REPORT.txt").write_text("\n".join(report) + "\n")
out = session / "result_export"
for name in ("SOURCE_PROVENANCE.json", "ANALYSIS_NOTES.txt", "FIRST_TOKEN_COMPARISON.json",
             "FIX_COMPARISON.json", "POST_RUN_REPORT.txt"):
    shutil.copy2(session / name, out / name)
for name in ("run_full_matrix.py", "compare_first_token_text.py", "postprocess_when_complete.py"):
    shutil.copy2(ROOT / name, out / "scripts" / name)
manifest = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in out.rglob("*") if p.is_file() and p.name != "EXPORT_SHA256.json"}
(out / "EXPORT_SHA256.json").write_text(json.dumps(manifest, indent=2))
(session / "POSTPROCESS_COMPLETE.json").write_text(json.dumps(
    {"complete": True, "report": str(session / "POST_RUN_REPORT.txt")}, indent=2))
print("\n".join(report), flush=True)
