#!/usr/bin/env python3
"""Preview adaptive strategy by default; workloads require explicit --execute."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import shutil
import signal

import bisect_logic as logic
import measurements
import resources
import runtime
import study


def point_plan(cfg, index, ratio):
    if not cfg["bounds"][0] <= ratio <= cfg["bounds"][1]:
        raise ValueError("No point may leave the confirmed interval.")
    return study.make_plan(cfg, f"point{index:02d}", [ratio], cfg["pairs_per_point"])


def history_from_disk(root, cfg):
    history = []
    steps = sorted((root / "points").glob("[0-9][0-9]_*"))
    for index, step in enumerate(steps, 1):
        action = logic.next_action(history, cfg)
        if action["action"] != "measure":
            raise ValueError("Unexpected saved points after a stopping decision.")
        plan = study.read(step / "plan.json")
        study.validate_plan(plan)
        if plan != point_plan(cfg, index, action["ratio"]):
            raise ValueError("Saved point does not match the adaptive decision history.")
        pairs = [measurements.collect_pair(step, pair, plan) for pair in plan["pairs"]]
        if any(pair["status"] != "complete" for pair in pairs):
            if index != len(steps):
                raise ValueError("Saved data continues after an incomplete point.")
            return history, (step, plan, pairs)
        point = logic.classify_point(action["ratio"], pairs, cfg)
        point["directory"] = str(step.relative_to(root))
        history.append(point)
    return history, None


def report(root, cfg):
    history, pending = history_from_disk(root, cfg)
    visible = list(history)
    if pending:
        step, plan, pairs = pending
        point = logic.classify_point(plan["ratios"][0], pairs, cfg)
        point["directory"] = str(step.relative_to(root))
        visible.append(point)
    action = logic.next_action(visible, cfg)
    summary = {"schema": "dp8-noisy-bisection-v1", "settings": cfg,
               "points": visible, "decision": action, "search_stopped": action["action"] == "stop",
               "candidate_bracket_ready": action.get("status") == "bracket_found",
               "inference": "Exploratory; candidate bracket conditional on approximate monotonicity. "
                            "No confidence or universal-threshold claim."}
    runtime.write_json(root / "state.json", summary)
    rows = [{"ratio": p["ratio"], "decision": p["status"], "votes": p.get("votes"),
             "ttft_gain_pct": p.get("ttft_gain_pct", ""),
             "throughput_gain_pct": p.get("throughput_gain_pct", ""),
             "directory": p["directory"], "reason": p["reason"]} for p in visible]
    measurements.csv_file(root / "points.csv", rows, ("ratio", "decision", "votes", "ttft_gain_pct",
                                                      "throughput_gain_pct", "directory", "reason"))
    measurements.csv_file(root / "pairs.csv", [pair for p in visible for pair in p["pairs"]],
                          ("id", "ratio", "replicate", "mode_order", "status", "reason",
                           "off_ttft_ms", "slot_ttft_ms", "off_input_tps", "slot_input_tps",
                           "ttft_gain_pct", "throughput_gain_pct"))
    plotted = [p for p in sorted(visible, key=lambda p: p["ratio"]) if "ttft_gain_pct" in p]
    measurements.render_svg(plotted, root / "curve.svg", cfg["practical_gain_pct"])
    lines = ["# 8卡1.00–1.75自适应摸测（探索性）", "",
             "每个点两对独立服务，分别判断TTFT和输入吞吐是否同时改善至少3%。",
             "两对均达标：gain；均不达标：no_gain；一对达标一对不达标：uncertain并停止。",
             "表中平均收益仅供描述；缩区间使用每对分别投票，不用平均值掩盖分歧。",
             "", "| s | 点判定 | 两对投票 | 平均TTFT改善 | 平均输入吞吐改善 |",
             "|---:|---|---|---:|---:|"]
    for p in visible:
        a = f"{p['ttft_gain_pct']:+.2f}%" if "ttft_gain_pct" in p else "—"
        b = f"{p['throughput_gain_pct']:+.2f}%" if "throughput_gain_pct" in p else "—"
        lines.append(f"| {p['ratio']:.6f} | {p['status']} | {p.get('votes','—')} | {a} | {b} |")
    lines += ["", "当前决策：" + json.dumps(action, ensure_ascii=False), "",
              "若bracket_found，在近似单调假设下候选阈值位于(low,high]，不是精确阈值。",
              "若上端未达标，只能说未建立二分区间，不能断言整个区间都无收益。",
              "若uncertain、非单调、数据失败或资源干扰，停止，不自动增加重复或丢弃数据。",
              "连接线仅辅助阅读，不代表未测点的实际性能；每服务内轮次不算独立重复。",
              "", "![样本收益](curve.svg)", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    return summary


def execute(cfg, resume=None):
    if not cfg.get("design_confirmed"):
        raise RuntimeError("Design not confirmed.")
    if study.read(study.HERE / "strategy.json") != logic.strategy(cfg):
        raise RuntimeError("Strategy file is stale; regenerate before execution.")
    runtime.preflight(cfg["runtime"])
    geometry = study.read(Path(cfg["runtime"]["model"]) / "config.json")
    if geometry.get("n_routed_experts") != 256 or geometry.get("num_experts_per_tok") != 6:
        raise RuntimeError("Model geometry differs from the routing family.")
    strategy = logic.strategy(cfg)
    with (study.HERE.parent / ".dp8_crossover_port_8444.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume is None:
            stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
            root = study.HERE / "data" / f"bisect_{stamp}"
            root.mkdir(parents=True, exist_ok=False)
            for name in resources.SOURCE_FILES:
                shutil.copy2(study.HERE / name, root / name)
            manifest = {"settings": cfg, "strategy": strategy, "sources": resources.source_hashes(),
                        "environment_signature": resources.environment_signature(cfg["runtime"], root)}
            runtime.write_json(root / "manifest.json", manifest)
        else:
            root = Path(resume).resolve()
            manifest = study.read(root / "manifest.json")
            if manifest["settings"] != cfg or manifest["strategy"] != strategy:
                raise RuntimeError("Resume settings or strategy changed.")
            if manifest["sources"] != resources.source_hashes():
                raise RuntimeError("Scripts/settings changed; resume refused.")
            stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
            current = resources.environment_signature(cfg["runtime"], root / "resume_checks" / stamp)
            if current != manifest["environment_signature"]:
                raise RuntimeError("Runtime environment changed; resume refused.")
        print("RESULTS", root, flush=True)

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Interrupted by signal {signum}")

        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            while True:
                history, pending = history_from_disk(root, cfg)
                action = logic.next_action(history, cfg)
                if action["action"] == "stop":
                    print("STOP", json.dumps(action, ensure_ascii=False), flush=True)
                    break
                index = len(history) + 1
                if pending:
                    step, plan, _ = pending
                else:
                    step = root / "points" / f"{index:02d}_s{study.ratio_key(action['ratio'])}"
                    step.mkdir(parents=True, exist_ok=False)
                    plan = point_plan(cfg, index, action["ratio"])
                    runtime.write_json(step / "plan.json", plan)
                    for descriptor in plan["routes"]:
                        study.write_new(step / "sim" / descriptor["filename"], descriptor["text"])
                print("MEASURE", index, action, flush=True)
                routes = {route["name"]: route for route in plan["routes"]}
                for pair in plan["pairs"]:
                    if measurements.collect_pair(step, pair, plan)["status"] == "complete":
                        print("SKIP_COMPLETE_PAIR", pair["id"], flush=True)
                        continue
                    resources.archive_partial_pair(step, pair)
                    for case in pair["cases"]:
                        runtime.run_case(cfg["runtime"], case,
                                         resources.archived_route(step, routes[case["route"]]), step)
                report(root, cfg)
        finally:
            signal.signal(signal.SIGTERM, previous)
            report(root, cfg)
            print("REPORT", root / "REPORT.md", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--report", type=Path, help="Rebuild a recorded session report only; no NPU.")
    args = parser.parse_args()
    cfg = logic.load_settings()
    if args.report:
        if args.execute or args.resume:
            parser.error("--report cannot combine with execution/resume")
        saved = study.read(args.report / "manifest.json")["settings"]
        print(json.dumps(report(args.report, saved)["decision"], ensure_ascii=False, indent=2))
        return
    if not args.execute:
        print(json.dumps({"strategy": logic.strategy(cfg), "workloads_started": False},
                         ensure_ascii=False, indent=2))
        return
    execute(cfg, args.resume)


if __name__ == "__main__":
    main()
