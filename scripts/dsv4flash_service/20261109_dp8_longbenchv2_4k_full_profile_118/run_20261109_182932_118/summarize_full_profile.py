import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MODES=("off","slot")

STAGES=(
 ("本地稀疏注意力内核","sparse_attention_kernel_us"),
 ("Router GEMM","router_gemm_us"),
 ("TopK 路由","topk_us"),
 ("Dispatch 整段","dispatch_span_us"),
 ("其中输入激活动态量化","input_quant_kernel_us"),
 ("专家 GEMM1","gemm1_us"),
 ("激活及 GEMM2 前量化","activation_quant_span_us"),
 ("专家 GEMM2","gemm2_us"),
 ("Combine 整段","combine_span_us"),
 ("路由专家主流合计","routed_wall_us"),
 ("共享专家","shared_expert_span_us"),
 ("共享与路由输出合并","routed_shared_merge_us"),
 ("Router 至合并完整跨度","router_to_merge_us"),
)
DETAILS=(
 ("Planning 内核","planning_kernel_us"),
 ("MXFP8 pack","pack_kernel_us"),
 ("Dispatch 内核","dispatch_kernel_us"),
 ("MXFP8 unpack","unpack_kernel_us"),
 ("融合 publish","fused_publish_kernel_us"),
 ("FP32 加权 combine 内核","weighted_combine_kernel_us"),
 ("副流预取内核活跃时间","prefetch_active_us"),
 ("副流预取跨度","prefetch_span_us"),
 ("副流预取与 dispatch 内核的交叠","prefetch_overlaps_dispatch_kernel_us"),
 ("Dispatch 内核结束后的预取尾部","prefetch_tail_after_dispatch_us"),
 ("Unpack 结束后的预取尾部","prefetch_tail_after_unpack_us"),
)

def stat(v,k,f="mean_ms"):
    s=v.get(k)
    return f"{s[f]:.4f}" if s else "—"

def main():
    analysis=json.loads((ROOT/"STAGE_PROFILE_RESULTS.json").read_text())
    assert set(analysis)==set(MODES)
    checks={}
    for mode in MODES:
        proof,=(ROOT/"formal/data/formal/aisbench").glob(f"*-{mode}-*/round_1/validation.json")
        v=json.loads(proof.read_text())
        assert v["valid"] and v["requests"]==316,v
        marks=[json.loads(line) for line in (proof.parent/"profile_events.jsonl").read_text().splitlines()]
        assert [x["step"] for x in marks]==["warmup_start","warmup_validated","profile_started","aisbench_complete",
                                               "profile_stop_requested","profile_stopped"],marks
        assert marks[1]["requests"]==16
        assert len(analysis[mode]["inventory"])==8
        checks[mode]=dict(validation=str(proof),profile_events=str(proof.parent/"profile_events.jsonl"),
                          debug_log=str(proof.parent/"aisbench_debug.log"),start=marks[2]["at"],stop=marks[-1]["at"])
    off=analysis["off"]["stages"]["native_alltoall:sparse_attention"]
    slot=analysis["slot"]["stages"]["slot_mte:sparse_attention"]
    rows=[
        "# 118 单机 DP8 TP1 LongBench-v2 4K 全轮 profiling","",
        "两组先预检最长样本并各预热 16 条，再开启设备采集，完整回放 316 条原文。并发 16、输出 1 token。Off 使用原生路径，Slot 使用 MTE、副流 PF_STREAM=2、MXFP8 dispatch 和融合 publish。每组八个 rank 原始 trace 保留。","",
        "以下单位为 ms/次 MoE 层×rank 本地调用。主表取有本地稀疏注意力标记、且阶段边界可解析的调用，Off 取 AllToAll，Slot 取 MTE。两组自然调度，按各自调用分布统计。完整录制中其他调用另见机器可读汇总。","",
        "| 阶段 | Off 均值 | Slot 均值 | Slot−Off | Off P50 / P95 | Slot P50 / P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label,k in STAGES:
        delta=f"{slot[k]['mean_ms']-off[k]['mean_ms']:+.4f}" if k in off and k in slot else "—"
        rows.append(f"| {label} | {stat(off,k)} | {stat(slot,k)} | {delta} | {stat(off,k,'p50_ms')} / {stat(off,k,'p95_ms')} | {stat(slot,k,'p50_ms')} / {stat(slot,k,'p95_ms')} |")
    rows+=["",f"主表的完整路由专家主流调用数为 Off {off['routed_wall_us']['n']:,}、Slot {slot['routed_wall_us']['n']:,}。各细项有效样本数见 JSON。","",
           "## Slot 内部及副流","",
           "| 阶段 | 均值 | P50 | P95 | 最大 | 有效调用 |","|---|---:|---:|---:|---:|---:|"]
    for label,k in DETAILS:
        if k in slot:
            s=slot[k]
            rows.append(f"| {label} | {s['mean_ms']:.4f} | {s['p50_ms']:.4f} | {s['p95_ms']:.4f} | {s['max_ms']:.4f} | {s['n']:,} |")
    rows+=["","主流 dispatch 和 combine 整段含相关准备、同步和等待，内核子项包含在整段内。副流预取的交叠和尾部按同一 rank 的设备事件计算，单独报告。采集轮受到 profiler 开销影响，正式端到端性能见 PLAN.json 记录的历史对照。","",
           "## 原始 trace 与覆盖",""]
    for mode in MODES:
        info=analysis[mode]["inventory"]
        rows.append(f"- {mode}，8 rank，识别 {sum(x['parsed_calls'] for x in info):,}/{sum(x['gate_count'] for x in info):,} 个 TopK 对应的调用。导出时间线共 {sum(x['trace_bytes'] for x in info)/1024**3:.2f} GiB。")
    rows+=["","[统计和八 rank trace 路径](STAGE_PROFILE_RESULTS.json) · [采集验证与 debug 日志](CAPTURE_VALIDATION.json)",""]
    (ROOT/"CAPTURE_VALIDATION.json").write_text(json.dumps(checks,indent=2)+"\n")
    (ROOT/"STAGE_PROFILE_REPORT.md").write_text("\n".join(rows)+"\n")
    print("\n".join(rows))

if __name__=="__main__":
    main()

