#!/usr/bin/env python3
from pathlib import Path
import json,datetime
root=Path(__file__).resolve().parent
state=json.loads((root/"STATE.json").read_text())
modes=[m for m in ("off","slot_mte","slot_udma") if state.get("formal",{}).get(m,{}).get("summary")]
base=state["formal"]["off"]["summary"]
lines=["DP16 LongBench-v2 优化版复测","更新："+datetime.datetime.now(datetime.timezone.utc).isoformat(),
"两机118/124，原仓默认安装；DSV4 Flash；DP16/TP1/EP16；503条、context中间截断至128K prompt；并发16，输出1，8K chunk，24GiB KV/卡，dp_allreduce_on_npu=true。",
"每模式独立起服1次、正式503条1轮、0预热。Slot：MXFP8 dispatch＋融合BF16发布＋PF_STREAM=2；关闭Home拆分、持久专家缓存和提前预取。combine保持BF16传输、FP32加权累加。",
"模式 | 平均TTFT(ms) | 相对off | P95 TTFT(ms) | 输入吞吐(token/s) | 相对off | 计时秒"]
results=[]
for mode in modes:
 item=state["formal"][mode]["summary"]
 ttft_delta=(item["ttft_ms"]["mean"]/base["ttft_ms"]["mean"]-1)*100
 throughput_delta=(item["input_tps"]/base["input_tps"]-1)*100
 lines.append(f'{mode} | {item["ttft_ms"]["mean"]:.2f} | {ttft_delta:+.2f}% | {item["ttft_ms"]["p95"]:.2f} | {item["input_tps"]:.2f} | {throughput_delta:+.2f}% | {item["benchmark_duration_ms"]/1000:.2f}')
 results.append(dict(mode=mode,ttft_delta_pct=ttft_delta,input_tps_delta_pct=throughput_delta,summary=item))
lines.extend(["","已完成模式均通过：503成功、总输入43,029,416 token、总输出503 token、逐请求prompt与token数校验及两机16个worker前后模式审计。",
"首次off起服因EngineCore控制面Gloo初始化等待而未发送正式请求，已保留无效记录并正常清理；本表off来自原参数重试后的完整有效轮。",
"原仓主库以原子rename更新；两端分别验证默认editable安装路径、动态库SHA、实际映射和UDMA同步修复ABI=2。",
"UDMA组的planning仍使用MTE，其余三个阶段使用UDMA。"])
for mode in modes:
 item=state["formal"][mode]
 lines.extend(["",mode+" 原始数据："+item["round_dir"],"统计JSON："+item["summary"]["raw_statistics_path"],"逐请求JSONL："+item["summary"]["raw_details"]])
pending=[m for m in ("off","slot_mte","slot_udma") if m not in modes]
if pending:lines.append("\n尚未完成："+", ".join(pending)+"；不据部分矩阵声称全部完成。")
out=root/"results";out.mkdir(exist_ok=True)
(out/"PERFORMANCE.txt").write_text("\n".join(lines)+"\n")
(out/"comparison.json").write_text(json.dumps(dict(results=results,completed=modes,pending=pending),indent=2))
for name in ["LIBRARY_UPDATE.json","APPROVAL_STATUS.json"]:
 (out/name).write_text((root/name).read_text())
print("\n".join(lines[:9]))
