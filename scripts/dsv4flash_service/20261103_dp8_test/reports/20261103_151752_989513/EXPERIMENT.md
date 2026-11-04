# DP8 测试（2026-11-03）

目标为 141.61.45.118 的 xzx_moonep 容器，使用物理卡0..7、DP8/TP1/EP8。
保留DP4目录和结果，所有新数据单独写入本目录data/时间戳目录。

默认配置沿用最近确认的8K输入/8K token预算、并发16、请求32、4GiB KV cache、
max-model-len=8193、slot三项优化和off对照。性能模式先运行slot再运行off，
每组小预热后测3轮，统计后2轮。是否恢复性能对比以用户当前选择为准。

check_slot_requests.py --execute只检查自然路由短请求，不跑性能矩阵：
清除SIM_ROUTE，用slot发送两个最多16输出token的completion请求，
校验HTTP、非空输出、8个rank的dispatch/combine/prefetch计数、slot运行时audit；
结束后停止自身服务并确认卡资源释放。诊断计数带额外开销，不作为性能数据。

DP8的性能路由另行生成，不能把DP4的3hot1r档案原样当作s≈2.48：
DP8每rank拥有32个专家，DP4档案的expert 0/21/42会落到两个rank，峰均比会改变。
本目录sim/sim_route_rank_skew_2p48_dp8.txt将两个热点放到rank0（expert 0和16），
每个热点占10.895172557877%，其余254个专家均分剩余份额。
目标rank0份额31.02770593%，其余7rank各9.85318487%，峰均比2.4822164743。
保持rank级目标峰均比；专家级分布不同，DP4/DP8结果不当作仅改变卡数的严格对照。

预览：
```bash
python check_slot_requests.py
python run_experiment.py --plan
```

明确执行：
```bash
python check_slot_requests.py --execute
# 仅在用户确认恢复性能对比后：
python run_experiment.py --execute
```

无任何脚本会停止其他容器或未知进程。只检查与使用0..7卡，发现已有NPU进程或端口占用则不启动。

用户已确认恢复s≈2.48、8K的off/slot性能对比。实际执行仅两组，使用全8卡DP8/EP8、KV每卡4GiB、max-model-len8193、token预算8192、并发16、32请求、预热后测3轮取后2轮。
DP8自然路由前置检查data/slot_requests_20261103_151354_532685的两个短请求均HTTP200，8个rank的dispatch/combine计数增长，自动路径校验通过。该检查开启诊断计数，不并入性能结果；性能进程重新起服并清除诊断环境变量。
