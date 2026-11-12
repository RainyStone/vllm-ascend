单卡算子微基准：124的物理0卡；不启动模型、不初始化SHMEM/HCCL，不采负载。
对比分库与统一设备库，保留内核源码和数据/stream流程；各3次独立进程，顺序split→unified→unified→split→split→unified。每例预热10次、7批×50次，记录设备事件与主机墙钟耗时。
覆盖wire打包/全量及有效行解包、publish及NaN尾部、Home/Slot BF16和MXFP4 GEMM1；原始JSON和日志保存在raw/，库哈希及加载路径写入每轮记录。
其他卡上的任务可能带来功耗/频率噪声；本结果不能替代多卡通信或模型端到端性能结论。

初始三对结果全部保留。MXFP4 prefill完整范围在同一进程内出现随批次从约250us到300us的变化，因此额外对该点执行500次预热、9批×100次的同机交叉三对复核，使用独立raw/focus_*输出；不选择性删除初始点。初始脚本快照保存在artifacts/initial_bench_unified_kernels.py。

脚本需要本次保存的artifacts/split/和artifacts/unified/产物，库哈希见PROVENANCE.json/RESULTS.json；这些二进制及raw原始数据仅保留在共享盘工作区，不提交Git。这里只测BF16 dense ND与MXFP4 NZ样本，未新增BF16 NZ/转置等格式的设备复测。
