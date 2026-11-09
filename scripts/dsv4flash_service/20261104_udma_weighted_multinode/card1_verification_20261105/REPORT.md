# 双机物理1号卡小算子验证（2026-11-05）

本轮使用141.61.45.118、141.61.45.124各自的物理NPU 1，共2个rank。
脚本固定CARD_ID=1；只在1号卡空闲时启动，禁止自动换卡。
首次按“各一张空卡”选择0号卡的数据仅作为补充保留，不计入本报告。旧UDMA等待作业已仅对本任务停止。

## 结果
- MTE：通过。两个rank各39项检查，共78项；22次加权归并入口调用/rank，所有有效路由指向另一台机器。
- FP32 CPU参考、NPU原生加权unpermute参考、旧非加权combine回归均通过；最大绝对差0，逐元素差异0。
- 覆盖T=1/17/32、K=1/6/8/16、负编码别名、越界路由、零系数、不等真实token数、NaN填充尾部、工作区重复使用。
- UDMA：未通过初始化，尚未运行归并内核或进行UDMA数值比较。两端均在aclshmemx_init_attr_with_buffers返回-3（ACLSHMEM_SMEM_ERROR）。
- 未加载模型、未启动DP16、未测吞吐；定时任务保持停用。

## UDMA直接失败原因
两端本任务的SHMEM原始日志都记录：
    topo_addr_info_get failed: strategy creation failed for phy_id=1, mainboard_id=0x27
    Failed to get generated rootinfo for phyId 1
    Failed to parse the rootinfo file.
    PrepareOpenDevice failed.

两端容器均没有/etc/hccl_rootinfo.json，因此库走自动生成rootinfo分支。
当前shmem源码的product strategy工厂没有0x27板型；运行日志也确实在该策略创建处失败。
此错误先于UDMA通信端点/数据搬运建立，不能据此认定硬件不支持UDMA，也不能归因为融合归并精度问题。

## 后续适配路径
需要取得与0x27实际硬件相符的拓扑信息，提供正确rootinfo，或新增匹配的product strategy。
适配要根据实际UE/EID、端口、网络层/平面和驱动拓扑确定，不能直接把0x27冒充其他板型。
之后再用物理1号卡复测UDMA初始化和相同数值用例。
原生MTE的2-rank combine通过不能替代planning、dispatch、量化专家权重prefetch或16-rank验证。

## 文件
mte/：两端启动脚本快照、node0.log/node1.log、rank0.json/rank1.json、combine_validation.json和库SHA。
udma/：两端启动脚本快照、失败日志、两份原始SHMEM日志、initialization_failure.json及清理凭证。
summary.json：本轮结果与范围。

## 环境与清理
使用原仓Python源码及原仓build/weighted_udma_20261104中新编译库，仅新测试进程加载。
原始安装位置的运行库SHA保持不变。
所有本次测试进程已退出，两机1号卡已确认空闲；未停止其他用户进程，未复位设备。
