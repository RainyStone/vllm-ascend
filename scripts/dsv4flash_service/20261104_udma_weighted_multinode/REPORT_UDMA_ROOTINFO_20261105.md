# 双机物理1号卡 UDMA 拓扑补齐后验证（2026-11-05）

结果：通过。118、124的xzx_moonep容器分别使用各自物理NPU 1，共2个rank。
每rank 39项检查，共78项，逐元素差异0、最大绝对差0；无NaN/Inf。
每rank实际调用CombineUrma.compute_weighted 22次，所有有效路由均指向另一物理节点。

## 本次变更
将118宿主机/etc/hccl_rootinfo.json复制到118容器同名位置；
将124宿主机/etc/hccl_rootinfo.json复制到124容器同名位置。
两端rootinfo内容不同，分别核对宿主机/容器SHA一致。
引用的/usr/local/Ascend/driver/topo/950/atlas_850_1.json已通过驱动目录挂载可见，SHA一致。
未修改宿主机拓扑文件，未增加挂载，未修改主板0x27的源码策略。

## 实际后端证据
两端SHMEM日志明确记录：
- Load rootinfo from file /etc/hccl_rootinfo.json
- Resolved phy id ... phy_id=1
- Create HCOMM channel success
- Build UDMA info success, rank_count=2, totalChannelNum=2, qp_num=2
结合真实CombineUrma调用记录、双向跨机路由和数值参考通过，确认本轮UDMA路径已成功运行。

## 数值用例
CPU上按FP32乘系数/累加、最终一次BF16舍入的参考，以及本地NPU原生weighted-unpermute参考均通过。
覆盖T=1/17/32、K=1/6/8/16、别名、无效路由、零系数、不等真实T、NaN尾部与工作区复用。
旧非加权combine回归也通过。

## 对此前失败的修正结论
之前容器缺少rootinfo，自动生成分支无法识别主板0x27。
宿主机已有有效rootinfo。复制本机文件到各自容器后，无需修改拓扑策略即可完成本轮UDMA测试。
不能将此前配置缺失导致的初始化失败解释成硬件不支持UDMA。

## 范围和清理
本轮仅为2-rank synthetic combine正确性验证，没有加载模型或测模型性能。
planning、dispatch、专家权重prefetch及16-rank仍未验证。
原仓Python源码与原仓build中新构建库仅在新测试进程加载；原始安装位置的运行库SHA保持不变。
两端测试进程已正常退出；卡状态及完整日志见cleanup_node118.json/cleanup_node124.json和npu_after文件。
未停止其他用户进程，未复位设备，未启动DP16，未恢复定时任务。

## 记录
node0.log / node1.log：测试输出；
rank0.json / rank1.json：每rank数值和实际调用记录；
shmem_node0/ / shmem_node1/：完整SHMEM初始化、通道、销毁日志；
rootinfo_copy_metadata.json：复制与SHA信息；
fingerprints.json：测试源码和构建库SHA；
summary.json、combine_validation.json：汇总及验证凭证。

实验目录：/mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261104_udma_weighted_multinode/results/cross2_card1_rootinfo_20261105_114548_udma
