> 状态：本1.00–2.50粗扫/细扫方案已于2026-11-03被用户的新指令替代，尚未执行。
> 当前有效方案：../20261103_dp8_bisect/EXPERIMENT.md，范围1.00–1.75，候选区间宽≤0.05，分歧即停止。旧计划仅存档，旧执行入口已禁用。

# 8卡 MoonEP slot 收益交叉区间实验

准备日期：2026-11-03。主机141.61.45.118，容器xzx_moonep。
本次只设计、生成计划和做纯逻辑验证。没有启动服务、没有压测、没有占卡；实际执行等用户通知。

## 用户已确认的口径

- 收益：TTFT和输入吞吐均改善至少3%。
- 范围：s=1.00～2.50，粗扫步长0.25，另加已测的2.4822164743点。
- 每个点两对独立重启的off/slot服务，仅作探索性结论；不做额外五对确认。
- 细扫只在候选交叉区间内，步长0.05；每点仍两对。
- 固定已跑通的配置：物理卡0..7、DP8/TP1/EP8、8K输入、8K token预算、输出1 token、
  并发16、每轮32请求、每卡4GiB KV、max-model-len=8193。
- slot固定开启PREFETCH_RELAXED、NO_RECONCILE、SKIP_DUP三项历史优化；
  off关闭MoonEP。两模式都开启EP，不开启profiling或诊断计数。
- 模型为/mnt/share/weight/DeepSeek-V4-Flash。

## 为什么固定一条路由族

s定义为专家原始归属下最大home-rank路由份额/平均份额，平均份额为12.5%。
这不是单专家峰均比，也不是slot均衡后的实测计算峰均比。

固定E=256、K=6、R=8，每rank32专家。热点始终为rank0的expert 0、16；
除这两个专家外，其余254个专家均分剩余份额，只随s连续改变热点份额。
这样避免不同点同时改变热点个数、位置和rank数量。

令热点单专家百分比为p，rank0目标份额为12.5s，则：
12.5s = 2p + 30(100-2p)/254。
脚本据此生成档案，并验证份额和为100、单专家不超过100/K、目标s准确。

s=1时所有专家均匀；s=2.4822164743与上一批成功DP8实验属于相同两热点路由族。
上一批数据只作背景参考，本批重新测量，不把旧单对结果当作新的独立重复。
该路由族最大可行s约3.30；本次按用户选择仅测到2.50。

有限token数会导致整数舍入，尤其T=1/16等dummy形状不等于目标s。
计划中保存T=16、128、1024、8192、65536时的整数分配估计，原始服务日志同时保留。
报告明确标注目标home-rank比值，不把dummy形状偏差解释为扫描参数改变。

## 阶段1：粗扫

点位：
1.00、1.25、1.50、1.75、2.00、2.25、2.4822164743、2.50。

8点×2独立服务对×2模式=32次起服。
每台服务先做2请求×1024输入预热，再测3轮32请求×8192输入；
丢弃第1轮，统计第2/3轮。因此共128次bench调用（含预热）。

每个点的一对先off后slot，另一对先slot后off；比例点的顺序固定随机化并保存。
每对相邻执行，减少时间漂移；不会把所有off跑完后再跑slot。
同一对的服务seed、route seed、bench seed、请求形状均相同。
不同对使用不同bench seed块；相同对序号在不同s使用相同prompt seed，控制请求差异。

根据此前两台服务约10分钟的实际运行经验，粗扫约2.5～3小时，仅为估计。

## 阶段2：候选区间细扫

粗扫完成后，由plan_sweep.py读取新session的summary.json：
- 找到“未达到双指标3%→达到双指标3%”的相邻区间；
- 在该区间保留端点、按0.05插入点位，每点两对独立服务；
- 若区间宽0.25，通常6点，24次起服、96次bench调用，约2小时；
- 粗扫若全部达标，只能说明最低测试点已获益，优先复测最小区间；
- 若测到2.50仍没有候选收益，不强行生成阈值；
- 若出现回落、多个交叉或不完整数据，拒绝自动挑一个区间，需要回看曲线后显式指定范围；
- 单次细扫最多8个点，防止误指定范围产生大量起服。

阶段不会自动串联；细扫计划生成也只写文件，不启动服务。
不默认二分搜索，不假设收益曲线单调。

## 统计和结果解释

每台服务先取第2/3轮均值，再组成一个off/slot配对收益；两对才是两个独立重复。
不能把服务内的四个保留轮次当作四个独立服务样本。

对每对计算：
- TTFT log速度比：log(TTFT_off/TTFT_slot)；
- 输入吞吐log速度比：log(tps_slot/tps_off)。

对两个服务对取速度比的几何平均，再转换成TTFT降低百分比和吞吐提高百分比。
CSV也保留两组原始均值、每对收益、mode先后顺序和同向获益比例，避免均值掩盖波动。

探索性达标点要求两项配对几何平均收益都至少3%，同时展示是否两对都同向获益。
仅两对不报告置信区间、统计显著性或“稳定阈值已确认”；confirmed_gain固定不成立。
报告给出候选交叉区间、最小探索性达标点以及是否观察到回落。
该区间只针对本机、固定两热点路由族、当前8K/KV/并发配置，不宣称更高s全部获益。
本实验使用合成路由，不作为自然路由数值正确性验证。

只统计双方完整成功、每轮32请求、输入262144 token、输出32 token、配对seed和路由hash一致的数据。
失败/缺失点不填0、不丢掉坏重复后单独报告好的重复，也不跨旧参数session拼接。

## 文件

- settings.json：用户确认的固定配置、范围、收益判据、两个独立重复。
- study.py：路由数学、计划构造、配对统计和边界识别，无GPU依赖。
- runtime.py：从已成功执行的DP8脚本冻结的运行库；直接执行会退出，不启动服务。
- PROVENANCE.json：运行库来源和SHA256，以及seed变更说明。
- plan_sweep.py：生成粗扫/细扫计划，不运行模型。
- run_sweep.py：默认仅预览，显式--execute才会起服。
- analyze.py：离线生成pairs.csv、curve.csv、curve.svg/curve.png、summary.json、REPORT.md。
- test_study.py：合成数据自检，绘图使用容器已有Matplotlib；不导入torch、不查询NPU、不起服务。
- plans/coarse.json：当前粗扫的固定顺序、32个运行项及路由全文/hash。
- data/：预留真实实验数据；目前只有README.txt。

## 现在允许做的操作：预览和纯逻辑检查

在xzx_moonep容器中：

```bash
cd /mnt/share/x00893896/vllm_ascend_moonep_integration/vllm-ascend/scripts/dsv4flash_service/20261103_dp8_crossover

python plan_sweep.py coarse
python run_sweep.py
PYTHONDONTWRITEBYTECODE=1 python -m unittest -v test_study.py
```

这些命令不探测NPU、不加载模型、不发送推理请求。

## 等用户通知后才执行

```bash
# 粗扫；--execute必须显式提供：
python run_sweep.py --plan plans/coarse.json --execute

# 粗扫完成后仅生成细扫计划：
python plan_sweep.py fine --from-session data/<粗扫session>

# 需要显式选择区间时：
python plan_sweep.py fine --from-session data/<粗扫session> --range 1.50 1.75 --out plans/fine_selected.json

# 审阅细扫计划后再明确执行：
python run_sweep.py --plan plans/fine.json --execute

# 原环境断点续跑：
python run_sweep.py --resume data/<session> --execute

# 单独重新汇总，不使用NPU：
python analyze.py data/<session>
```

## 资源保护、归档与续跑

执行前检查8卡空闲、健康、网卡IP、模型文件和8444端口；只终止自身创建的服务进程组。
每轮前后检查所选卡worker进程集合，保存NPU快照；发现干扰或OOM即停止。
不启动占卡器、不终止其他任务、不自动调整KV/并发/预算来凑成功数据。

每个session冻结计划、脚本、配置、路由全文/hash、仓库HEAD/修改状态、pip freeze、
模型配置/索引hash、已编译扩展hash、CANN/driver信息和相关环境摘要。
环境摘要只存hash，不把完整进程环境或凭据写入报告。

续跑只跳过完整成功的整对；若一对只有半边完成，保留旧文件到attempts/后整对重跑，
避免跨很长时间拼一个off/slot对。脚本、settings、Git、安装包、模型索引、扩展或相关环境改变时拒绝续跑。

不自动合并粗扫和细扫的样本；各阶段单独报告，计划记录来源summary的hash。
本次未执行任何硬件实验，等待用户下一次明确通知。

## 本次交付验证

8项纯逻辑测试通过；语法检查通过，测试进程未导入torch/torch_npu。详见VALIDATION.txt。
粗扫的32条运行清单见RUNS.txt；执行时以带hash的plans/coarse.json为准。
本次未启动服务、未发送推理或压测请求、未查询或占用NPU；data/仍只有说明文件。等待用户通知执行。
