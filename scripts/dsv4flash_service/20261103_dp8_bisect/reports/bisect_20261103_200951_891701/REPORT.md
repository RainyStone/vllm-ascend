# 8卡1.00–1.75自适应摸测（探索性）

每个点两对独立服务，分别判断TTFT和输入吞吐是否同时改善至少3%。
两对均达标：gain；均不达标：no_gain；一对达标一对不达标：uncertain并停止。
表中平均收益仅供描述；缩区间使用每对分别投票，不用平均值掩盖分歧。

| s | 点判定 | 两对投票 | 平均TTFT改善 | 平均输入吞吐改善 |
|---:|---|---|---:|---:|
| 1.750000 | gain | [True, True] | +13.91% | +14.88% |
| 1.000000 | gain | [True, True] | +8.48% | +12.05% |

当前决策：{"action": "stop", "status": "lower_already_gain", "bracket": null, "reason": "The smallest allowed ratio already gains; no fail-to-gain bracket here."}

若bracket_found，在近似单调假设下候选阈值位于(low,high]，不是精确阈值。
若上端未达标，只能说未建立二分区间，不能断言整个区间都无收益。
若uncertain、非单调、数据失败或资源干扰，停止，不自动增加重复或丢弃数据。
连接线仅辅助阅读，不代表未测点的实际性能；每服务内轮次不算独立重复。

![样本收益](curve.svg)
