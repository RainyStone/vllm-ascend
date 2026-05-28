import math
from abc import ABC, abstractmethod
from typing import Dict

class BaseLoadCalculator(ABC):
    """负载计算策略抽象基类"""

    @abstractmethod
    def calculate(self, metrics: Dict[str, float], estimated_tokens: float = 0.0) -> float:
        pass


class DefaultLoadCalculator(BaseLoadCalculator):
    """默认负载计算策略：基于 KV cache 使用率 + 运行请求数 + 等待队列"""

    def __init__(self, kv_weight=0.6, running_weight=0.3, waiting_penalty=0.1, max_concurrent=256):
        self.kv_weight = kv_weight
        self.running_weight = running_weight
        self.waiting_penalty = waiting_penalty
        self.max_concurrent = max_concurrent

    def calculate(self, metrics: Dict[str, float], estimated_tokens: float = 0.0) -> float:
        # 原有实现完全不变，estimated_tokens 在百分比模式下暂不参与
        kv_usage = metrics.get("kv_cache_usage_perc", 0.0)
        running = metrics.get("num_requests_running", 0.0)
        waiting = metrics.get("num_requests_waiting", 0.0)

        # 归一化运行请求数
        running_norm = min(running / self.max_concurrent, 1.0)
        waiting_flag = 1.0 if waiting > 0 else 0.0

        load = (kv_usage * self.kv_weight +
                running_norm * self.running_weight +
                waiting_flag * self.waiting_penalty)
        # 确保在 [0,1] 范围内
        return max(0.0, min(load, 1.0))


class RunningRequestsOnlyCalculator(BaseLoadCalculator):
    """仅使用运行请求数（简单策略）"""

    def __init__(self, max_concurrent=256):
        self.max_concurrent = max_concurrent

    def calculate(self, metrics: Dict[str, float], estimated_tokens: float = 0.0) -> float:
        running = metrics.get("num_requests_running", 0.0)
        return min(running / self.max_concurrent, 1.0)


class KvCacheAwareCalculator(BaseLoadCalculator):
    """
    基于绝对 KV cache 容量的请求感知负载计算器。
    当 total_blocks > 0 时，基于剩余 KV cache 绝对 token 数计算负载；
    否则回退到百分比模式。
    """

    def __init__(self, total_blocks: int, block_size: int, max_num_seqs: int):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs

    # def calculate(self, metrics: Dict[str, float], estimated_tokens: float = 0.0) -> float:
    #     kv_usage_perc = metrics.get("kv_cache_usage_perc", 0.0)
    #     running = metrics.get("num_requests_running", 0.0)
    #     waiting = metrics.get("num_requests_waiting", 0.0)
    #
    #     # 未配置总容量时回退到百分比模式
    #     if self.total_blocks <= 0:
    #         running_norm = min(running / self.max_num_seqs, 1.0)
    #         waiting_flag = 1.0 if waiting > 0 else 0.0
    #         return kv_usage_perc * 0.6 + running_norm * 0.3 + waiting_flag * 0.1
    #
    #     # TODO 负载计算公式需要优化
    #     # 绝对数量计算
    #     used_blocks = kv_usage_perc * self.total_blocks
    #     remaining_blocks = self.total_blocks - used_blocks
    #     remaining_tokens = remaining_blocks * self.block_size
    #
    #     # 等待队列惩罚，等待越多，惩罚越大，封顶 0.3
    #     queue_penalty = min(waiting / 8.0, 1.0) * 0.3
    #
    #     # 请求感知：装不下则返回超载
    #     if estimated_tokens > 0 and estimated_tokens > remaining_tokens:
    #         return 2.0
    #
    #     # KV 容量压力，非线性映射
    #     free_ratio = remaining_blocks / self.total_blocks if self.total_blocks > 0 else 1.0
    #     load_score = math.exp(-3.0 * free_ratio)
    #
    #     # 算力压力，KV 很空时，算力权重上升；KV 很满时，算力权重下降
    #     compute_weight = 0.3 * (1.0 - kv_usage_perc)  # 动态权重
    #     compute_pressure = min(running / self.max_num_seqs, 1.0) * compute_weight
    #
    #     return min(load_score + compute_pressure + queue_penalty, 2.0)

    def calculate(self, metrics: Dict[str, float], estimated_tokens: float = 0.0) -> float:
        kv_usage_perc = metrics.get("kv_cache_usage_perc", 0.0)
        running = metrics.get("num_requests_running", 0.0)
        waiting = metrics.get("num_requests_waiting", 0.0)

        # 未配置总容量时回退到百分比模式
        if self.total_blocks <= 0:
            running_norm = min(running / self.max_num_seqs, 1.0)
            waiting_flag = 1.0 if waiting > 0 else 0.0
            return kv_usage_perc * 0.6 + running_norm * 0.3 + waiting_flag * 0.1

        # TODO 负载计算公式需要优化

        # KV 容量压力，非线性映射
        # Sigmoid: 在 50% 使用率附近最敏感，空载≈0，满载≈1
        # ┌────────┬────────┬──────────────┐
        # │ 使用率  │ 压力值  │    直观感受    │
        # ├────────┼────────┼──────────────┤
        # │ 0 %    │ ~0.00  │ 完全空闲       │
        # ├────────┼────────┼──────────────┤
        # │ 25 %   │ ~0.05  │ 轻微使用      │
        # ├────────┼────────┼──────────────┤
        # │ 50 %   │ 0.50   │ 拐点，中等压力 │
        # ├────────┼────────┼──────────────┤
        # │ 75 %   │ ~0.95  │ 高度紧张      │
        # ├────────┼────────┼──────────────┤
        # │ 90 %   │ ~0.998 │ 几乎满载      │
        # ├────────┼────────┼─────────────┤
        # │ 100 %  │ ~1.00  │ 满载         │
        # └────────┴────────┴─────────────┘
        kv_pressure = 1.0 / (1.0 + math.exp(-6.0 * (kv_usage_perc - 0.5)))

        # 算力压力，KV 很空时，算力权重上升；KV 很满时，算力权重下降
        compute_weight = (1.0 - kv_usage_perc)  # 动态权重
        compute_pressure = min(running / self.max_num_seqs, 1.0) * compute_weight

        # 等待队列惩罚，等待越多，惩罚越大
        queue_penalty = min(waiting / self.max_num_seqs, 1.0)

        return min(kv_pressure * 0.6 + compute_pressure*0.1 + queue_penalty*0.3, 1.0)