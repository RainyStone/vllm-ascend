from abc import ABC, abstractmethod
from typing import Dict

class BaseLoadCalculator(ABC):
    """负载计算策略抽象基类"""

    @abstractmethod
    def calculate(self, metrics: Dict[str, float]) -> float:
        """
        将采集的指标转换为一个归一化的负载分数（0~1）。
        数值越大表示负载越重。
        """
        pass


class DefaultLoadCalculator(BaseLoadCalculator):
    """默认负载计算策略：基于 KV cache 使用率 + 运行请求数 + 等待队列"""

    def __init__(self, kv_weight=0.6, running_weight=0.3, waiting_penalty=0.1, max_concurrent=256):
        self.kv_weight = kv_weight
        self.running_weight = running_weight
        self.waiting_penalty = waiting_penalty
        self.max_concurrent = max_concurrent

    def calculate(self, metrics: Dict[str, float]) -> float:
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

    def calculate(self, metrics: Dict[str, float]) -> float:
        running = metrics.get("num_requests_running", 0.0)
        return min(running / self.max_concurrent, 1.0)