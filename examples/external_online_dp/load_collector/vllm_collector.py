import re
from typing import Optional

import httpx
from base import BaseLoadCollector
from metric_load_calculator import BaseLoadCalculator, DefaultLoadCalculator

class VLLMMetricsCollector(BaseLoadCollector):
    """基于 vLLM /metrics 端点的负载采集器"""

    def __init__(self, client: httpx.AsyncClient, load_calculator: Optional[BaseLoadCalculator] = None):
        self.client = client
        self.load_calculator = load_calculator or DefaultLoadCalculator()

    async def fetch_load(self, server_url: str) -> Optional[float]:
        """获取负载值（0~1 之间）"""
        try:
            resp = await self.client.get(f"{server_url}/metrics")
            resp.raise_for_status()
            metrics_text = resp.text
            metrics = self._parse_metrics(metrics_text)
            load = self.load_calculator.calculate(metrics)
            return load
        except Exception as e:
            # 日志记录
            return None

    async def health_check(self, server_url: str) -> bool:
        try:
            resp = await self.client.get(f"{server_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    def _parse_metrics(self, text: str) -> dict:
        """解析 Prometheus 格式的 metrics，返回关键指标的字典"""
        result = {}
        # 指标名称与值的映射
        patterns = {
            "kv_cache_usage_perc": r'vllm:kv_cache_usage_perc\s+([0-9.]+)',
            "num_requests_running": r'vllm:num_requests_running\s+([0-9.]+)',
            "num_requests_waiting": r'vllm:num_requests_waiting\s+([0-9.]+)',
            # 可以继续添加其他需要的指标
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                result[key] = float(match.group(1))
        return result