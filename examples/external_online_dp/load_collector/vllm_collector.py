import re
from collections import defaultdict
from typing import Optional, Dict

import httpx
from .base import BaseLoadCollector
from .metric_load_calculator import BaseLoadCalculator, DefaultLoadCalculator

class VLLMMetricsCollector(BaseLoadCollector):
    """基于 vLLM /metrics 端点的负载采集器"""

    def __init__(self, client: httpx.AsyncClient, load_calculator: Optional[BaseLoadCalculator] = None):
        self.client = client
        self.load_calculator = load_calculator or DefaultLoadCalculator()

    # 新增：获取原始 metrics 字典
    async def fetch_metrics(self, server_url: str) -> Optional[Dict[str, float]]:
        try:
            resp = await self.client.get(f"{server_url}/metrics")
            resp.raise_for_status()
            return self._parse_metrics(resp.text)
        except httpx.HTTPStatusError as e:
            print(f"Metrics fetch HTTP error from {server_url}: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            print(f"Metrics fetch connection error from {server_url}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error fetching metrics from {server_url}: {e}")
            return None

     # 修改：通过 fetch_metrics 计算 load # TODO 代码架构需要优化，现在外面是使用 fetch_metrics 再计算 load，而不是直接使用 fetch_load
    async def fetch_load(self, server_url: str) -> Optional[float]:
        metrics = await self.fetch_metrics(server_url)
        if metrics is None:
            return None
        return self.load_calculator.calculate(metrics)

    async def health_check(self, server_url: str) -> bool:
        try:
            resp = await self.client.get(f"{server_url}/health")
            return resp.status_code == 200
        except Exception:
            return False

    def _parse_metrics(self, text: str) -> dict:
        """解析 Prometheus 格式的 metrics，返回关键指标的字典"""
        accumulators = defaultdict(list)
        patterns = {
            "kv_cache_usage_perc": r'^vllm:kv_cache_usage_perc(?:\{[^}]*\})?\s+([0-9.eE+-]+)',
            "num_requests_running": r'^vllm:num_requests_running(?:\{[^}]*\})?\s+([0-9.eE+-]+)',
            "num_requests_waiting": r'^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.eE+-]+)',
        }
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    try:
                        accumulators[key].append(float(match.group(1)))
                    except ValueError:
                        pass
                    break
        # TODO 取均值还是取总值
        result = {k: sum(v) / len(v) for k, v in accumulators.items()}
        return result