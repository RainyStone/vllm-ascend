from typing import Optional
import httpx
from .base import BaseLoadCollector
from .vllm_collector import VLLMMetricsCollector
from .metric_load_calculator import BaseLoadCalculator, DefaultLoadCalculator

def create_load_collector(backend_type: str,
                          client: httpx.AsyncClient,
                          calculator: Optional[BaseLoadCalculator] = None) -> BaseLoadCollector:
    """根据后端类型创建采集器实例"""
    if backend_type == "vllm":
        return VLLMMetricsCollector(client, calculator or DefaultLoadCalculator())
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")