import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseLoadCollector(ABC):
    """负载采集器抽象基类"""

    @abstractmethod
    async def fetch_load(self, server_url: str) -> Optional[float]:
        """
        采集单个服务器的当前负载值。

        Args:
            server_url: 后端服务器的 base URL（例如 http://localhost:8100）

        Returns:
            归一化的负载值（0 ~ 1 之间，或自定义范围，但调度器会将其转换为整数比较），
            如果采集失败返回 None
        """
        pass

    @abstractmethod
    async def health_check(self, server_url: str) -> bool:
        """检查服务器是否健康"""
        pass


class LoadUpdateConfig:
    """采集器配置"""
    def __init__(self, interval_seconds: float = 1.0,
                 fallback_load: float = 1.0,
                 scale_factor: int = 1000):
        self.interval_seconds = interval_seconds
        self.fallback_load = fallback_load      # 采集失败时使用的负载值
        self.scale_factor = scale_factor