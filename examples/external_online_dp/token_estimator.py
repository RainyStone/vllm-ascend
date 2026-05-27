from abc import ABC, abstractmethod
from typing import Dict, Type


class TokenEstimator(ABC):
    """请求 token 估算策略抽象基类"""

    @abstractmethod
    def estimate(self, req_data: dict) -> float:
        """
        从请求数据中提取输入文本并估算 token 数。

        Args:
            req_data: OpenAI API 格式的请求体（completions 或 chat.completions）

        Returns:
            估算的输入 token 数量
        """
        pass

    @staticmethod
    def extract_text(req_data: dict) -> str:
        """从请求数据中提取所有输入文本内容"""
        texts = []

        # /v1/completions
        prompt = req_data.get("prompt")
        if prompt is not None:
            if isinstance(prompt, str):
                texts.append(prompt)
            elif isinstance(prompt, list):
                texts.extend(p for p in prompt if isinstance(p, str))

        # /v1/chat/completions
        elif "messages" in req_data:
            for msg in req_data.get("messages", []):
                content = msg.get("content", "")
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):  # 多模态/vision 格式
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part.get("text", ""))

        return "".join(texts)


class CharBasedEstimator(TokenEstimator):
    """
    基于字符数的经验估算。
    混合中英文平均 1 token ≈ 3 chars（英文约 4，中文约 1~1.5）。
    """

    def __init__(self, chars_per_token: float = 3.0):
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        self.chars_per_token = chars_per_token

    def estimate(self, req_data: dict) -> float:
        text = self.extract_text(req_data)
        return len(text) / self.chars_per_token


class ByteBasedEstimator(TokenEstimator):
    """兼容旧逻辑：基于 JSON 请求体的字节长度估算。"""

    def __init__(self, bytes_per_token: float = 4.0):
        if bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive")
        self.bytes_per_token = bytes_per_token

    def estimate(self, req_data: dict) -> float:
        import json
        try:
            body_bytes = json.dumps(req_data).encode("utf-8")
            return len(body_bytes) / self.bytes_per_token
        except (TypeError, ValueError):
            return 0.0


class TiktokenEstimator(TokenEstimator):
    """使用 tiktoken 进行精确的 token 计算（如果可用）。"""

    def __init__(self, model: str = "cl100k_base"):
        try:
            import tiktoken
            self.encoder = tiktoken.get_encoding(model)
        except ImportError as e:
            raise ImportError(
                "tiktoken is required for TiktokenEstimator. "
                "Install it with: pip install tiktoken"
            ) from e

    def estimate(self, req_data: dict) -> float:
        text = self.extract_text(req_data)
        if not text:
            return 0.0
        return float(len(self.encoder.encode(text)))


# 工厂方法：通过名称创建估算器
_ESTIMATOR_REGISTRY: Dict[str, Type[TokenEstimator]] = {
    "char": CharBasedEstimator,
    "byte": ByteBasedEstimator,
    "tiktoken": TiktokenEstimator,
}


def create_token_estimator(name: str, **kwargs) -> TokenEstimator:
    """
    通过名称创建 token 估算器实例。

    Args:
        name: 估算器名称，可选 "char", "byte", "tiktoken"
        **kwargs: 传递给估算器构造函数的参数

    Returns:
        TokenEstimator 实例

    Raises:
        ValueError: 如果名称未注册
    """
    estimator_cls = _ESTIMATOR_REGISTRY.get(name.lower())
    if estimator_cls is None:
        available = ", ".join(_ESTIMATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown token estimator '{name}'. "
            f"Available: {available}"
        )
    return estimator_cls(**kwargs)


def register_token_estimator(name: str, cls: Type[TokenEstimator]):
    """
    注册自定义的 token 估算器。

    示例：
        @register_token_estimator("my_custom")
        class MyEstimator(TokenEstimator):
            def estimate(self, req_data: dict) -> float:
                return 42.0
    """
    if not issubclass(cls, TokenEstimator):
        raise TypeError(f"{cls.__name__} must inherit from TokenEstimator")
    _ESTIMATOR_REGISTRY[name.lower()] = cls
    return cls