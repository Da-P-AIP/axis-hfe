"""
_providers/__init__.py
LLM プロバイダーファクトリー
"""
from __future__ import annotations

from typing import Any, Optional

from ._base import LLMResponse

# プロバイダーごとのデフォルトモデル
DEFAULT_MODELS: dict[str, str] = {
    "ollama":    "gemma4:e4b",
    "openai":    "gpt-4o",
    "anthropic": "claude-sonnet-4-6",
}

SUPPORTED_PROVIDERS = list(DEFAULT_MODELS.keys())


def create_client(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    ollama_base_url: str = "http://localhost:11434",
    timeout: int = 300,
) -> Any:
    """
    プロバイダー名からクライアントを生成するファクトリー関数。

    Args:
        provider: "ollama" / "openai" / "anthropic"
        api_key:  OpenAI / Anthropic の API キー（Noneなら環境変数から取得）
        ollama_base_url: Ollama サーバー URL（provider="ollama" のみ使用）
        timeout:  タイムアウト秒数

    Returns:
        対応するクライアントオブジェクト

    Raises:
        ValueError: 未知のプロバイダー名の場合
    """
    if provider == "ollama":
        from ._ollama import OllamaClient
        return OllamaClient(base_url=ollama_base_url, timeout=timeout)
    elif provider == "openai":
        from ._openai import OpenAIClient
        return OpenAIClient(api_key=api_key, timeout=timeout)
    elif provider == "anthropic":
        from ._anthropic import AnthropicClient
        return AnthropicClient(api_key=api_key, timeout=timeout)
    else:
        raise ValueError(
            f"未知のプロバイダー: '{provider}'\n"
            f"使用可能: {SUPPORTED_PROVIDERS}"
        )


__all__ = [
    "LLMResponse",
    "DEFAULT_MODELS",
    "SUPPORTED_PROVIDERS",
    "create_client",
]
