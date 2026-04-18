"""
_providers/_anthropic.py
Anthropic API クライアント（Claude Sonnet / Opus / Haiku など）
依存: pip install axis-hfe[anthropic]  →  anthropic>=0.20
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from ._base import LLMResponse

logger = logging.getLogger(__name__)


class AnthropicClient:
    """
    Anthropic Messages API への非同期クライアント。

    APIキーは引数で渡すか、環境変数 ANTHROPIC_API_KEY から自動取得。

    Usage:
        config = EngineConfig(
            provider="anthropic",
            model="claude-sonnet-4-6",
            api_key="sk-ant-...",   # または環境変数 ANTHROPIC_API_KEY
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.timeout = timeout
        if not self._api_key:
            raise ValueError(
                "Anthropic API キーが設定されていません。\n"
                "引数 api_key='sk-ant-...' または環境変数 ANTHROPIC_API_KEY を設定してください。"
            )

    async def generate(
        self,
        prompt: str,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: Optional[str] = None,  # プロンプトインジェクション対策: 指示を分離
        **_: Any,
    ) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic プロバイダーを使うには以下を実行してください:\n"
                "  pip install axis-hfe[anthropic]"
            )

        client = anthropic.AsyncAnthropic(
            api_key=self._api_key,
            timeout=self.timeout,
        )

        logger.debug(f"[Anthropic] model={model} prompt_len={len(prompt)}")

        kwargs: dict = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)

        return LLMResponse(
            content=response.content[0].text,
            model=model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens,
        )
