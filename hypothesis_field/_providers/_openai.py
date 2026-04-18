"""
_providers/_openai.py
OpenAI API クライアント（ChatGPT / GPT-4o など）
依存: pip install axis-hfe[openai]  →  openai>=1.0
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from ._base import LLMResponse

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    OpenAI Chat Completions API への非同期クライアント。

    APIキーは引数で渡すか、環境変数 OPENAI_API_KEY から自動取得。

    Usage:
        config = EngineConfig(
            provider="openai",
            model="gpt-4o",
            api_key="sk-...",   # または環境変数 OPENAI_API_KEY
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        if not self._api_key:
            raise ValueError(
                "OpenAI API キーが設定されていません。\n"
                "引数 api_key='sk-...' または環境変数 OPENAI_API_KEY を設定してください。"
            )

    async def generate(
        self,
        prompt: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: Optional[str] = None,  # プロンプトインジェクション対策: 指示を分離
        **_: Any,
    ) -> LLMResponse:
        try:
            import openai
        except ImportError:
            raise ImportError(
                "OpenAI プロバイダーを使うには以下を実行してください:\n"
                "  pip install axis-hfe[openai]"
            )

        client = openai.AsyncOpenAI(
            api_key=self._api_key,
            timeout=self.timeout,
        )

        logger.debug(f"[OpenAI] model={model} prompt_len={len(prompt)}")

        messages: list = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
        )
