"""
_providers/_gemini.py
Google Gemini API クライアント（gemini-2.5-flash など）
依存: pip install axis-hfe[gemini]  ->  google-genai>=1.0
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from ._base import LLMResponse

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Google Gemini API への非同期クライアント。

    APIキーは引数で渡すか、環境変数 GEMINI_API_KEY または GOOGLE_API_KEY から自動取得。

    Usage:
        config = EngineConfig(
            provider="gemini",
            model="gemini-2.5-flash",
            api_key="AIza...",   # または環境変数 GEMINI_API_KEY
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        self._api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.timeout = timeout
        if not self._api_key:
            raise ValueError(
                "Gemini API キーが設定されていません。\n"
                "引数 api_key='AIza...' または環境変数 GEMINI_API_KEY を設定してください。"
            )

    async def generate(
        self,
        prompt: str,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: Optional[str] = None,  # プロンプトインジェクション対策: 指示を分離
        **_: Any,
    ) -> LLMResponse:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "Gemini プロバイダーを使うには以下を実行してください:\n"
                "  pip install axis-hfe[gemini]"
            )

        client = genai.Client(api_key=self._api_key)
        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system,
        )

        logger.debug(f"[Gemini] model={model} prompt_len={len(prompt)}")

        resp = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=cfg,
        )

        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            content=resp.text,
            model=model,
            tokens_used=getattr(usage, "total_token_count", 0) or 0,
        )
