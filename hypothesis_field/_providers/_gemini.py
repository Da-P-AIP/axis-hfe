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


def _strip_json_fences(text: str) -> str:
    """Geminiが返すmarkdownコードフェンス（```json ... ```）を除去する。

    response_mime_type="application/json" を設定しても古いモデルや
    一部ケースでフェンス付き出力になる場合の保険として呼ぶ。
    """
    text = text.strip()
    # 開きフェンス (```json または ```) を除去
    if text.startswith("```"):
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1:]
    # 閉じフェンスを除去
    text = text.rstrip()
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()


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
        response_schema: Optional[Any] = None,  # JSON応答の構造を固定するスキーマ
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

        cfg_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "system_instruction": system,
            # JSON出力モードを強制: Geminiがmarkdownフェンスや説明文を付けずに
            # 純粋なJSONを返すようにする。これがmock-fallbackの根本修正。
            "response_mime_type": "application/json",
        }
        if response_schema is not None:
            # 応答の「形」まで固定することでパース失敗率をさらに下げる
            cfg_kwargs["response_schema"] = response_schema

        cfg = types.GenerateContentConfig(**cfg_kwargs)

        logger.debug(
            f"[Gemini] model={model} prompt_len={len(prompt)} "
            f"has_schema={response_schema is not None}"
        )

        resp = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=cfg,
        )

        usage = getattr(resp, "usage_metadata", None)
        # _strip_json_fences は response_mime_type が効かない場合の保険
        return LLMResponse(
            content=_strip_json_fences(resp.text),
            model=model,
            tokens_used=getattr(usage, "total_token_count", 0) or 0,
        )
