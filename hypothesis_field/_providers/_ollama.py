"""
_providers/_ollama.py
Ollama ローカルサーバー向けクライアント
依存: httpx のみ（追加インストール不要）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from ._base import LLMResponse
from .._security import validate_ollama_url, mask_secrets

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Ollama /api/chat エンドポイントへの非同期クライアント。

    Usage:
        config = EngineConfig(provider="ollama", model="gemma4:e4b")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        api_key: Optional[str] = None,  # Ollama は不要（互換引数）
    ):
        self.base_url = validate_ollama_url(base_url)  # SSRF対策: スキーム・ホスト検証
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=float(self.timeout),
                    write=30.0,
                    pool=5.0,
                )
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        model: str = "gemma4:e4b",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: Optional[str] = None,  # プロンプトインジェクション対策: 指示を分離
        **_: Any,
    ) -> LLMResponse:
        client = await self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

        logger.debug(f"[Ollama] model={model} prompt_len={len(prompt)}")
        response = await client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            model=model,
            tokens_used=data.get("eval_count", 0),
        )

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            r = await client.get(f"{self.base_url}/api/tags")
            return r.status_code == 200
        except Exception as e:
            logger.warning(f"[Ollama] health check failed: {type(e).__name__}")
            return False

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
