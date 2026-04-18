"""
_ollama_client.py
後方互換: _providers._ollama への再エクスポート
"""
from ._providers._ollama import OllamaClient
from ._providers._base import LLMResponse

__all__ = ["OllamaClient", "LLMResponse"]
