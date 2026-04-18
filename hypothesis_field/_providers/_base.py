"""
_providers/_base.py
全プロバイダー共通の LLMResponse データクラス
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int = 0
