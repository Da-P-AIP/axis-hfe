"""
hypothesis_field — Hypothesis Field Engine (HFE)
axis-hfe スタンドアロンパッケージ

Designed by Da-P (Da-P-AIP)
Part of the Cogito Project
"""
import logging

# ライブラリの作法: パッケージ内でハンドラを設定しない。
# 利用者がルートロガーを設定していない環境でも警告が出ないよう NullHandler を登録する。
logging.getLogger(__name__).addHandler(logging.NullHandler())

from .models import (
    Hypothesis,
    ConstraintResult,
    FieldResult,
    RelationalScores,
    VECTOR_KEYS,
    DEFAULT_IDEAL,
    DEFAULT_WEIGHTS,
    IDEAL_PRESETS,
)
from .field_engine import HypothesisFieldEngine, EngineConfig, build_engine
from ._providers import DEFAULT_MODELS, SUPPORTED_PROVIDERS, create_client
from ._providers._ollama import OllamaClient
from ._providers._openai import OpenAIClient
from ._providers._anthropic import AnthropicClient

__version__ = "0.2.0"
__author__ = "Da-P (Da-P-AIP)"

__all__ = [
    # メインAPI
    "HypothesisFieldEngine",
    "EngineConfig",
    "build_engine",
    # プロバイダー
    "OllamaClient",
    "OpenAIClient",
    "AnthropicClient",
    "create_client",
    "DEFAULT_MODELS",
    "SUPPORTED_PROVIDERS",
    # データモデル
    "Hypothesis",
    "ConstraintResult",
    "FieldResult",
    "RelationalScores",
    "VECTOR_KEYS",
    "DEFAULT_IDEAL",
    "DEFAULT_WEIGHTS",
    "IDEAL_PRESETS",
]
