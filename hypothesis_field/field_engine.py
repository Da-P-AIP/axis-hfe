"""
hypothesis_field/field_engine.py
HypothesisFieldEngine — 統合エントリーポイント（axis-hfe スタンドアロン版）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .models import (
    Hypothesis, FieldResult, DEFAULT_IDEAL, DEFAULT_WEIGHTS, IDEAL_PRESETS
)
from .generator import HypothesisGenerator
from .vectorizer import Vectorizer
from .evaluator import Evaluator
from .constraint_engine import ConstraintEngine
from .fusion_engine import FusionEngine
from .jump_explorer import JumpExplorer
from .self_corrector import SelfCorrector
from .relational_layer import RelationalLayer
from ._providers import DEFAULT_MODELS, SUPPORTED_PROVIDERS, create_client
from ._security import validate_engine_params, sanitize_input, MAX_PROBLEM_LENGTH

logger = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    """
    HypothesisFieldEngine の設定クラス。

    対応プロバイダー:
        - "ollama"    : ローカル Ollama（追加インストール不要）
        - "openai"    : ChatGPT / GPT-4o  （pip install axis-hfe[openai]）
        - "anthropic" : Claude             （pip install axis-hfe[anthropic]）

    Examples:
        # Ollama（デフォルト）
        config = EngineConfig(provider="ollama", model="gemma4:e4b")

        # OpenAI
        config = EngineConfig(provider="openai", model="gpt-4o", api_key="sk-...")

        # Anthropic（環境変数 ANTHROPIC_API_KEY から自動取得）
        config = EngineConfig(provider="anthropic", model="claude-sonnet-4-6")

        # テスト用（LLMなし）
        config = EngineConfig(mock_llm=True)
    """
    provider: str = "ollama"           # "ollama" / "openai" / "anthropic"
    model: Optional[str] = None        # None → プロバイダーのデフォルトモデルを使用
    api_key: Optional[str] = None      # OpenAI / Anthropic の API キー（None → 環境変数）
    ollama_base_url: str = "http://localhost:11434"
    ideal_preset: str = "default"      # default / creative / safe / balanced
    iterations: int = 3
    hypothesis_count: int = 3
    mock_llm: bool = False             # True にするとフォールバック仮説で即返却（テスト用）
    custom_ideal: Optional[Dict[str, float]] = field(default=None)
    custom_weights: Optional[Dict[str, float]] = field(default=None)

    def __post_init__(self) -> None:
        """DoS対策: パラメータ範囲を早期検証する。"""
        validate_engine_params(self.iterations, self.hypothesis_count)

    def __repr__(self) -> str:
        """api_key をマスクして repr に含める。ログ・例外トレースへの漏洩を防ぐ。"""
        masked = "***" if self.api_key else None
        return (
            f"EngineConfig(provider={self.provider!r}, model={self.model!r}, "
            f"api_key={masked!r}, ideal_preset={self.ideal_preset!r}, "
            f"iterations={self.iterations}, hypothesis_count={self.hypothesis_count}, "
            f"mock_llm={self.mock_llm})"
        )

    def resolved_model(self) -> str:
        """model が None の場合、プロバイダーのデフォルトモデルを返す。"""
        return self.model or DEFAULT_MODELS.get(self.provider, "gemma4:e4b")


class HypothesisFieldEngine:
    """
    多次元仮説空間推論エンジン。

    pip install axis-hfe 後、以下のように使う::

        from hypothesis_field import HypothesisFieldEngine, EngineConfig
        import asyncio

        # Ollama
        config = EngineConfig(provider="ollama", model="gemma4:e4b")

        # OpenAI
        config = EngineConfig(provider="openai", model="gpt-4o", api_key="sk-...")

        # Anthropic
        config = EngineConfig(provider="anthropic", model="claude-sonnet-4-6")

        engine = HypothesisFieldEngine(config)
        result = asyncio.run(engine.run("月5万円稼ぐビジネスモデルを考えて"))

        print(result.best.content)   # 最良仮説
        print(result.best.score)     # スコア
        print(result.best.vector)    # 6次元ベクトル
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        # 後方互換: llm_client を直接渡す場合（CogitoV10スタイル）
        llm_client=None,
        model: str = "gemma4:e4b",
        ideal: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        if config is not None:
            self._config = config
            ideal = config.custom_ideal or IDEAL_PRESETS.get(config.ideal_preset, DEFAULT_IDEAL)
            weights = config.custom_weights or DEFAULT_WEIGHTS.copy()

            if config.mock_llm:
                llm_client = None
            else:
                if config.provider not in SUPPORTED_PROVIDERS:
                    raise ValueError(
                        f"未知のプロバイダー: '{config.provider}'\n"
                        f"使用可能: {SUPPORTED_PROVIDERS}"
                    )
                llm_client = create_client(
                    provider=config.provider,
                    api_key=config.api_key,
                    ollama_base_url=config.ollama_base_url,
                )
            model = config.resolved_model()
        else:
            self._config = EngineConfig(model=model, mock_llm=(llm_client is None))

        self.ideal   = ideal or DEFAULT_IDEAL.copy()
        self.weights = weights or DEFAULT_WEIGHTS.copy()

        self._generator  = HypothesisGenerator(llm_client, model)
        self._vectorizer = Vectorizer(llm_client, model)
        self._evaluator  = Evaluator(self.ideal, self.weights)
        self._constraint = ConstraintEngine()
        self._self_corr  = SelfCorrector(self.ideal)
        self._relational = RelationalLayer()

    async def run(
        self,
        problem: Union[str, List[Hypothesis]],
        iterations: Optional[int] = None,
        count: int = 3,
    ) -> FieldResult:
        """
        問題文字列または仮説リストを受け取り、進化ループを回して最良仮説を返す。

        Args:
            problem:    問題テキスト（str）または仮説リスト（List[Hypothesis]）
            iterations: ループ回数（None なら EngineConfig の値を使う）
            count:      初期仮説生成数（problem が str の場合のみ使用）
        """
        iters = iterations if iterations is not None else self._config.iterations

        if isinstance(problem, str):
            problem = sanitize_input(problem, max_length=MAX_PROBLEM_LENGTH)
            hypotheses = await self._generator.generate(problem, count=count)
        else:
            hypotheses = problem

        return await self._run_loop(hypotheses, iters)

    async def _run_loop(self, hypotheses: List[Hypothesis], iterations: int) -> FieldResult:
        problem = hypotheses[0].content if hypotheses else ""
        current = hypotheses

        for i in range(iterations):
            current = await self._vectorizer.vectorize(current)
            current = [self._relational.apply(h) for h in current]
            current = [self._constraint.validate_and_project(h) for h in current]

            ranked = self._evaluator.rank(current)
            top3   = [h for h, _ in ranked[:3]]
            generated: List[Hypothesis] = []

            if len(top3) >= 2:
                generated.append(FusionEngine.linear_fusion(top3[0], top3[1], alpha=0.6))
                generated.append(FusionEngine.nonlinear_fusion(top3[0], top3[1], alpha=0.55, gamma=0.25))

            if len(ranked) >= 2:
                best_h, worst_h = ranked[0][0], ranked[-1][0]
                generated.append(JumpExplorer.opposite_jump(best_h, worst_h, factor=0.35))

            for h in top3:
                generated.append(JumpExplorer.perturb(h, scale=0.10))
                generated.append(self._self_corr.revise(h))

            if generated:
                generated = await self._vectorizer.vectorize(generated)
                generated = [self._relational.apply(h) for h in generated]
                generated = [self._constraint.validate_and_project(h) for h in generated]

            pool = current + generated
            current = self._evaluator.top_n(pool, n=5)
            logger.debug(f"[HFE] iter={i+1} pool={len(pool)} -> survived={len(current)}")

        final_ranked = self._evaluator.rank(current)
        best = final_ranked[0][0]
        return FieldResult(
            best=best,
            ranked=[h for h, _ in final_ranked],
            iterations_run=iterations,
            problem=problem,
        )


def build_engine(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    ollama_base_url: str = "http://localhost:11434",
    ideal_preset: str = "default",
    iterations: int = 3,
    mock_llm: bool = False,
) -> HypothesisFieldEngine:
    """EngineConfig を経由せずに素早くエンジンを作る便利関数。"""
    config = EngineConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        ollama_base_url=ollama_base_url,
        ideal_preset=ideal_preset,
        iterations=iterations,
        mock_llm=mock_llm,
    )
    return HypothesisFieldEngine(config)
