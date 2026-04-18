"""
src/core/hypothesis_field/evaluator.py
理想ベクトルへの距離スコア計算 + ペナルティ統合
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .models import (
    Hypothesis, FieldResult, DEFAULT_IDEAL, DEFAULT_WEIGHTS, VECTOR_KEYS, clamp
)


class Evaluator:
    """
    全仮説を総合スコアで評価し、ランキングを返す。
    スコア = 基本スコア(理想距離) + Relational評価 - ペナルティ
    """

    def __init__(
        self,
        ideal: Dict[str, float] | None = None,
        weights: Dict[str, float] | None = None,
    ):
        self.ideal = ideal or DEFAULT_IDEAL.copy()
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    def score(self, h: Hypothesis) -> float:
        v = h.vector

        # 基本スコア: 理想ベクトルへの加重距離の逆数
        distance = h.vector_distance(self.ideal, self.weights)
        base = 1.0 / (1.0 + distance)

        # ソフト制約ペナルティ
        penalty = 0.0
        if v.get("novelty", 0.5) < 0.50:
            penalty += (0.50 - v["novelty"]) * 0.5
        if v.get("divergence", 0.5) < 0.30:
            penalty += (0.30 - v["divergence"]) * 0.3
        if v.get("risk", 0.2) > 0.40:
            penalty += (v["risk"] - 0.40) * 2.0
        if v.get("feasibility", 0.9) < 0.60:
            penalty += (0.60 - v["feasibility"]) * 2.0

        # Relational Layer 統合
        rel_weight = self.weights.get("relational", 0.3)
        relational = rel_weight * h.relational.score

        return max(0.0, base + relational - penalty)

    def rank(self, hypotheses: List[Hypothesis]) -> List[Tuple[Hypothesis, float]]:
        scored = [(h, self.score(h)) for h in hypotheses]
        scored.sort(key=lambda x: x[1], reverse=True)
        for h, s in scored:
            h.score = s
        return scored

    def top_n(self, hypotheses: List[Hypothesis], n: int = 5) -> List[Hypothesis]:
        return [h for h, _ in self.rank(hypotheses)[:n]]
