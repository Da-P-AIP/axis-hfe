"""
src/core/hypothesis_field/relational_layer.py
相対尊厳評価（理解・整合・尊重・調和）
"""
from __future__ import annotations

from .models import Hypothesis, RelationalScores, clamp


class RelationalLayer:
    """
    仮説テキストと6次元ベクトルから関係者間適合スコアを推定する。
    LLMなしのルールベース実装（拡張可能）。
    """

    def evaluate(self, h: Hypothesis) -> RelationalScores:
        v = h.vector
        content = h.content.lower()

        # understanding: 情報の共有度・曖昧さの少なさ → consistency + accuracy
        understanding = clamp(
            0.5 * v.get("accuracy", 0.5) + 0.5 * v.get("consistency", 0.5)
        )

        # alignment: 目的一致度 → feasibility + (1 - divergence*0.3)
        alignment = clamp(
            0.6 * v.get("feasibility", 0.5) + 0.4 * (1.0 - 0.3 * v.get("divergence", 0.5))
        )

        # respect: 一方的搾取・破壊のなさ → 1 - risk
        respect = clamp(1.0 - v.get("risk", 0.2))

        # harmony: 長期安定性 → consistency + feasibility - risk*0.5
        harmony = clamp(
            0.4 * v.get("consistency", 0.5) +
            0.4 * v.get("feasibility", 0.5) -
            0.2 * v.get("risk", 0.2)
        )

        # テキストに協調・共存ワードがあればボーナス
        coop_words = ["共存", "協調", "共有", "調和", "連携", "統合", "バランス"]
        if any(w in content for w in coop_words):
            understanding = clamp(understanding + 0.05)
            alignment     = clamp(alignment + 0.05)
            harmony       = clamp(harmony + 0.05)

        return RelationalScores(
            understanding=understanding,
            alignment=alignment,
            respect=respect,
            harmony=harmony,
        )

    def apply(self, h: Hypothesis) -> Hypothesis:
        h.relational = self.evaluate(h)
        return h
