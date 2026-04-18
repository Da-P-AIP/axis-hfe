"""
src/core/hypothesis_field/fusion_engine.py
線形・非線形ベクトル合成で新仮説を生成
"""
from __future__ import annotations

import math
import uuid
from typing import Dict

from .models import Hypothesis, VECTOR_KEYS, clamp


class FusionEngine:

    @staticmethod
    def linear_fusion(h1: Hypothesis, h2: Hypothesis, alpha: float = 0.6) -> Hypothesis:
        """加重平均による線形合成"""
        new_vector: Dict[str, float] = {}
        for k in VECTOR_KEYS:
            new_vector[k] = clamp(alpha * h1.vector.get(k, 0.5) + (1 - alpha) * h2.vector.get(k, 0.5))

        content = f"[線形合成: {h1.id}*{alpha:.1f} + {h2.id}*{1-alpha:.1f}] {h1.content[:60]} / {h2.content[:60]}"
        return Hypothesis(
            id=f"{h1.id}_{h2.id}_lin",
            content=content,
            vector=new_vector,
            source="linear_fusion",
            history=[f"linear_fusion({h1.id},{h2.id},alpha={alpha})"],
        )

    @staticmethod
    def nonlinear_fusion(h1: Hypothesis, h2: Hypothesis, alpha: float = 0.55, gamma: float = 0.25) -> Hypothesis:
        """軸ごとに異なるルールを適用する非線形合成（核心）"""
        new_vector: Dict[str, float] = {}
        for k in VECTOR_KEYS:
            a = h1.vector.get(k, 0.5)
            b = h2.vector.get(k, 0.5)
            mixed = alpha * a + (1 - alpha) * b
            interaction = gamma * (a * b)

            if k == "risk":
                value = min(a, b) + 0.1 * interaction          # 低い方を優先
            elif k == "novelty":
                value = math.tanh(mixed + interaction)          # 増幅
            elif k == "accuracy":
                value = max(a, b) * 0.7 + mixed * 0.3          # 高い方寄り
            else:
                value = mixed + interaction

            new_vector[k] = clamp(value)

        content = f"[非線形合成: {h1.id}x{h2.id}] {h1.content[:60]} + {h2.content[:60]}"
        return Hypothesis(
            id=f"{h1.id}_{h2.id}_nonlin",
            content=content,
            vector=new_vector,
            source="nonlinear_fusion",
            history=[f"nonlinear_fusion({h1.id},{h2.id},alpha={alpha},gamma={gamma})"],
        )
