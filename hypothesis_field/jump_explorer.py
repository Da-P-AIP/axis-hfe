"""
src/core/hypothesis_field/jump_explorer.py
ランダム跳躍・逆方向ブーストによる局所最適脱出
"""
from __future__ import annotations

import random
from typing import Dict

from .models import Hypothesis, VECTOR_KEYS, clamp


class JumpExplorer:

    @staticmethod
    def perturb(h: Hypothesis, scale: float = 0.12) -> Hypothesis:
        """各軸にランダムな摂動を加える（ランダム跳躍）"""
        import copy
        jumped = copy.deepcopy(h)
        for k in VECTOR_KEYS:
            delta = random.uniform(-scale, scale)
            jumped.vector[k] = clamp(jumped.vector.get(k, 0.5) + delta)
        jumped.id = f"{h.id}_perturb"
        jumped.source = "jump"
        jumped.history = list(h.history) + [f"perturb(scale={scale})"]
        jumped.content = f"[ランダム探索: {h.id}] {h.content}"
        return jumped

    @staticmethod
    def opposite_jump(best: Hypothesis, worst: Hypothesis, factor: float = 0.35) -> Hypothesis:
        """最悪仮説から最良仮説を逆方向に伸ばす（逆方向ブースト）"""
        import copy
        jumped = copy.deepcopy(best)
        for k in VECTOR_KEYS:
            b = best.vector.get(k, 0.5)
            w = worst.vector.get(k, 0.5)
            jumped.vector[k] = clamp(b + factor * (b - w))
        jumped.id = f"{best.id}_opposite"
        jumped.source = "jump"
        jumped.history = list(best.history) + [f"opposite_jump(worst={worst.id},factor={factor})"]
        jumped.content = f"[逆方向跳躍: {best.id}↑↑] {best.content}"
        return jumped
