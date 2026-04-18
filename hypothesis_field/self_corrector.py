"""
src/core/hypothesis_field/self_corrector.py
理想ベクトルへの段階的自己補正 + 制約再射影
"""
from __future__ import annotations

import copy
from typing import Dict

from .models import Hypothesis, DEFAULT_IDEAL, VECTOR_KEYS, clamp
from .constraint_engine import ConstraintEngine


class SelfCorrector:

    def __init__(
        self,
        ideal: Dict[str, float] | None = None,
        step: float = 0.25,
    ):
        self.ideal = ideal or DEFAULT_IDEAL.copy()
        self.step = step
        self._constraint = ConstraintEngine()

    def revise(self, h: Hypothesis) -> Hypothesis:
        """理想ベクトルに step 分近づけ、制約違反があれば射影する"""
        revised = copy.deepcopy(h)
        for k in VECTOR_KEYS:
            current = revised.vector.get(k, 0.5)
            target = self.ideal[k]
            revised.vector[k] = clamp(current + self.step * (target - current))

        revised = self._constraint.validate_and_project(revised)
        revised.id = f"{h.id}_corrected"
        revised.source = "self_corrected"
        revised.history = list(h.history) + [f"self_correct(step={self.step})"]
        revised.content = f"[自己修正: {h.id}] {h.content}"
        return revised
