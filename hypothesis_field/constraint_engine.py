"""
src/core/hypothesis_field/constraint_engine.py
ハード制約チェック + 射影（安全領域への押し戻し）
"""
from __future__ import annotations

from typing import Dict, List

from .models import Hypothesis, ConstraintResult, clamp

# ハード制約しきい値
HARD_RISK_MAX      = 0.40
HARD_FEASIBILITY_MIN = 0.60
HARD_CONSISTENCY_MIN = 0.70


class ConstraintEngine:

    def validate(self, h: Hypothesis) -> ConstraintResult:
        violations: List[str] = []
        v = h.vector

        if v.get("risk", 0.0) > HARD_RISK_MAX:
            violations.append(f"risk={v['risk']:.3f} > {HARD_RISK_MAX} (ハード制約違反)")
        if v.get("feasibility", 1.0) < HARD_FEASIBILITY_MIN:
            violations.append(f"feasibility={v['feasibility']:.3f} < {HARD_FEASIBILITY_MIN} (ハード制約違反)")
        if v.get("consistency", 1.0) < HARD_CONSISTENCY_MIN:
            violations.append(f"consistency={v['consistency']:.3f} < {HARD_CONSISTENCY_MIN} (ハード制約違反)")

        return ConstraintResult(passed=len(violations) == 0, violations=violations)

    def project(self, h: Hypothesis) -> Hypothesis:
        """制約違反ベクトルを安全領域に射影して返す（元仮説は変更しない）"""
        import copy
        projected = copy.deepcopy(h)
        v = projected.vector

        if v.get("risk", 0.0) > HARD_RISK_MAX:
            v["risk"] = HARD_RISK_MAX
        if v.get("feasibility", 1.0) < HARD_FEASIBILITY_MIN:
            v["feasibility"] = HARD_FEASIBILITY_MIN
        if v.get("consistency", 1.0) < HARD_CONSISTENCY_MIN:
            v["consistency"] = HARD_CONSISTENCY_MIN

        projected.history.append("constraint_projected")
        return projected

    def validate_and_project(self, h: Hypothesis) -> Hypothesis:
        result = self.validate(h)
        if not result.passed:
            return self.project(h)
        return h
