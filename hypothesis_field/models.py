"""
src/core/hypothesis_field/models.py
Hypothesis Field Engine — データモデル定義
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 6次元評価ベクトルのキー
VECTOR_KEYS = ["accuracy", "consistency", "risk", "novelty", "feasibility", "divergence"]

# デフォルト理想ベクトル
DEFAULT_IDEAL: Dict[str, float] = {
    "accuracy":    0.95,
    "consistency": 0.90,
    "risk":        0.20,
    "novelty":     0.60,
    "feasibility": 0.90,
    "divergence":  0.45,
}

# デフォルト重みベクトル
DEFAULT_WEIGHTS: Dict[str, float] = {
    "accuracy":    1.0,
    "consistency": 0.8,
    "risk":        1.2,
    "novelty":     0.7,
    "feasibility": 1.0,
    "divergence":  0.6,
    "relational":  0.3,
}

# 理想ベクトルプリセット
IDEAL_PRESETS: Dict[str, Dict[str, float]] = {
    "default":    DEFAULT_IDEAL,
    "creative":   {"accuracy": 0.70, "consistency": 0.70, "risk": 0.35, "novelty": 0.90, "feasibility": 0.70, "divergence": 0.85},
    "safe":       {"accuracy": 0.95, "consistency": 0.95, "risk": 0.05, "novelty": 0.30, "feasibility": 0.95, "divergence": 0.20},
    "balanced":   {"accuracy": 0.80, "consistency": 0.80, "risk": 0.20, "novelty": 0.70, "feasibility": 0.80, "divergence": 0.60},
}


def zero_vector() -> Dict[str, float]:
    return {k: 0.0 for k in VECTOR_KEYS}


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class RelationalScores:
    understanding: float = 0.5
    alignment:     float = 0.5
    respect:       float = 0.5
    harmony:       float = 0.5

    @property
    def score(self) -> float:
        return 0.25 * (self.understanding + self.alignment + self.respect + self.harmony)


@dataclass
class Hypothesis:
    id:        str
    content:   str
    vector:    Dict[str, float] = field(default_factory=zero_vector)
    score:     float = 0.0
    source:    str = "generated"      # generated / linear_fusion / nonlinear_fusion / jump / self_corrected
    history:   List[str] = field(default_factory=list)
    relational: RelationalScores = field(default_factory=RelationalScores)

    def vector_distance(self, ideal: Dict[str, float], weights: Dict[str, float]) -> float:
        return math.sqrt(sum(
            weights.get(k, 1.0) * (self.vector.get(k, 0.0) - ideal[k]) ** 2
            for k in VECTOR_KEYS
        ))


@dataclass
class ConstraintResult:
    passed:     bool
    violations: List[str] = field(default_factory=list)
    projected:  Optional[Dict[str, float]] = None


@dataclass
class FieldResult:
    best:    Hypothesis
    ranked:  List[Hypothesis]
    iterations_run: int
    problem: str
