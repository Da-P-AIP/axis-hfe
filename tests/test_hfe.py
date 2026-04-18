"""
tests/test_hfe.py
Axis-HFE テストスイート（モックモードで実行）

実行:
    cd axis-hfe
    python -m pytest tests/ -v
"""
import math
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis_field.models import (
    Hypothesis, FieldResult, RelationalScores,
    VECTOR_KEYS, DEFAULT_IDEAL, DEFAULT_WEIGHTS, IDEAL_PRESETS,
    zero_vector, clamp,
)
from hypothesis_field.evaluator import Evaluator
from hypothesis_field.constraint_engine import ConstraintEngine, HARD_RISK_MAX, HARD_FEASIBILITY_MIN, HARD_CONSISTENCY_MIN
from hypothesis_field.fusion_engine import FusionEngine
from hypothesis_field.jump_explorer import JumpExplorer
from hypothesis_field.self_corrector import SelfCorrector
from hypothesis_field.relational_layer import RelationalLayer
from hypothesis_field.generator import HypothesisGenerator
from hypothesis_field.vectorizer import Vectorizer
from hypothesis_field.field_engine import HypothesisFieldEngine, EngineConfig, build_engine


# ─── helpers ───────────────────────────────────────────────────────────────

def make_hypothesis(id: str = "H1", content: str = "テスト仮説", **vector_overrides) -> Hypothesis:
    v = {k: 0.7 for k in VECTOR_KEYS}
    v.update(vector_overrides)
    return Hypothesis(id=id, content=content, vector=v)


# ─── 1. models ──────────────────────────────────────────────────────────────

def test_clamp_within_range():
    assert clamp(0.5) == 0.5
    assert clamp(-0.1) == 0.0
    assert clamp(1.5) == 1.0


def test_zero_vector_has_all_keys():
    v = zero_vector()
    assert set(v.keys()) == set(VECTOR_KEYS)
    assert all(val == 0.0 for val in v.values())


def test_hypothesis_vector_distance():
    h = make_hypothesis(accuracy=0.9, consistency=0.9, risk=0.2, novelty=0.6, feasibility=0.9, divergence=0.45)
    dist = h.vector_distance(DEFAULT_IDEAL, DEFAULT_WEIGHTS)
    assert dist >= 0.0
    assert dist < 0.1  # 理想に近いはず


def test_relational_scores_average():
    rs = RelationalScores(understanding=0.8, alignment=0.6, respect=0.7, harmony=0.5)
    expected = 0.25 * (0.8 + 0.6 + 0.7 + 0.5)
    assert abs(rs.score - expected) < 1e-9


def test_ideal_presets_all_keys():
    for name, preset in IDEAL_PRESETS.items():
        assert set(preset.keys()) == set(VECTOR_KEYS), f"preset '{name}' has missing keys"


# ─── 2. constraint_engine ───────────────────────────────────────────────────

def test_constraint_passes_valid_hypothesis():
    h = make_hypothesis(risk=0.2, feasibility=0.8, consistency=0.8)
    result = ConstraintEngine().validate(h)
    assert result.passed is True
    assert result.violations == []


def test_constraint_catches_high_risk():
    h = make_hypothesis(risk=0.9)
    result = ConstraintEngine().validate(h)
    assert result.passed is False
    assert any("risk" in v for v in result.violations)


def test_constraint_project_clamps_risk():
    h = make_hypothesis(risk=0.99, feasibility=0.3, consistency=0.5)
    projected = ConstraintEngine().project(h)
    assert projected.vector["risk"] <= HARD_RISK_MAX
    assert projected.vector["feasibility"] >= HARD_FEASIBILITY_MIN
    assert projected.vector["consistency"] >= HARD_CONSISTENCY_MIN
    assert "constraint_projected" in projected.history


# ─── 3. evaluator ───────────────────────────────────────────────────────────

def test_evaluator_score_positive():
    h = make_hypothesis()
    score = Evaluator().score(h)
    assert score > 0.0


def test_evaluator_rank_sorted_descending():
    h1 = make_hypothesis("H1", risk=0.1, feasibility=0.95, accuracy=0.95)
    h2 = make_hypothesis("H2", risk=0.4, feasibility=0.5, accuracy=0.5)
    ranked = Evaluator().rank([h1, h2])
    assert ranked[0][0].id == "H1"
    assert ranked[0][1] >= ranked[1][1]


def test_evaluator_top_n():
    hs = [make_hypothesis(f"H{i}") for i in range(10)]
    top = Evaluator().top_n(hs, n=3)
    assert len(top) == 3


# ─── 4. fusion_engine ───────────────────────────────────────────────────────

def test_linear_fusion_creates_new_hypothesis():
    h1 = make_hypothesis("H1", novelty=0.9)
    h2 = make_hypothesis("H2", novelty=0.3)
    fused = FusionEngine.linear_fusion(h1, h2, alpha=0.5)
    assert fused.source == "linear_fusion"
    assert abs(fused.vector["novelty"] - 0.6) < 0.01


def test_nonlinear_fusion_all_keys_present():
    h1 = make_hypothesis("H1")
    h2 = make_hypothesis("H2")
    fused = FusionEngine.nonlinear_fusion(h1, h2)
    assert set(fused.vector.keys()) == set(VECTOR_KEYS)
    assert all(0.0 <= v <= 1.0 for v in fused.vector.values())


# ─── 5. jump_explorer ───────────────────────────────────────────────────────

def test_perturb_changes_vector():
    h = make_hypothesis()
    original_vector = dict(h.vector)
    perturbed = JumpExplorer.perturb(h, scale=0.2)
    # 少なくとも1軸は変化しているはず
    changed = any(perturbed.vector[k] != original_vector[k] for k in VECTOR_KEYS)
    assert changed
    assert perturbed.source == "jump"


def test_opposite_jump_moves_away_from_worst():
    best = make_hypothesis("best", accuracy=0.9, feasibility=0.9)
    worst = make_hypothesis("worst", accuracy=0.2, feasibility=0.2)
    jumped = JumpExplorer.opposite_jump(best, worst, factor=0.3)
    # accuracy は 0.9 + 0.3*(0.9-0.2) = 0.9+0.21=1.11 → clamp → 1.0
    assert jumped.vector["accuracy"] >= best.vector["accuracy"]
    assert jumped.source == "jump"


# ─── 6. self_corrector ──────────────────────────────────────────────────────

def test_self_corrector_moves_toward_ideal():
    ideal = DEFAULT_IDEAL.copy()
    h = make_hypothesis(accuracy=0.3)  # 理想(0.95)から遠い
    corrected = SelfCorrector(ideal, step=0.5).revise(h)
    # accuracy が理想に近づいているはず
    assert corrected.vector["accuracy"] > 0.3
    assert corrected.source == "self_corrected"


# ─── 7. relational_layer ────────────────────────────────────────────────────

def test_relational_layer_returns_scores_in_range():
    h = make_hypothesis()
    layer = RelationalLayer()
    scores = layer.evaluate(h)
    for attr in ["understanding", "alignment", "respect", "harmony"]:
        v = getattr(scores, attr)
        assert 0.0 <= v <= 1.0, f"{attr}={v} out of range"


def test_relational_layer_coop_bonus():
    h_plain = make_hypothesis(content="普通の仮説")
    h_coop  = make_hypothesis(content="共存と協調を重視した解決策")
    layer = RelationalLayer()
    scores_plain = layer.evaluate(h_plain)
    scores_coop  = layer.evaluate(h_coop)
    assert scores_coop.score >= scores_plain.score


# ─── 8. generator (mock) ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generator_fallback_returns_hypotheses():
    gen = HypothesisGenerator(llm_client=None)
    hs = await gen.generate("テスト問題", count=3)
    assert len(hs) == 3
    for h in hs:
        assert isinstance(h, Hypothesis)
        assert h.content != ""
        assert set(h.vector.keys()) == set(VECTOR_KEYS)


# ─── 9. vectorizer (mock) ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_vectorizer_skips_already_vectorized():
    h = make_hypothesis(accuracy=0.8)  # ゼロベクトルでない
    vec = Vectorizer(llm_client=None)
    original_accuracy = h.vector["accuracy"]
    result = await vec.vectorize([h])
    assert result[0].vector["accuracy"] == original_accuracy  # 変更なし


@pytest.mark.asyncio
async def test_vectorizer_fills_zero_vector():
    h = Hypothesis(id="H0", content="test", vector=zero_vector())
    vec = Vectorizer(llm_client=None)
    result = await vec.vectorize([h])
    # ルールベースで埋まっているはず（全軸が 0.0 でない）
    assert any(v > 0.0 for v in result[0].vector.values())


# ─── 10. engine end-to-end (mock) ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_engine_mock_run_returns_field_result():
    config = EngineConfig(mock_llm=True, iterations=2)
    engine = HypothesisFieldEngine(config)
    result = await engine.run("テスト問題")
    assert isinstance(result, FieldResult)
    assert result.best is not None
    assert len(result.ranked) > 0
    assert result.iterations_run == 2


@pytest.mark.asyncio
async def test_engine_mock_all_presets():
    for preset in ["default", "creative", "safe", "balanced"]:
        config = EngineConfig(mock_llm=True, ideal_preset=preset, iterations=1)
        engine = HypothesisFieldEngine(config)
        result = await engine.run("プリセットテスト")
        assert result.best.score >= 0.0, f"preset={preset} returned negative score"


@pytest.mark.asyncio
async def test_build_engine_shorthand():
    engine = build_engine(mock_llm=True, iterations=1)
    result = await engine.run("ショートハンドテスト")
    assert isinstance(result, FieldResult)


@pytest.mark.asyncio
async def test_engine_accepts_hypothesis_list():
    hs = [make_hypothesis(f"H{i}", content=f"仮説{i}") for i in range(3)]
    config = EngineConfig(mock_llm=True, iterations=1)
    engine = HypothesisFieldEngine(config)
    result = await engine.run(hs)
    assert isinstance(result, FieldResult)
    assert len(result.ranked) > 0


@pytest.mark.asyncio
async def test_engine_best_vector_in_range():
    config = EngineConfig(mock_llm=True, iterations=2)
    engine = HypothesisFieldEngine(config)
    result = await engine.run("ベクトル範囲テスト")
    for k, v in result.best.vector.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"


# ─── 11. providers ──────────────────────────────────────────────────────────

from hypothesis_field._providers import DEFAULT_MODELS, SUPPORTED_PROVIDERS, create_client
from hypothesis_field._providers._ollama import OllamaClient
from hypothesis_field._providers._openai import OpenAIClient
from hypothesis_field._providers._anthropic import AnthropicClient


def test_default_models_all_providers():
    for p in SUPPORTED_PROVIDERS:
        assert p in DEFAULT_MODELS
        assert isinstance(DEFAULT_MODELS[p], str)


def test_engine_config_resolved_model_ollama():
    config = EngineConfig(provider="ollama")
    assert config.resolved_model() == DEFAULT_MODELS["ollama"]


def test_engine_config_resolved_model_openai():
    config = EngineConfig(provider="openai")
    assert config.resolved_model() == DEFAULT_MODELS["openai"]


def test_engine_config_resolved_model_anthropic():
    config = EngineConfig(provider="anthropic")
    assert config.resolved_model() == DEFAULT_MODELS["anthropic"]


def test_engine_config_custom_model_overrides_default():
    config = EngineConfig(provider="openai", model="gpt-4o-mini")
    assert config.resolved_model() == "gpt-4o-mini"


def test_create_client_ollama_returns_ollama_client():
    client = create_client("ollama")
    assert isinstance(client, OllamaClient)


def test_create_client_openai_returns_openai_client():
    # fail-fast: API キーなしは ValueError — ダミーキーで型だけ確認
    client = create_client("openai", api_key="sk-test-dummy")
    assert isinstance(client, OpenAIClient)


def test_create_client_anthropic_returns_anthropic_client():
    # fail-fast: API キーなしは ValueError — ダミーキーで型だけ確認
    client = create_client("anthropic", api_key="sk-ant-test-dummy")
    assert isinstance(client, AnthropicClient)


def test_create_client_unknown_provider_raises():
    with pytest.raises(ValueError, match="未知のプロバイダー"):
        create_client("gemini")


def test_engine_config_unknown_provider_raises_on_init():
    config = EngineConfig(provider="gemini", mock_llm=False)
    with pytest.raises(ValueError):
        HypothesisFieldEngine(config)


@pytest.mark.asyncio
async def test_engine_mock_overrides_any_provider():
    for provider in SUPPORTED_PROVIDERS:
        config = EngineConfig(provider=provider, mock_llm=True, iterations=1)
        engine = HypothesisFieldEngine(config)
        result = await engine.run("プロバイダー無効化テスト")
        assert isinstance(result, FieldResult)


def test_build_engine_with_openai_provider():
    engine = build_engine(provider="openai", mock_llm=True)
    assert engine._config.provider == "openai"
    assert engine._config.resolved_model() == DEFAULT_MODELS["openai"]


def test_openai_client_uses_env_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-from-env")
    client = OpenAIClient()
    assert client._api_key == "test-key-from-env"


def test_anthropic_client_uses_env_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-ant-key-from-env")
    client = AnthropicClient()
    assert client._api_key == "test-ant-key-from-env"


def test_openai_client_explicit_key_takes_priority(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    client = OpenAIClient(api_key="explicit-key")
    assert client._api_key == "explicit-key"


# ---------------------------------------------------------------------------
# セキュリティ機能テスト
# ---------------------------------------------------------------------------

from hypothesis_field._security import (
    sanitize_input, sanitize_output, validate_ollama_url,
    mask_secrets, validate_engine_params,
    MAX_PROBLEM_LENGTH, MAX_CONTENT_LENGTH, MAX_ITERATIONS, MAX_HYPOTHESIS_COUNT,
)


def test_sanitize_input_strips_control_chars():
    assert "\x00" not in sanitize_input("hello\x00world")
    assert "\x01" not in sanitize_input("hello\x01world")


def test_sanitize_input_preserves_newline_and_tab():
    result = sanitize_input("line1\nline2\ttabbed")
    assert "\n" in result
    assert "\t" in result


def test_sanitize_input_truncates_long_text():
    long = "a" * (MAX_PROBLEM_LENGTH + 100)
    result = sanitize_input(long)
    assert len(result) == MAX_PROBLEM_LENGTH


def test_sanitize_output_truncates_long_content():
    long = "x" * (MAX_CONTENT_LENGTH + 100)
    result = sanitize_output(long)
    assert len(result) == MAX_CONTENT_LENGTH


def test_validate_ollama_url_accepts_localhost():
    assert validate_ollama_url("http://localhost:11434") == "http://localhost:11434"


def test_validate_ollama_url_rejects_non_http():
    with pytest.raises(ValueError, match="http"):
        validate_ollama_url("ftp://localhost:11434")


def test_validate_ollama_url_rejects_file_scheme():
    with pytest.raises(ValueError):
        validate_ollama_url("file:///etc/passwd")


def test_validate_ollama_url_blocks_metadata_endpoint():
    with pytest.raises(ValueError):
        validate_ollama_url("http://169.254.169.254/latest/meta-data/")


def test_validate_ollama_url_blocks_google_metadata():
    with pytest.raises(ValueError):
        validate_ollama_url("http://metadata.google.internal/")


def test_mask_secrets_replaces_openai_key():
    text = "error: invalid key sk-abc123XYZ456 found"
    assert "sk-***MASKED***" in mask_secrets(text)
    assert "sk-abc123XYZ456" not in mask_secrets(text)


def test_mask_secrets_replaces_anthropic_key():
    text = "key=sk-ant-abcdef123456"
    assert "sk-***MASKED***" in mask_secrets(text)


def test_mask_secrets_passes_clean_text():
    text = "no secrets here"
    assert mask_secrets(text) == text


def test_validate_engine_params_valid():
    validate_engine_params(3, 5)  # should not raise


def test_validate_engine_params_iterations_too_high():
    with pytest.raises(ValueError, match="iterations"):
        validate_engine_params(MAX_ITERATIONS + 1, 3)


def test_validate_engine_params_count_too_high():
    with pytest.raises(ValueError, match="hypothesis_count"):
        validate_engine_params(3, MAX_HYPOTHESIS_COUNT + 1)


def test_engine_config_rejects_excess_iterations():
    with pytest.raises(ValueError, match="iterations"):
        EngineConfig(iterations=MAX_ITERATIONS + 1, mock_llm=True)


def test_engine_config_rejects_excess_hypothesis_count():
    with pytest.raises(ValueError, match="hypothesis_count"):
        EngineConfig(hypothesis_count=MAX_HYPOTHESIS_COUNT + 1, mock_llm=True)


def test_openai_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API"):
        OpenAIClient()


def test_anthropic_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API"):
        AnthropicClient()
