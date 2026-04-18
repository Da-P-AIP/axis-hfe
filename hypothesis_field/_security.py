"""
_security.py — 入力サニタイズ・バリデーションユーティリティ

全セキュリティ対策をここに集約し、各モジュールから呼び出す。
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

# --- 定数 -------------------------------------------------------------------

MAX_PROBLEM_LENGTH: int = 8_000   # problem 文字列の最大長（文字数）
MAX_CONTENT_LENGTH: int = 2_000   # 仮説 content の最大長（文字数）
MAX_ITERATIONS: int = 10
MAX_HYPOTHESIS_COUNT: int = 20

# 許可しないクラウドメタデータエンドポイント（SSRF 対策）
_BLOCKED_HOSTS: tuple[str, ...] = (
    "169.254.169.254",        # AWS / Azure IMDSv1
    "metadata.google.internal",
    "100.100.100.200",        # Alibaba Cloud ECS
    "fd00:ec2::254",          # AWS IMDSv2 IPv6
    "metadata.internal",
)

# ログ等に漏洩しやすいパターン（マスク用）
_SECRET_PATTERN = re.compile(
    r'(sk-[A-Za-z0-9\-_]{10,}|sk-ant-[A-Za-z0-9\-_]{10,})',
    re.IGNORECASE,
)

# 制御文字（タブ・改行以外）
_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


# --- 公開 API ---------------------------------------------------------------

def sanitize_input(text: str, max_length: int = MAX_PROBLEM_LENGTH) -> str:
    """
    LLM へ渡すユーザー入力をサニタイズする。

    - NUL バイト・制御文字を除去（タブ・改行は保持）
    - 最大長を超えた場合は切り詰め
    """
    text = _CONTROL_CHARS.sub("", text)
    if len(text) > max_length:
        text = text[:max_length]
    return text.strip()


def sanitize_output(text: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """
    LLM 出力（仮説 content）をサニタイズする。

    - 制御文字除去・長さ制限のみ（内容は変更しない）
    """
    text = _CONTROL_CHARS.sub("", text)
    if len(text) > max_length:
        text = text[:max_length]
    return text.strip()


def validate_ollama_url(url: str) -> str:
    """
    Ollama base URL を検証し、正規化済み URL を返す。

    Raises:
        ValueError: スキームが http/https 以外、またはメタデータエンドポイントの場合
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"無効な Ollama URL: {url!r}") from exc

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Ollama URL は http:// または https:// のみ許可されています: {url!r}\n"
            f"例: http://localhost:11434"
        )

    host = (parsed.hostname or "").lower()
    for blocked in _BLOCKED_HOSTS:
        if blocked in host:
            raise ValueError(
                f"セキュリティ上の理由により、このホストへの接続はブロックされています: {host!r}"
            )

    return url.rstrip("/")


def mask_secrets(text: str) -> str:
    """
    ログ出力前に API キー等の機密文字列をマスクする。

    例: sk-abc123... → sk-***MASKED***
    """
    return _SECRET_PATTERN.sub("sk-***MASKED***", text)


def validate_engine_params(iterations: int, hypothesis_count: int) -> None:
    """
    EngineConfig のパラメータ範囲を検証する。

    Raises:
        ValueError: 範囲外の場合
    """
    if not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError(
            f"iterations は 1〜{MAX_ITERATIONS} の範囲で指定してください（指定値: {iterations}）"
        )
    if not 1 <= hypothesis_count <= MAX_HYPOTHESIS_COUNT:
        raise ValueError(
            f"hypothesis_count は 1〜{MAX_HYPOTHESIS_COUNT} の範囲で指定してください"
            f"（指定値: {hypothesis_count}）"
        )
