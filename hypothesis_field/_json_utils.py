"""
hypothesis_field/_json_utils.py
LLM応答からJSONを安全に抽出する共通ユーティリティ。
全プロバイダー・全パイプラインステージで共有する。
"""
from __future__ import annotations

import json
import re
from typing import Optional


def strip_fences(text: str) -> str:
    """markdownコードフェンス（```json...```、```...```）と前後の空白を除去する。"""
    text = text.strip()
    if text.startswith("```"):
        newline = text.find("\n")
        if newline != -1:
            text = text[newline + 1:]
    text = text.rstrip()
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text.strip()


def extract_json_array(text: str) -> list:
    """
    LLM応答からJSON配列を抽出する。先頭の散文・コードフェンスを剥がし、
    複数パターンで最初のJSON配列を返す。失敗時は空リストを返す。
    """
    text = strip_fences(text)

    # パターン1: ```json ... ``` ブロック内の配列（strip後に残っている場合）
    m = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # パターン2: [ ... ] の最初の出現
    m = re.search(r'(\[.*\])', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            chunk = m.group(1)
            last_close = chunk.rfind('}')
            if last_close != -1:
                try:
                    return json.loads(chunk[:last_close + 1] + ']')
                except Exception:
                    pass

    # パターン3: 個別の {...} を全部拾う
    objects = re.findall(r'\{[^{}]+\}', text, re.DOTALL)
    if objects:
        try:
            return [json.loads(o) for o in objects]
        except Exception:
            pass

    return []


def extract_json_object(text: str) -> Optional[dict]:
    """
    LLM応答からJSONオブジェクト（{}）を抽出する。先頭の散文・コードフェンスを剥がし、
    最初のJSON objectを返す。失敗時はNoneを返す。
    """
    text = strip_fences(text)

    # まず全体をパース試行（response_mime_type が正しく効いている場合）
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # { ... } の最初の出現（ネストに対応）
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    # シンプルな単一 { ... }（ネストなし）
    m = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None
