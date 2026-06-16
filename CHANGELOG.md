# Changelog — axis-hfe

## [0.4.0] — 2026-06-17

### Fixed
- **Gemini: enforce JSON output mode + robust JSON parsing (fixes silent mock fallback)**
  - `GenerateContentConfig` に `response_mime_type="application/json"` を追加し、
    GeminiがMarkdownフェンスや説明文を付けずに純粋なJSONを返すよう強制する。
    これが `intentloop-harness` 較正で「5本中1〜2本しかGeminiスコアにならない」問題の根本修正。
  - 保険として `_strip_json_fences()` ヘルパーを追加。`response_mime_type` が効かない
    場合でも ` ```json ... ``` ` / ` ``` ... ``` ` 形式のフェンスを自動除去してから
    `LLMResponse.content` に詰める。
  - Geminiクライアントの既存挙動（`gemini-2.5-flash` デフォルト、`GEMINI_API_KEY`/
    `GOOGLE_API_KEY` 環境変数、エラー処理）は変更なし。

### Changed
- バージョン: `0.3.0` → `0.4.0`（バグ修正によるマイナー上げ）

## [0.3.0] — 2026-06-16

### Added
- **Gemini プロバイダー** (`provider="gemini"`) — `GeminiClient` を新設 (`_providers/_gemini.py`)
  - google-genai SDK (>=1.0) を使用した非同期実装
  - 環境変数 `GEMINI_API_KEY`（優先）または `GOOGLE_API_KEY`（フォールバック）に対応
  - デフォルトモデル: `gemini-2.5-flash`
- `pip install axis-hfe[gemini]` extra を追加（`google-genai>=1.0,<2.0`）
- `all` extra に `google-genai` を追加
- `GeminiClient` をトップレベル (`from hypothesis_field import GeminiClient`) で公開
- `_security.py` のマスクパターンに Gemini キー (`AIza...`) を追加
- キーワードに `"gemini"`, `"google"` を追加

### Changed
- バージョン: `0.2.0` → `0.3.0`（新プロバイダー追加によるマイナー上げ）

## [0.2.0] — 初回リリース（axis-hfe スタンドアロン版）

- Ollama / OpenAI / Anthropic プロバイダー対応
- 6次元仮説ベクトル推論エンジン
- セキュリティサニタイズ・入力バリデーション
