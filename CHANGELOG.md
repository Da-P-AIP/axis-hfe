# Changelog — axis-hfe

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
