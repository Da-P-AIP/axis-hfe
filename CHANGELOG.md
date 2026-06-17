# Changelog — axis-hfe

## [0.4.1] — 2026-06-17

### Fixed
- **Gemini: JSONモード＋スキーマを全パイプライン呼び出しに適用、パース失敗リトライ、mock_llm=False 時の fail-loud 化**
  - `GeminiClient.generate()` に `response_schema` パラメータを追加し、
    呼び出し側（generator/vectorizer）が期待するJSON構造（配列/オブジェクト）を
    `GenerateContentConfig.response_schema` で固定できるようにした。
    これにより `response_mime_type` だけでなく構造レベルでも応答を拘束する。
  - `hypothesis_field/_json_utils.py` を新設し、全LLM応答の `json.loads` 前に通す
    共通サニタイザ（`strip_fences` / `extract_json_array` / `extract_json_object`）を提供。
    generator と vectorizer の両方で使用する。
  - `HypothesisGenerator` と `Vectorizer` にパース失敗時リトライを追加（最大2回再試行）。
    JSONが抽出できない場合は同じプロンプトで最大3回試みてから判断する。
  - `fail_loud` フラグを追加（`HypothesisGenerator`/`Vectorizer`）:
    `mock_llm=False`（実プロバイダー指定）時は全リトライ消費後に例外を上げ、
    **絶対に mock 定数へ無言フォールバックしない**。`mock_llm=True` 時は既存の
    サイレントフォールバック挙動を維持する。
  - `HypothesisFieldEngine` が `fail_loud=not config.mock_llm` を
    Generator/Vectorizer に自動伝播するよう修正。

### Changed
- バージョン: `0.4.0` → `0.4.1`（バグ修正パッチ）

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
