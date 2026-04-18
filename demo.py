"""
demo.py - Axis-HFE インタラクティブデモ

起動方法:
    python demo.py              # Ollama 自動検出
    python demo.py --mock       # LLMなし（即時動作・テスト用）
    python demo.py --openai     # OpenAI 使用
    python demo.py --anthropic  # Anthropic (Claude) 使用
"""
import asyncio
import os
import sys
import time

# API キーを .env ファイルで管理したい場合:
#   pip install python-dotenv
#   from dotenv import load_dotenv; load_dotenv()
# その後 OPENAI_API_KEY や ANTHROPIC_API_KEY を .env に書けば自動で読み込まれます。
# ※ .env は .gitignore に追加して絶対にコミットしないこと。

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hypothesis_field import HypothesisFieldEngine, EngineConfig, IDEAL_PRESETS
from hypothesis_field._providers._ollama import OllamaClient

# --- 表示ユーティリティ -----------------------------------------------------

W = 64  # 画面幅

def hr(char="="):
    print(char * W)

def title(text):
    hr()
    pad = (W - len(text) - 2) // 2
    print(" " * pad + f"[ {text} ]")
    hr()

def section(text):
    print(f"\n--- {text} " + "-" * (W - len(text) - 5))

def vector_bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"

def print_hypothesis(rank: int, h, show_vector: bool = True):
    label = {
        "generated":       "生成",
        "linear_fusion":   "線形合成",
        "nonlinear_fusion":"非線形合成",
        "jump":            "跳躍探索",
        "self_corrected":  "自己修正",
    }.get(h.source, h.source)

    print(f"  [{rank}] スコア {h.score:.4f}  ({label})")
    # content を折り返して表示
    content = h.content
    max_w = W - 6
    while len(content) > max_w:
        print(f"      {content[:max_w]}")
        content = content[max_w:]
    print(f"      {content}")

    if show_vector:
        for k, v in h.vector.items():
            bar = vector_bar(v)
            print(f"      {k:12s} {v:.3f} {bar}")
    print()

def print_banner():
    print()
    hr("=")
    print("  Axis-HFE - Hypothesis Field Engine")
    print("  AI no suiron wo 1-hon no kotae kara")
    print("  tajigen kuukan no tansakuhe")
    hr("=")
    print()

def print_provider_info(provider: str, model: str, mock: bool):
    section("プロバイダー設定")
    if mock:
        print("  モード    : モック（LLMなし・テスト用）")
        print("  注意      : フォールバック仮説を使用します")
    else:
        print(f"  プロバイダー: {provider}")
        print(f"  モデル      : {model}")
    print()

def print_presets():
    section("理想ベクトル プリセット一覧")
    presets = {
        "1": ("default",  "汎用バランス型"),
        "2": ("creative", "新規性・発散重視（アイデア出し）"),
        "3": ("safe",     "精度・一貫性重視（リスク管理）"),
        "4": ("balanced", "全軸バランス（意思決定支援）"),
    }
    for k, (name, desc) in presets.items():
        print(f"  {k}. {name:10s} : {desc}")
    return presets

# --- 設定解析 ---------------------------------------------------------------

def parse_args():
    args = sys.argv[1:]
    mock      = "--mock"      in args
    use_open  = "--openai"    in args
    use_claude= "--anthropic" in args

    if mock:
        return dict(provider="ollama", model="(mock)", mock_llm=True)
    if use_open:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("[ERROR] 環境変数 OPENAI_API_KEY が設定されていません。")
            sys.exit(1)
        return dict(provider="openai", model="gpt-4o", api_key=key, mock_llm=False)
    if use_claude:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            print("[ERROR] 環境変数 ANTHROPIC_API_KEY が設定されていません。")
            sys.exit(1)
        return dict(provider="anthropic", model="claude-sonnet-4-6", api_key=key, mock_llm=False)
    # デフォルト: Ollama 自動検出
    return dict(provider="ollama", model="gemma4:e4b", mock_llm=False)

# --- Ollama 自動検出 ---------------------------------------------------------

async def check_ollama() -> bool:
    client = OllamaClient()
    ok = await client.health_check()
    await client.close()
    return ok

# --- メインループ ------------------------------------------------------------

async def run_demo(cfg: dict):
    print_banner()

    # Ollama 自動フォールバック
    if not cfg["mock_llm"] and cfg["provider"] == "ollama":
        print("Ollama 接続確認中...")
        ok = await check_ollama()
        if not ok:
            print("  -> Ollama が起動していません。モックモードで起動します。")
            print("  (Ollama を起動して再実行すると実際のLLMで動きます)")
            cfg["mock_llm"] = True
            cfg["model"] = "(mock)"
        else:
            print("  -> Ollama 接続OK")
        print()

    # プリセット選択
    print_presets()
    choice = input("\nプリセットを選んでください [1-4, Enter=1]: ").strip() or "1"
    preset_map = {"1": "default", "2": "creative", "3": "safe", "4": "balanced"}
    ideal_preset = preset_map.get(choice, "default")

    # イテレーション数
    iter_input = input("ループ回数を入力してください [1-5, Enter=2]: ").strip() or "2"
    try:
        iterations = max(1, min(5, int(iter_input)))
    except ValueError:
        iterations = 2

    print()
    print_provider_info(cfg["provider"], cfg["model"], cfg["mock_llm"])

    # エンジン構築
    config = EngineConfig(
        provider     = cfg["provider"],
        model        = None if cfg["mock_llm"] else cfg["model"],
        api_key      = cfg.get("api_key"),
        ideal_preset = ideal_preset,
        iterations   = iterations,
        mock_llm     = cfg["mock_llm"],
    )
    engine = HypothesisFieldEngine(config)

    # --- 対話ループ -------------------------------------------------------
    print("=" * W)
    print("  問題を入力して Enter で推論開始")
    print("  'q' または空 Enter で終了")
    print("=" * W)

    session = 0
    while True:
        print()
        problem = input("問題 > ").strip()
        if not problem or problem.lower() == "q":
            break

        session += 1
        title(f"推論 #{session}")
        print(f"  問題: {problem}")
        print(f"  プリセット: {ideal_preset}  ループ: {iterations}回")
        print()

        start = time.time()
        print("  推論中", end="", flush=True)

        try:
            result = await engine.run(problem, iterations=iterations)
        except Exception as e:
            print(f"\n  [ERROR] {e}")
            continue

        elapsed = time.time() - start
        for _ in range(3):
            print(".", end="", flush=True)
            await asyncio.sleep(0.1)
        print(f"  完了 ({elapsed:.1f}s)")

        # --- 最良仮説 ---------------------------------------------
        section("最良仮説")
        print_hypothesis(1, result.best, show_vector=True)

        # --- 全仮説ランキング -------------------------------------
        section(f"全仮説ランキング（{len(result.ranked)}本）")
        for i, h in enumerate(result.ranked, 1):
            print_hypothesis(i, h, show_vector=(i == 1))

        # --- 統計 -------------------------------------------------
        section("統計")
        sources = {}
        for h in result.ranked:
            sources[h.source] = sources.get(h.source, 0) + 1
        for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
            label = {
                "generated":       "LLM生成",
                "linear_fusion":   "線形合成",
                "nonlinear_fusion":"非線形合成",
                "jump":            "跳躍探索",
                "self_corrected":  "自己修正",
            }.get(src, src)
            print(f"  {label:12s}: {cnt}本")

        scores = [h.score for h in result.ranked]
        print(f"\n  最高スコア: {max(scores):.4f}")
        print(f"  平均スコア: {sum(scores)/len(scores):.4f}")
        print(f"  処理時間  : {elapsed:.2f}s")

        print()
        another = input("別の問題を試しますか？ [Enter=続ける / q=終了]: ").strip()
        if another.lower() == "q":
            break

    print()
    hr()
    print("  Axis-HFE デモ終了")
    print("  Designed by Da-P (Da-P-AIP) | Part of the Cogito Project")
    hr()
    print()


def main():
    cfg = parse_args()
    asyncio.run(run_demo(cfg))


if __name__ == "__main__":
    main()
