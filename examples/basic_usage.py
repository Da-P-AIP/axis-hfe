"""
examples/basic_usage.py
Axis-HFE 基本的な使い方サンプル

実行方法:
    cd axis-hfe
    python examples/basic_usage.py

Ollama が起動していない場合は --mock フラグを使う:
    python examples/basic_usage.py --mock
"""
import asyncio
import os
import sys

# API キーを .env ファイルで管理したい場合:
#   pip install python-dotenv
#   from dotenv import load_dotenv; load_dotenv()
# .env に OPENAI_API_KEY="sk-..." などを書けば自動で読み込まれます。
# ※ .env は .gitignore に追加して絶対にコミットしないこと。

# パッケージが未インストールの場合はルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypothesis_field import HypothesisFieldEngine, EngineConfig


async def example_default(mock: bool = False) -> None:
    """基本的な使い方（defaultプリセット）"""
    print("=" * 60)
    print("【例1】デフォルト設定での推論")
    print("=" * 60)

    config = EngineConfig(
        model="gemma4:e4b",
        ideal_preset="default",
        iterations=2,
        mock_llm=mock,
    )
    engine = HypothesisFieldEngine(config)
    result = await engine.run("ローカルAIで月5万円稼ぐ方法")

    print(f"問題: {result.problem}")
    print(f"\n最良仮説:")
    print(f"  内容:   {result.best.content[:80]}...")
    print(f"  スコア: {result.best.score:.4f}")
    print(f"  生成元: {result.best.source}")
    print(f"\n6次元ベクトル:")
    for k, v in result.best.vector.items():
        bar = "#" * int(v * 20)
        print(f"  {k:12s}: {v:.3f}  {bar}")

    print(f"\n全仮説ランキング（上位{len(result.ranked)}本）:")
    for i, h in enumerate(result.ranked, 1):
        print(f"  {i}. [{h.source:15s}] score={h.score:.4f}  {h.content[:50]}...")


async def example_creative(mock: bool = False) -> None:
    """クリエイティブプリセット（新規性重視）"""
    print("\n" + "=" * 60)
    print("【例2】creativeプリセット（新規性・発散重視）")
    print("=" * 60)

    config = EngineConfig(
        model="gemma4:e4b",
        ideal_preset="creative",
        iterations=2,
        mock_llm=mock,
    )
    engine = HypothesisFieldEngine(config)
    result = await engine.run("2030年のAI時代に価値のある職業")

    print(f"最良仮説: {result.best.content[:100]}")
    print(f"novelty={result.best.vector.get('novelty', 0):.3f}  divergence={result.best.vector.get('divergence', 0):.3f}")


async def example_build_engine(mock: bool = False) -> None:
    """build_engine() ショートハンドの使い方"""
    print("\n" + "=" * 60)
    print("【例3】build_engine() ショートハンド")
    print("=" * 60)

    from hypothesis_field import build_engine

    engine = build_engine(
        model="gemma4:e4b",
        ideal_preset="safe",
        iterations=1,
        mock_llm=mock,
    )
    result = await engine.run("新しい薬の副作用リスクを最小化する方法")

    print(f"最良仮説: {result.best.content[:100]}")
    print(f"risk={result.best.vector.get('risk', 0):.3f}  feasibility={result.best.vector.get('feasibility', 0):.3f}")


async def example_openai(api_key: str | None = None) -> None:
    """OpenAI プロバイダーの使い方"""
    print("\n" + "=" * 60)
    print("【例4】OpenAI（GPT-4o）プロバイダー")
    print("=" * 60)

    config = EngineConfig(
        provider="openai",
        model="gpt-4o",
        api_key=api_key,          # None → 環境変数 OPENAI_API_KEY
        ideal_preset="creative",
        iterations=2,
    )
    engine = HypothesisFieldEngine(config)
    result = await engine.run("AIスタートアップが最初の1000万円を稼ぐ方法")

    print(f"最良仮説: {result.best.content[:100]}")
    print(f"novelty={result.best.vector.get('novelty', 0):.3f}")


async def example_anthropic(api_key: str | None = None) -> None:
    """Anthropic プロバイダーの使い方"""
    print("\n" + "=" * 60)
    print("【例5】Anthropic（Claude）プロバイダー")
    print("=" * 60)

    config = EngineConfig(
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key=api_key,          # None → 環境変数 ANTHROPIC_API_KEY
        ideal_preset="balanced",
        iterations=2,
    )
    engine = HypothesisFieldEngine(config)
    result = await engine.run("持続可能なビジネスモデルを設計する")

    print(f"最良仮説: {result.best.content[:100]}")
    print(f"score={result.best.score:.4f}")


async def main() -> None:
    mock       = "--mock" in sys.argv
    use_openai = "--openai" in sys.argv
    use_claude = "--claude" in sys.argv

    openai_key    = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if mock:
        print("[MOCK] Ollama接続なしのモックモードで実行\n")
    else:
        # Ollama 接続確認
        from hypothesis_field._providers._ollama import OllamaClient
        client = OllamaClient()
        ok = await client.health_check()
        await client.close()
        if not ok:
            print("[WARN] Ollama が起動していません。--mock フラグで実行してください。")
            print("   python examples/basic_usage.py --mock\n")
            sys.exit(1)
        print("[OK] Ollama 接続OK\n")

    await example_default(mock)
    await example_creative(mock)
    await example_build_engine(mock)

    if use_openai:
        if not openai_key:
            print("[SKIP] OPENAI_API_KEY が設定されていません")
        else:
            await example_openai(openai_key)

    if use_claude:
        if not anthropic_key:
            print("[SKIP] ANTHROPIC_API_KEY が設定されていません")
        else:
            await example_anthropic(anthropic_key)

    print("\n[DONE] 全サンプル完了")
    print()
    print("他のプロバイダーを試すには:")
    print("  OPENAI_API_KEY=sk-...    python examples/basic_usage.py --openai")
    print("  ANTHROPIC_API_KEY=sk-... python examples/basic_usage.py --claude")


if __name__ == "__main__":
    asyncio.run(main())
