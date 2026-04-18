@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
echo.
echo  Axis-HFE Demo Launcher
echo  ======================
echo.
echo  1. モックモード（LLMなし・即時動作）
echo  2. Ollama（ローカルLLM）
echo  3. OpenAI （OPENAI_API_KEY 必要）
echo  4. Anthropic / Claude （ANTHROPIC_API_KEY 必要）
echo.
set /p CHOICE="番号を入力してください [1-4]: "

if "%CHOICE%"=="1" (
    python demo.py --mock
    goto end
)
if "%CHOICE%"=="2" (
    python demo.py
    goto end
)
if "%CHOICE%"=="3" (
    if "%OPENAI_API_KEY%"=="" (
        set /p OPENAI_API_KEY="OPENAI_API_KEY を入力してください: "
    )
    python demo.py --openai
    goto end
)
if "%CHOICE%"=="4" (
    if "%ANTHROPIC_API_KEY%"=="" (
        set /p ANTHROPIC_API_KEY="ANTHROPIC_API_KEY を入力してください: "
    )
    python demo.py --anthropic
    goto end
)

echo 無効な選択です。
:end
pause
