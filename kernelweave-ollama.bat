@echo off
setlocal
echo ==============================================
echo Launching KernelWeave Ollama Integration...
echo ==============================================

:: Default model if none specified
set MODEL=%~1
if "%MODEL%"=="" set MODEL=gemma4:e2b

echo.
echo Select Interface:
echo 1. Interactive Terminal (CLI)
echo 2. Glass-Panel Dashboard (GUI)
set /p CHOICE="Enter choice (1-2, default 1): "

if "%CHOICE%"=="2" (
    echo Starting GUI...
    python "%~dp0kernelweave_gui.py" --model %MODEL%
) else (
    echo Starting CLI...
    python "%~dp0kernelweave_ollama.py" --model %MODEL%
)

endlocal
