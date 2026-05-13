@echo off
setlocal
echo ==============================================
echo Launching KernelWeave Ollama Integration...
echo ==============================================

:: Default model if none specified
set MODEL=%~1
if "%MODEL%"=="" set MODEL=gemma4:e2b

python "%~dp0kernelweave_ollama.py" --model %MODEL%
endlocal
