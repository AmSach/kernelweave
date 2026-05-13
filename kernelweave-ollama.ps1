param (
    [string]$Model = "gemma4:e2b"
)

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Launching KernelWeave Ollama Integration..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

$ScriptPath = Join-Path $PSScriptRoot "kernelweave_ollama.py"
python $ScriptPath --model $Model
