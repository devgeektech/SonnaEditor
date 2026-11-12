@echo off
setlocal
cd /d "%~dp0"

echo =================================================
echo  Starting Sonna Editor (Saha) for Windows
echo =================================================

where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [saha] uv package manager not found on PATH.
    echo [saha] Please install uv from https://astral.sh/uv or run:
    echo        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    pause
    exit /b 1
)

where node >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [saha] Node.js not found on PATH.
    echo [saha] Please install Node.js LTS from https://nodejs.org/
    pause
    exit /b 1
)

echo [saha] Starting desktop application...
uv run python scripts\run_app.py %*
pause
