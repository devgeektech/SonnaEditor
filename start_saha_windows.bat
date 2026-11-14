@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo =================================================
echo  Starting Sonna Editor (Saha) for Windows
echo =================================================

REM 1. Clean up cross-platform virtual environment (e.g. unzipped from macOS archive)
if exist ".venv" (
    if not exist ".venv\Scripts\python.exe" (
        echo [saha] Detected non-Windows virtual environment (from macOS zip). Resetting .venv for Windows...
        rmdir /s /q ".venv"
    )
)

REM 2. Clean up cross-platform node_modules (e.g. unzipped from macOS archive)
if exist "saha-app\node_modules" (
    if not exist "saha-app\node_modules\electron\dist\electron.exe" (
        echo [saha] Detected non-Windows frontend binaries (from macOS zip). Resetting node_modules for Windows...
        rmdir /s /q "saha-app\node_modules"
    )
)

REM 3. Prepend standard directories for nodejs, uv, and python to PATH
if exist "C:\Program Files\nodejs" set "PATH=C:\Program Files\nodejs;!PATH!"
if exist "%USERPROFILE%\.local\bin" set "PATH=%USERPROFILE%\.local\bin;!PATH!"
if exist "%APPDATA%\npm" set "PATH=%APPDATA%\npm;!PATH!"
if exist "%LOCALAPPDATA%\Programs\Python\Python311" set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;!PATH!"

where uv >nul 2>&1
if !ERRORLEVEL! neq 0 (
    if exist "%USERPROFILE%\.local\bin\uv.exe" (
        set "PATH=%USERPROFILE%\.local\bin;!PATH!"
    ) else if exist ".venv\Scripts\python.exe" (
        goto :launch_direct
    ) else (
        echo [saha] uv package manager not found on PATH.
        echo [saha] Please install uv from https://astral.sh/uv or run:
        echo        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        pause
        exit /b 1
    )
)

where node >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [saha] Node.js not found on PATH.
    echo [saha] Please install Node.js LTS from https://nodejs.org/
    pause
    exit /b 1
)

echo [saha] Starting desktop application...
uv run python scripts\run_app.py %*
if !ERRORLEVEL! neq 0 (
    echo.
    echo [saha] Application exited with error code !ERRORLEVEL!.
    pause
)
exit /b 0

:launch_direct
echo [saha] Starting desktop application via repo virtual environment...
.venv\Scripts\python.exe scripts\run_app.py %*
if !ERRORLEVEL! neq 0 (
    echo.
    echo [saha] Application exited with error code !ERRORLEVEL!.
    pause
)
exit /b 0
