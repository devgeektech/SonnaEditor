@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Prepend standard directories for nodejs, uv, and python to PATH
if exist "C:\Program Files\nodejs" set "PATH=C:\Program Files\nodejs;!PATH!"
if exist "%USERPROFILE%\.local\bin" set "PATH=%USERPROFILE%\.local\bin;!PATH!"
if exist "%APPDATA%\npm" set "PATH=%APPDATA%\npm;!PATH!"

where uv >nul 2>nul
if !ERRORLEVEL! neq 0 (
  if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PATH=%USERPROFILE%\.local\bin;!PATH!"
    uv run python scripts\run_app.py %*
  ) else if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe scripts\run_app.py %*
  ) else (
    echo uv was not found on PATH. Install uv, then run this command again.
    echo See RUN.md for setup instructions.
    pause
    exit /b 1
  )
) else (
  uv run python scripts\run_app.py %*
)
