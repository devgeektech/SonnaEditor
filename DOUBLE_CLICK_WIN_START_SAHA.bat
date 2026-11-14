@echo off
setlocal
cd /d "%~dp0"

echo =================================================
echo  Starting Sonna Editor (Saha) for Windows
echo =================================================

REM Add all possible standard paths to PATH without delayed expansion
if exist "C:\Program Files\nodejs" set "PATH=C:\Program Files\nodejs;%PATH%"
if exist "C:\Program Files (x86)\nodejs" set "PATH=C:\Program Files (x86)\nodejs;%PATH%"
if exist "%LOCALAPPDATA%\Programs\nodejs" set "PATH=%LOCALAPPDATA%\Programs\nodejs;%PATH%"
if exist "%USERPROFILE%\.local\bin" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if exist "%APPDATA%\npm" set "PATH=%APPDATA%\npm;%PATH%"
if exist "%LOCALAPPDATA%\Programs\Python\Python311" set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"
if exist "C:\Windows\System32" set "PATH=%PATH%;C:\Windows\System32;C:\Windows"

REM Clean up cross-platform virtual environment (e.g. from macOS)
if exist ".venv" if not exist ".venv\Scripts\python.exe" (
    echo [saha] Detected non-Windows virtual environment. Resetting .venv for Windows...
    rmdir /s /q ".venv"
)

REM Clean up cross-platform node_modules (e.g. from macOS)
if exist "saha-app\node_modules" if not exist "saha-app\node_modules\electron\dist\electron.exe" (
    echo [saha] Detected non-Windows frontend binaries. Resetting node_modules for Windows...
    rmdir /s /q "saha-app\node_modules"
)

REM Check Node.js
set "NODE_CMD="
if exist "C:\Program Files\nodejs\node.exe" set "NODE_CMD=C:\Program Files\nodejs\node.exe"
if not defined NODE_CMD for %%i in (node.exe) do if not "%%~$PATH:i"=="" set "NODE_CMD=%%~$PATH:i"

if not defined NODE_CMD (
    echo [saha] Node.js not found on your system.
    echo [saha] Please install Node.js LTS from https://nodejs.org/
    pause
    exit /b 1
)

REM Check uv or repo python
set "PYTHON_LAUNCHER="
if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "PYTHON_LAUNCHER=uv run python"
) else (
    for %%i in (uv.exe) do if not "%%~$PATH:i"=="" set "PYTHON_LAUNCHER=uv run python"
)

if not defined PYTHON_LAUNCHER (
    if exist ".venv\Scripts\python.exe" (
        set "PYTHON_LAUNCHER=.venv\Scripts\python.exe"
    ) else (
        for %%i in (python.exe) do if not "%%~$PATH:i"=="" set "PYTHON_LAUNCHER=python"
    )
)

if not defined PYTHON_LAUNCHER (
    echo [saha] Python or uv was not found. Installing uv...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    set "PYTHON_LAUNCHER=uv run python"
)

echo [saha] Starting desktop application...
%PYTHON_LAUNCHER% scripts\run_app.py %*
if errorlevel 1 (
    echo.
    echo [saha] Application exited with error code %ERRORLEVEL%.
    pause
)
