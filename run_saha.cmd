@echo off
setlocal
cd /d "%~dp0"

REM Add all possible standard paths to PATH without delayed expansion
if exist "C:\Program Files\nodejs" set "PATH=C:\Program Files\nodejs;%PATH%"
if exist "C:\Program Files (x86)\nodejs" set "PATH=C:\Program Files (x86)\nodejs;%PATH%"
if exist "%LOCALAPPDATA%\Programs\nodejs" set "PATH=%LOCALAPPDATA%\Programs\nodejs;%PATH%"
if exist "%USERPROFILE%\.local\bin" set "PATH=%USERPROFILE%\.local\bin;%PATH%"
if exist "%APPDATA%\npm" set "PATH=%APPDATA%\npm;%PATH%"
if exist "%LOCALAPPDATA%\Programs\Python\Python311" set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%PATH%"
if exist "C:\Windows\System32" set "PATH=%PATH%;C:\Windows\System32;C:\Windows"

if exist "%USERPROFILE%\.local\bin\uv.exe" (
    uv run python scripts\run_app.py %*
) else if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe scripts\run_app.py %*
) else (
    uv run python scripts\run_app.py %*
)
