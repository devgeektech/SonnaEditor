@echo off
setlocal
cd /d "%~dp0"

echo =================================================
echo  Packaging Saha (Clean Production Distribution)
echo =================================================

REM Add local and standard tools to PATH
if exist "%USERPROFILE%\.local\bin" set "PATH=%USERPROFILE%\.local\bin;%PATH%"

if exist "%USERPROFILE%\.local\bin\uv.exe" (
    uv run python scripts\package_app.py %*
) else if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe scripts\package_app.py %*
) else (
    python scripts\package_app.py %*
)

if %ERRORLEVEL% equ 0 (
    echo.
    echo [saha] Package created successfully!
) else (
    echo.
    echo [saha] Packaging encountered an error.
)
pause
