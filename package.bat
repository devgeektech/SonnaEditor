@echo off
setlocal
cd /d "%~dp0"

echo =================================================
echo  Packaging Saha for Distribution (Mac + Windows)
echo =================================================

REM Prepend standard directories for nodejs, uv, and python to PATH
if exist "%USERPROFILE%\.local\bin" set "PATH=%USERPROFILE%\.local\bin;%PATH%"

where uv >nul 2>&1
if %ERRORLEVEL% equ 0 (
    uv run python scripts\package_for_client.py --output dist\saha-app-bundle.zip
) else if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe scripts\package_for_client.py --output dist\saha-app-bundle.zip
) else (
    python scripts\package_for_client.py --output dist\saha-app-bundle.zip
)

if %ERRORLEVEL% equ 0 (
    echo.
    echo [saha] Package created successfully at: dist\saha-app-bundle.zip
) else (
    echo.
    echo [saha] Packaging encountered an error.
)
pause
