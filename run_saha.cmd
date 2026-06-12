@echo off
where uv >nul 2>nul
if errorlevel 1 (
  echo uv was not found on PATH. Install uv, then run this command again.
  echo See RUN.md for setup instructions.
  exit /b 1
)
uv run python scripts\run_app.py %*
