$ErrorActionPreference = "Stop"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv was not found on PATH. Install uv, then run this command again. See RUN.md for setup instructions."
}
uv run python scripts\run_app.py @args
