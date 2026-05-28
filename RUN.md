# Run Sonna Editor

This file explains how to set up the Python backend and start the Sonna Editor project on Windows.

## 1) Prepare the Python environment

Open PowerShell and change to the project folder:

```powershell
cd 'F:\Projects\SonnaEditor'
```

Create and activate a virtual environment (one-time):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

Upgrade packaging tools and install Python dependencies:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .[dev]
```

Optional import check:

```powershell
python -c "import torch, torchvision, pandas, fastapi; from PyQt6 import QtWidgets; print('py imports OK')"
```

## 2) Verify the environment

Optional but recommended:

```powershell
python scripts\verify_environment.py
```

## 3) Start the backend API

Keep this terminal open. With the virtual environment active, run:

```powershell
python scripts\serve.py --port 8765
```

The backend should start on `http://127.0.0.1:8765`.

## 4) Install and run the frontend

Open a second terminal and install Node dependencies once:

```powershell
cd .\saha-app
npm install
```

Then start the frontend:

```powershell
npm run dev
```

This starts the Vite + Electron UI and connects it to the backend.

## 5) Run the project

In PowerShell terminal 1:

```powershell
cd 'F:\Projects\SonnaEditor'
& .\.venv\Scripts\Activate.ps1
python scripts\serve.py --port 8765
```

In PowerShell terminal 2:

```powershell
cd 'F:\Projects\SonnaEditor\saha-app'
npm run dev
```
