@echo off
rem Starts FaceVeda (http://127.0.0.1:7890) and, through it, the Acne Lab engine (port 8000). Uses the Python environment that has all packages.
set PY=D:\CLINDERMA\prototype\backend\venv\Scripts\python.exe
cd /d "%~dp0"
"%PY%" app.py
