@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (py -3.10 -m venv .venv && .venv\Scripts\python -m pip install -e .)
start "" .venv\Scripts\pythonw.exe -m navsim
