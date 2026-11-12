@echo off
cd /d "%~dp0"
python -B sim/main.py
if errorlevel 1 pause
