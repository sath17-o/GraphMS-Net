@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Set up the inference environment first. See docs\EVALUATOR_RUN.md
  exit /b 1
)
.venv\Scripts\python.exe scripts\run_pipeline.py --mode patient %*
exit /b %errorlevel%
