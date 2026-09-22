@echo off
setlocal
if not exist .venv (
  py -3.11 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-evaluation.txt
python scripts\run_pipeline.py --mode evaluation-replay
if errorlevel 1 exit /b 1
echo.
echo GRAPHMS EVALUATION REPLAY PASS
