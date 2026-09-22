@echo off
setlocal
if not exist .venv (
  py -3.11 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-verify.txt
python scripts\run_pipeline.py --mode verify
if errorlevel 1 exit /b 1
echo.
echo GRAPHMS VERIFY PASS
