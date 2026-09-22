@echo off
setlocal
if not exist .venv (
  py -3.11 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install numpy==2.1.3 pandas scipy==1.14.1 joblib scikit-learn==1.6.1
python scripts\smoke_stage13.py
if errorlevel 1 exit /b 1
echo.
echo GRAPHMS STAGE13 SMOKE PASS
