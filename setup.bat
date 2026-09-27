@echo off
REM One-command setup for Windows.   setup.bat        or   setup.bat --ml
cd /d "%~dp0"
python -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ needed'" || exit /b 1
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q --no-deps
if "%1"=="--ml" (
  pip install torch --index-url https://download.pytorch.org/whl/cpu -q
  pip install -r requirements-ml.txt -q
)
if not exist .env copy .env.example .env
python scripts\make_splits.py
python -m pytest -q
echo.
echo Setup complete. Try:
echo   .venv\Scripts\activate
echo   python -m demo_agent.run_demo
echo   streamlit run dashboard\app.py
