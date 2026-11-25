@echo off
chcp 65001 >nul
echo [1/3] Poisk Python...

py -3.12 --version >nul 2>&1
if %errorlevel%==0 (
    set PY_CMD=py -3.12
) else (
    where py >nul 2>&1
    if %errorlevel%==0 (
        set PY_CMD=py -3
    ) else (
        where python >nul 2>&1
        if %errorlevel%==0 (
            set PY_CMD=python
        ) else (
            echo [OSHIBKA] Python ne nayden! Ustanovite Python 3.10-3.12 s python.org
            pause
            exit /b 1
        )
    )
)

echo [2/3] Podgotovka okruzheniya i ustanovka bibliotek...
if not exist "venv" (
    %PY_CMD% -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo [3/3] Zapusk TirekAI...
streamlit run app.py
pause
