@echo off
echo Starting IoT Sensor Data RAG System...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Run the Streamlit app
echo Starting Streamlit application...
streamlit run streamlit_app.py

pause
