@echo off
:: Always run from the workspace root regardless of where this script is called from
cd /d "%~dp0\.."
echo Starting Manufacturer Compliance Intelligence API + Frontend
echo ============================================================
echo API:      http://localhost:8000
echo Frontend: http://localhost:5173
echo Docs:     http://localhost:8000/docs
echo.

:: Start FastAPI in background (cwd = workspace root so 'api' package is found)
start "MCI API" C:\Users\Stewart\AppData\Local\Python\bin\python.exe -m uvicorn api.main:app --reload --reload-dir api --reload-dir src --port 8000

:: Wait a moment for API to start
timeout /t 3 /nobreak > nul

:: Start Vite dev server
cd frontend
call npm run dev
