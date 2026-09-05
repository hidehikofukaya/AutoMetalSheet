@echo off
setlocal
cd /d "%~dp0"

set PYTHON=C:\Users\hide2\anaconda3\python.exe
set SCRIPT=%~dp0src\r0\adaptive_remesh.py

if "%~3"=="" (
    echo Usage:
    echo   %~nx0 INPUT_PLY OUTPUT_DIR TARGET_EDGE_MM
    exit /b 1
)

"%PYTHON%" "%SCRIPT%" ^
  --input "%~1" ^
  --output-dir "%~2" ^
  --target-edge-mm "%~3"

