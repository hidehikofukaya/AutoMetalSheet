@echo off
setlocal
cd /d "%~dp0"

set PYTHON=C:\Users\hide2\anaconda3\python.exe
set SCRIPT=%~dp0src\r0\aggregate_runs.py

if "%~2"=="" (
    echo Usage:
    echo   %~nx0 RUNS_ROOT OUTPUT_JSON [BASELINE_LABEL] [CANDIDATE_LABEL]
    exit /b 1
)

set BASELINE=%~3
if "%BASELINE%"=="" set BASELINE=coarse
set CANDIDATE=%~4
if "%CANDIDATE%"=="" set CANDIDATE=refine3

"%PYTHON%" "%SCRIPT%" ^
  --root "%~1" ^
  --output "%~2" ^
  --cohort-config "%~dp0configs\r0_cohort.yaml" ^
  --baseline-label "%BASELINE%" ^
  --candidate-label "%CANDIDATE%"
