@echo off
setlocal
cd /d "%~dp0"

set PYTHON=C:\Users\hide2\anaconda3\python.exe
set SCRIPT=%~dp0src\r0\reconstruct_run.py
set PROFILE=%~dp0configs\r0_audit.yaml
set KMP_DUPLICATE_LIB_OK=TRUE

if "%~4"=="" (
    echo Usage:
    echo   %~nx0 CHECKPOINT DATASET_H5 REFERENCE_PLY RUN_DIR
    exit /b 1
)

"%PYTHON%" "%SCRIPT%" ^
  --profile "%PROFILE%" ^
  --checkpoint "%~1" ^
  --data "%~2" ^
  --reference "%~3" ^
  --run-dir "%~4"

