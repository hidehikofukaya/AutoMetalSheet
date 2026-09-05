@echo off
setlocal
cd /d "%~dp0"

set PYTHON=C:\Users\hide2\anaconda3\python.exe
set SCRIPT=%~dp0src\viewer\r0_result_viewer.py
set KMP_DUPLICATE_LIB_OK=TRUE

if "%~1"=="" (
    "%PYTHON%" "%SCRIPT%"
) else (
    "%PYTHON%" "%SCRIPT%" --manifest "%~1"
)

