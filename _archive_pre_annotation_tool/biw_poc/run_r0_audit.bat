@echo off
setlocal
cd /d "%~dp0"

set PYTHON=C:\Users\hide2\anaconda3\python.exe
set SCRIPT=%~dp0src\r0\audit_run.py
set PROFILE=%~dp0configs\r0_audit.yaml

if "%~3"=="" (
    echo Usage:
    echo   %~nx0 REFERENCE_PLY OUTPUT_DIR LABEL1=STAGE1_PLY LABEL2=STAGE2_PLY [...]
    exit /b 1
)

set REFERENCE=%~1
set OUTPUT=%~2
shift
shift

set STAGES=
:collect
if "%~1"=="" goto run
set STAGES=%STAGES% --stage "%~1"
shift
goto collect

:run
"%PYTHON%" "%SCRIPT%" --profile "%PROFILE%" --reference "%REFERENCE%" --output-dir "%OUTPUT%" %STAGES%
