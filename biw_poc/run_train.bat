@echo off
setlocal
cd /d "%~dp0"

set PYTHON=C:\Users\hide2\anaconda3\python.exe
set SCRIPT=%~dp0src\model\train.py
set KMP_DUPLICATE_LIB_OK=TRUE

echo ================================================================
echo train  --  Stage B VecSet Autoencoder training
echo            (Stage B of CLAUDE.md section 13, Round 6)
echo ================================================================
echo Python : %PYTHON%
echo.

if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    pause & exit /b 1
)

"%PYTHON%" -c "import torch; print('  torch     : OK', torch.__version__, 'cuda=', torch.cuda.is_available())"
if %ERRORLEVEL% neq 0 (
    echo ERROR: torch not found. See CLAUDE.md section 13 for environment notes.
    pause & exit /b 1
)
echo.

if "%~1"=="" (
    echo Usage:
    echo   %~nx0 path\to\body003_filled_dataset.h5  -- single-part overfit sanity check
    echo   %~nx0 path\to\output_root --epochs 200 --batch-size 2  -- multi-part training
    echo.
    echo Please provide --data ^<path^> as the first argument ^(or any train.py args^).
    pause & exit /b 1
)

"%PYTHON%" "%SCRIPT%" --data %1 %2 %3 %4 %5 %6 %7 %8 %9

echo.
echo Done.
pause
