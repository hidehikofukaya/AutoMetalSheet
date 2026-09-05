@echo off
setlocal
cd /d "%~dp0"

set PYTHON=C:\Users\hide2\anaconda3\python.exe
set SCRIPT=%~dp0src\model\reconstruct.py
set KMP_DUPLICATE_LIB_OK=TRUE

echo ================================================================
echo reconstruct  --  Stage C: UDF grid eval + gradient-projection +
echo                   surface reconstruction + thickness offset
echo                   (Stage C of CLAUDE.md section 13, Round 6)
echo ================================================================
echo Python : %PYTHON%
echo.
echo Output is mesh-only (R6-05): an OPEN +-thickness/2 double-shell,
echo NOT a watertight solid / parametric CAD feature tree.
echo.

if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    pause & exit /b 1
)

"%PYTHON%" -c "import torch; print('  torch     : OK', torch.__version__)"
if %ERRORLEVEL% neq 0 (
    echo ERROR: torch not found. See CLAUDE.md section 13 for environment notes.
    pause & exit /b 1
)
echo.

if "%~3"=="" (
    echo Usage:
    echo   %~nx0 checkpoints\best.pt  body003_filled_dataset.h5  body003_reconstructed
    echo.
    echo   arg1: trained checkpoint .pt
    echo   arg2: *_dataset.h5 to supply the input point cloud
    echo   arg3: output path stem ^(no extension^)
    pause & exit /b 1
)

"%PYTHON%" "%SCRIPT%" --ckpt "%~1" --data "%~2" --out "%~3" %4 %5 %6 %7

echo.
echo Done.
pause
