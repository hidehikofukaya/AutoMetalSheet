@echo off
setlocal
cd /d "%~dp0"

set PYTHON=C:\Users\hide2\anaconda3\python.exe
set SCRIPT=%~dp0src\preprocess\annotator.py

echo ================================================================
echo annotator  --  Interactive constraint point annotation tool
echo ================================================================
echo Python : %PYTHON%
echo.
echo Viewer controls:
echo   Left-click : Add constraint point (projects to mid-surface)
echo   T          : Cycle constraint type (weld / bolt / rivet / glue)
echo   D          : Delete last point
echo   U          : Clear all points
echo   S          : Save annotations.json and quit
echo   Q          : Quit WITHOUT saving
echo.

if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    pause & exit /b 1
)

"%PYTHON%" -c "import pyvista; print('  pyvista   : OK', pyvista.__version__)"
if %ERRORLEVEL% neq 0 (
    echo ERROR: pyvista not found. Run: conda install -c conda-forge pyvista
    pause & exit /b 1
)

"%PYTHON%" -c "from OCC.Core.STEPControl import STEPControl_Reader; print('  pythonocc : OK')"
if %ERRORLEVEL% neq 0 (
    echo ERROR: pythonocc not found. Run: conda install -c conda-forge pythonocc-core
    pause & exit /b 1
)
echo.

if "%~1"=="" (
    echo Usage:
    echo   %~nx0 path\body003_filled.stp
    echo   %~nx0 path\body003_filled.stp --partner path\body007_filled.stp
    echo   %~nx0 path\body003_filled.stp --load body003_filled_annotations.json
    echo.
    echo Please provide a body###_filled.stp as first argument.
    pause & exit /b 1
)

set STP_ARG=%~1
set EXTRA_ARGS=

if not "%~2"=="" set EXTRA_ARGS=%EXTRA_ARGS% %2
if not "%~3"=="" set EXTRA_ARGS=%EXTRA_ARGS% %3
if not "%~4"=="" set EXTRA_ARGS=%EXTRA_ARGS% %4
if not "%~5"=="" set EXTRA_ARGS=%EXTRA_ARGS% %5

"%PYTHON%" "%SCRIPT%" "%STP_ARG%" %EXTRA_ARGS%

echo.
echo Done.
pause
