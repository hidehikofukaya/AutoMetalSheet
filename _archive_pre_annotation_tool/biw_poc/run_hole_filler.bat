@echo off
setlocal
cd /d "%~dp0"

set PYTHON=C:\Users\hide2\anaconda3\python.exe
set SCRIPT=%~dp0src\preprocess\hole_filler.py
set OUTPUT_ROOT=C:\Users\hide2\IdeaBox\Mesh_Generater\output

echo ================================================================
echo hole_filler  --  Fill jig/bolt holes in sheet metal STEP solids
echo ================================================================
echo Python      : %PYTHON%
echo Output root : %OUTPUT_ROOT%
echo.
echo Thresholds:
echo   Jig holes  (dia <= 5mm)  : fill silently
echo   Bolt holes (dia <= 15mm) : fill + record in JSON
echo   Larger holes             : keep unchanged
echo.

if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    pause & exit /b 1
)

"%PYTHON%" -c "from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Defeaturing; print('  pythonocc : OK')"
if %ERRORLEVEL% neq 0 (
    echo ERROR: pythonocc not found. Run: conda install -c conda-forge pythonocc-core
    pause & exit /b 1
)
echo.

if "%~1"=="" (
    echo Usage:
    echo   %~nx0                          -- batch all parts in output root
    echo   %~nx0 path\to\body003.stp     -- single body STEP file
    echo   %~nx0 path\to\hierarchy.json  -- single part (all sheet_metal bodies)
    echo   %~nx0 --dry-run               -- detect holes only, no output
    echo.
    echo Running batch on: %OUTPUT_ROOT%
    echo Press any key to continue or Ctrl+C to abort...
    pause > nul
    "%PYTHON%" "%SCRIPT%" --batch-dir "%OUTPUT_ROOT%"
) else if "%~1"=="--dry-run" (
    echo DRY RUN mode -- no files will be written
    echo.
    "%PYTHON%" "%SCRIPT%" --batch-dir "%OUTPUT_ROOT%" --dry-run
) else if "%~x1"==".stp" (
    echo Processing single STP: %~1
    "%PYTHON%" "%SCRIPT%" "%~1"
) else if "%~x1"==".json" (
    echo Processing hierarchy: %~1
    "%PYTHON%" "%SCRIPT%" --hierarchy "%~1"
) else (
    echo ERROR: Unrecognized argument: %~1
    exit /b 1
)

echo.
echo Done.
pause
