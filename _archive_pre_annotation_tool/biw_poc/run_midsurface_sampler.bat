@echo off
setlocal
cd /d "%~dp0"

set PYTHON=C:\Users\hide2\anaconda3\python.exe
set SCRIPT=%~dp0src\preprocess\midsurface_sampler.py
set OUTPUT_ROOT=C:\Users\hide2\IdeaBox\Mesh_Generater\output

echo ================================================================
echo midsurface_sampler  --  Mid-surface point cloud + UDF dataset
echo                         (Stage A of CLAUDE.md section 13, Round 6)
echo ================================================================
echo Python      : %PYTHON%
echo Output root : %OUTPUT_ROOT%
echo.
echo Requires body###_filled.stp (run hole_filler.py first, R6-03).
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

"%PYTHON%" -c "import trimesh; print('  trimesh   : OK', trimesh.__version__)"
if %ERRORLEVEL% neq 0 (
    echo ERROR: trimesh not found. Run: conda install -c conda-forge trimesh
    pause & exit /b 1
)

"%PYTHON%" -c "import h5py; print('  h5py      : OK', h5py.__version__)"
if %ERRORLEVEL% neq 0 (
    echo ERROR: h5py not found. Run: conda install -c conda-forge h5py
    pause & exit /b 1
)
echo.

if "%~1"=="" (
    echo Usage:
    echo   %~nx0                          -- batch all parts in output root
    echo   %~nx0 path\to\body003_filled.stp  -- single hole-filled body
    echo   %~nx0 path\to\hierarchy.json   -- single part (all sheet_metal bodies)
    echo.
    echo Running batch on: %OUTPUT_ROOT%
    echo Press any key to continue or Ctrl+C to abort...
    pause > nul
    "%PYTHON%" "%SCRIPT%" --batch-dir "%OUTPUT_ROOT%"
) else if "%~x1"==".stp" (
    echo Processing single STP: %~1
    "%PYTHON%" "%SCRIPT%" "%~1" %2 %3 %4 %5
) else if "%~x1"==".json" (
    echo Processing hierarchy: %~1
    "%PYTHON%" "%SCRIPT%" --hierarchy "%~1" %2 %3 %4 %5
) else (
    echo ERROR: Unrecognized argument: %~1
    exit /b 1
)

echo.
echo Done.
pause
