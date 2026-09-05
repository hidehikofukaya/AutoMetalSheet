@echo off
setlocal
cd /d "%~dp0"

set PYTHON=C:\Users\hide2\anaconda3\python.exe
set SCRIPT=%~dp0src\viewer\stp_viewer.py
set OUTPUT_ROOT=C:\Users\hide2\IdeaBox\Mesh_Generater\output

echo ================================================================
echo BIW STP Viewer  --  Part browser with Hole Editor
echo ================================================================
echo Python      : %PYTHON%
echo Output root : %OUTPUT_ROOT%
echo.
echo Controls:
echo   Left panel  -- Select a body to display it in 3D
echo   [Edit Holes] -- Enter hole edit mode for selected body
echo   Hole list   -- Check holes to mark for removal
echo   [Fill All]  -- Mark all detected holes
echo   [Save]      -- Apply defeaturing and write STP (original backed up .stp.bak)
echo.

if not exist "%PYTHON%" (
    echo ERROR: Python not found at %PYTHON%
    pause & exit /b 1
)

echo Checking dependencies...
"%PYTHON%" -c "from PyQt5.QtWidgets import QApplication; print('  PyQt5     : OK')"
if %ERRORLEVEL% neq 0 (
    echo ERROR: PyQt5 not found. Run: conda install pyqt
    pause & exit /b 1
)
"%PYTHON%" -c "from pyvistaqt import QtInteractor; print('  pyvistaqt : OK')"
if %ERRORLEVEL% neq 0 (
    echo ERROR: pyvistaqt not found. Run: conda install -c conda-forge pyvistaqt
    pause & exit /b 1
)
"%PYTHON%" -c "from OCC.Core.STEPControl import STEPControl_Reader; print('  pythonocc : OK')"
if %ERRORLEVEL% neq 0 (
    echo ERROR: pythonocc not found. Run: conda install -c conda-forge pythonocc-core
    pause & exit /b 1
)
echo.

if "%~1"=="" (
    echo Opening all parts in: %OUTPUT_ROOT%
    echo.
    echo Press any key to launch...
    pause > nul
    "%PYTHON%" "%SCRIPT%" --dir "%OUTPUT_ROOT%"
) else if "%~x1"==".json" (
    echo Opening hierarchy: %~1
    "%PYTHON%" "%SCRIPT%" "%~1"
) else (
    echo Opening directory: %~1
    "%PYTHON%" "%SCRIPT%" --dir "%~1"
)

pause
