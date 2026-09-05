@echo off
cd /d "%~dp0"
python viewer\server.py %*
pause
