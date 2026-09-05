@echo off
cd /d "%~dp0"
python hole_filler\web_server.py %*
pause
