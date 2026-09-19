@echo off
rem Sihti live - double-click to start
python "%~dp0sihti_live.py" %*
if errorlevel 1 pause
