@echo off
rem Cassi unter Windows einrichten (startet scripts\setup.ps1, auch bei gesperrten Skripten).
rem Optionen werden durchgereicht, z.B.: scripts\setup.cmd -SkipModels
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set CASSI_SETUP_EXIT=%ERRORLEVEL%
if not "%CASSI_SETUP_EXIT%"=="0" pause
exit /b %CASSI_SETUP_EXIT%
