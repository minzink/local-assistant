@echo off
rem Flay unter Windows einrichten (startet scripts\setup.ps1, auch bei gesperrten Skripten).
rem Optionen werden durchgereicht, z.B.: scripts\setup.cmd -SkipModels
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
set FLAY_SETUP_EXIT=%ERRORLEVEL%
if not "%FLAY_SETUP_EXIT%"=="0" pause
exit /b %FLAY_SETUP_EXIT%
