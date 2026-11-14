@echo off
rem Cassi installieren oder reparieren (Windows): Doppelklick genuegt.
rem Optionen werden durchgereicht, z.B.: install.cmd -SkipModels
rem Ohne Checkout (laedt Cassi nach %USERPROFILE%\cassi), in PowerShell:
rem   irm https://raw.githubusercontent.com/minzink/local-assistant/main/install.ps1 | iex
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
set CASSI_INSTALL_EXIT=%ERRORLEVEL%
if not "%CASSI_INSTALL_EXIT%"=="0" pause
exit /b %CASSI_INSTALL_EXIT%
