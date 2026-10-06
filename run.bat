@echo off
title NSA Cognitive Runtime Server
color 0A

echo =====================================================================
echo           Neural State Architecture (NSA) Cognitive Server
echo =====================================================================
echo.

:: Detect current WSL2 IP address for OpenWebUI connection
for /f "tokens=*" %%i in ('wsl -d Ubuntu-20.04 -e bash -c "hostname -I"') do set WSL_ALL_IPS=%%i
for /f "tokens=1" %%a in ("%WSL_ALL_IPS%") do set WSL_IP=%%a

echo [INFO] WSL2 IP Address: %WSL_IP%
echo [INFO] OpenWebUI URL  : http://%WSL_IP%:8000/v1
echo [INFO] Starting NSA Cognitive Server (System 1 + System 2)...
echo.
echo Press Ctrl+C in this window to stop the server anytime.
echo ---------------------------------------------------------------------
echo.

wsl -d Ubuntu-20.04 -e bash -lic "cd /home/adam/dev/neural-state-architecture && make serve-nsa-local"

echo.
echo Server stopped.
pause
