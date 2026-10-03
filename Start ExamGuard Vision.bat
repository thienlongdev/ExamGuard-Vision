@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\windows\start_examguard.ps1"
if %ERRORLEVEL% neq 0 (
    echo.
    echo -------------------------------------------------------------
    echo [THONG BAO] Khoi dong khong thanh cong. Nhan phim bat ky de dong...
    echo -------------------------------------------------------------
    pause >nul
)
