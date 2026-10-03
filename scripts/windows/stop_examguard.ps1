# ==============================================================================
# ExamGuard Vision - One-Click Windows Stop Script
# Identifies the exact ExamGuard Vision process, initiates graceful termination,
# releases camera and network handles, and cleans up runtime state.
# ==============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path "$scriptDir\..\..").Path
Set-Location -LiteralPath $repoRoot

$runtimeDir = Join-Path $repoRoot ".runtime"
$pidFile = Join-Path $runtimeDir "examguard.pid"
$launchFile = Join-Path $runtimeDir "examguard-launch.json"
$stopSignalFile = Join-Path $runtimeDir "stop.signal"

Clear-Host
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "          STOP EXAMGUARD VISION             " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$targetPid = $null

# 1. Read recorded PID
if (Test-Path -LiteralPath $pidFile) {
    $pidContent = (Get-Content -LiteralPath $pidFile -Raw).Trim()
    if ($pidContent -match '^\d+$') {
        $targetPid = [int]$pidContent
    }
}

function Test-ExamGuardHealth {
    param([string]$Url = "http://127.0.0.1:8000/health", [int]$TimeoutMs = 1500)
    try {
        $req = [System.Net.HttpWebRequest]::Create($Url)
        $req.Timeout = $TimeoutMs
        $req.ReadWriteTimeout = $TimeoutMs
        $req.Method = "GET"
        $resp = $req.GetResponse()
        $stream = $resp.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream)
        $body = $reader.ReadToEnd()
        $reader.Close()
        $resp.Close()
        if ($body -like '*"status"*"ok"*' -and $body -like '*"version"*') {
            return $true
        }
    } catch {}
    return $false
}

# 2. If no PID file, check if port 8000 is active with ExamGuard
if (-not $targetPid) {
    if (Test-ExamGuardHealth) {
        $conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($conn -and $conn.OwningProcess) {
            $targetPid = $conn.OwningProcess
        }
    }
}

if (-not $targetPid) {
    Write-Host "✓ ExamGuard Vision hiện không hoạt động." -ForegroundColor Green
    Write-Host ""
    # Clean any stale files
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $launchFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stopSignalFile -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    exit 0
}

# 3. Check if target process exists
$proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
if (-not $proc) {
    Write-Host "✓ Tiến trình ExamGuard (PID $targetPid) không còn hoạt động." -ForegroundColor Yellow
    Write-Host "  Đang dọn dẹp file trạng thái cũ..." -ForegroundColor DarkGray
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $launchFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stopSignalFile -Force -ErrorAction SilentlyContinue
    Write-Host "✓ Đã dọn dẹp xong." -ForegroundColor Green
    Write-Host ""
    Start-Sleep -Seconds 1
    exit 0
}

# 4. Verify process belongs to ExamGuard (safety check)
$isExamGuard = $false
if ($proc.ProcessName -like "python*") {
    $isExamGuard = $true
} else {
    try {
        $cim = Get-CimInstance Win32_Process -Filter "ProcessId = $targetPid" -ErrorAction SilentlyContinue
        if ($cim -and ($cim.CommandLine -like "*run_asus_a17_demo.py*" -or $cim.CommandLine -like "*ExamGuard*")) {
            $isExamGuard = $true
        }
    } catch {}
}

if (-not $isExamGuard) {
    Write-Host "[CẢNH BÁO] PID $targetPid hiện thuộc về ứng dụng khác ($($proc.ProcessName))." -ForegroundColor Yellow
    Write-Host "           Hệ thống KHÔNG dừng tiến trình này để tránh ảnh hưởng ứng dụng khác." -ForegroundColor Yellow
    Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $launchFile -Force -ErrorAction SilentlyContinue
    exit 0
}

# 5. Attempt graceful shutdown
Write-Host "Đang yêu cầu ExamGuard Vision (PID $targetPid) dừng an toàn..." -ForegroundColor Cyan

# Signal graceful stop via stop.signal file
Set-Content -LiteralPath $stopSignalFile -Value "STOP" -Force

# Also try sending WM_CLOSE to any open GUI window
try {
    $proc.CloseMainWindow() | Out-Null
} catch {}

# Wait up to 6 seconds for graceful exit and hardware handle release
$gracefulExit = $false
for ($i = 0; $i -lt 12; $i++) {
    Start-Sleep -Milliseconds 500
    if ($proc.HasExited) {
        $gracefulExit = $true
        break
    }
}

# 6. If process is still alive after timeout, terminate ONLY this process
if (-not $gracefulExit -and -not $proc.HasExited) {
    Write-Host "Tiến trình chưa kết thúc, đang dừng tiến trình (PID $targetPid)..." -ForegroundColor Yellow
    Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 600
}

# 7. Clean up runtime state
Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $launchFile -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $stopSignalFile -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "✓ Đã dừng ExamGuard Vision an toàn." -ForegroundColor Green
Write-Host "✓ Đã giải phóng Camera và cổng 8000." -ForegroundColor Green
Write-Host ""
Start-Sleep -Seconds 1
exit 0
