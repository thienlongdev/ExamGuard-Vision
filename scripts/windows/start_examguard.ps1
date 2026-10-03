# ==============================================================================
# ExamGuard Vision - One-Click Windows Startup Script
# Validates runtime requirements, starts canonical Asus A17 demo,
# waits for health readiness, and automatically opens the browser dashboard.
# ==============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 1. Resolve project root dynamically from script location
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path "$scriptDir\..\..").Path
Set-Location -LiteralPath $repoRoot

$runtimeDir = Join-Path $repoRoot ".runtime"
$logsDir = Join-Path $runtimeDir "logs"
if (-not (Test-Path -LiteralPath $logsDir)) {
    New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
}

$pidFile = Join-Path $runtimeDir "examguard.pid"
$launchFile = Join-Path $runtimeDir "examguard-launch.json"
$runtimeLog = Join-Path $logsDir "examguard-runtime.log"
$stopSignalFile = Join-Path $runtimeDir "stop.signal"

Clear-Host
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "          EXAMGUARD VISION                  " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# [1/4] Lightweight Environment & Model Validation
Write-Host "[1/4] Kiểm tra môi trường..." -ForegroundColor White

# 1.1 Python environment check
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    Write-Host ""
    Write-Host " [LỖI] Không tìm thấy môi trường Python .venv." -ForegroundColor Red
    Write-Host "       Đường dẫn tìm kiếm: $pythonExe" -ForegroundColor Yellow
    Write-Host ""
    Write-Host " Hướng dẫn cài đặt/khắc phục:" -ForegroundColor Cyan
    Write-Host " 1. Mở PowerShell tại thư mục dự án: $repoRoot" -ForegroundColor White
    Write-Host " 2. Tạo môi trường ảo: python -m venv .venv" -ForegroundColor White
    Write-Host " 3. Kích hoạt và cài đặt dependencies theo tài liệu docs/deployment.md" -ForegroundColor White
    Write-Host ""
    exit 1
}
Write-Host "      ✓ Python (.venv)" -ForegroundColor Green

# 1.2 Configuration & runner script check
$demoScript = Join-Path $repoRoot "scripts\run_asus_a17_demo.py"
$demoConfig = Join-Path $repoRoot "configs\runtime\asus_a17_demo.yaml"
if (-not (Test-Path -LiteralPath $demoScript) -or -not (Test-Path -LiteralPath $demoConfig)) {
    Write-Host ""
    Write-Host " [LỖI] Không tìm thấy file script hoặc cấu hình runtime cần thiết." -ForegroundColor Red
    Write-Host "       - Script: $demoScript" -ForegroundColor Yellow
    Write-Host "       - Cấu hình: $demoConfig" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
Write-Host "      ✓ Cấu hình runtime" -ForegroundColor Green

# 1.3 Lightweight canonical models existence check
$requiredModels = @(
    "models/trained/yolo26m.pt",
    "models/trained/stage1_5_best.pt",
    "models/trained/v4_posture_best.pt",
    "models/trained/v4_headpose_yaw_best.pt",
    "models/trained/stage1_best.pt",
    "models/fallback/posture_320/best_model.pt",
    "models/fallback/headpose_resnet18/best_model.pt"
)

$missingModels = @()
foreach ($relPath in $requiredModels) {
    $fullPath = Join-Path $repoRoot $relPath
    if (-not (Test-Path -LiteralPath $fullPath)) {
        $missingModels += $relPath
    } else {
        $fileSize = (Get-Item -LiteralPath $fullPath).Length
        if ($fileSize -lt 1000) {
            $missingModels += "$relPath (LFS pointer hoặc file rỗng, dung lượng $fileSize bytes)"
        }
    }
}

if ($missingModels.Count -gt 0) {
    Write-Host ""
    Write-Host " [LỖI] Thiếu checkpoint mô hình AI hoặc chưa tải LFS đầy đủ:" -ForegroundColor Red
    foreach ($m in $missingModels) {
        Write-Host "       - $m" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host " Vui lòng chạy lệnh sau để tải các file mô hình:" -ForegroundColor Cyan
    Write-Host "   git lfs pull" -ForegroundColor White
    Write-Host ""
    exit 1
}
Write-Host "      ✓ Mô hình AI (7/7 Checkpoints)" -ForegroundColor Green

# [2/4] Service & Port Checks
Write-Host ""
Write-Host "[2/4] Kiểm tra dịch vụ..." -ForegroundColor White

$healthUrl = "http://127.0.0.1:8000/health"
$dashboardUrl = "http://127.0.0.1:8000/"

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

# 2.1 Check if port 8000 is occupied
$portOccupied = $false
try {
    $tcp = New-Object System.Net.Sockets.TcpClient
    $iar = $tcp.BeginConnect("127.0.0.1", 8000, $null, $null)
    if ($iar.AsyncWaitHandle.WaitOne(600, $false) -and $tcp.Connected) {
        $tcp.EndConnect($iar)
        $portOccupied = $true
    }
    $tcp.Close()
} catch {
    $portOccupied = $false
}

if ($portOccupied) {
    # Check if already running ExamGuard
    if (Test-ExamGuardHealth) {
        Write-Host "      ✓ ExamGuard Vision đang chạy." -ForegroundColor Yellow
        Write-Host ""
        Write-Host "[4/4] Mở bảng điều khiển..." -ForegroundColor White
        Write-Host "      ✓ Mở trình duyệt: $dashboardUrl" -ForegroundColor Green
        Start-Process $dashboardUrl
        Start-Sleep -Seconds 2
        exit 0
    }

    $foreignPid = "Không xác định"
    try {
        $conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($conn -and $conn.OwningProcess) {
            $foreignPid = $conn.OwningProcess
        }
    } catch {}

    Write-Host ""
    Write-Host " [LỖI] Không thể khởi động ExamGuard Vision vì cổng 8000 đang được chương trình khác sử dụng." -ForegroundColor Red
    Write-Host "       PID đang chiếm cổng 8000: $foreignPid" -ForegroundColor Yellow
    Write-Host "       Vui lòng tắt ứng dụng đang sử dụng cổng 8000 trước khi khởi động ExamGuard." -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

Write-Host "      ✓ Cổng 8000 sẵn sàng" -ForegroundColor Green

# [3/4] Start Canonical Runtime
Write-Host ""
Write-Host "[3/4] Khởi động Camera & AI..." -ForegroundColor White
Write-Host "      Đang khởi động..." -ForegroundColor Cyan

# Remove leftover stop signal from previous session
if (Test-Path -LiteralPath $stopSignalFile) {
    Remove-Item -LiteralPath $stopSignalFile -Force -ErrorAction SilentlyContinue
}

# Start the demo runner minimized
$proc = Start-Process -FilePath $pythonExe `
    -ArgumentList "scripts\run_asus_a17_demo.py" `
    -WorkingDirectory $repoRoot `
    -WindowStyle Minimized `
    -PassThru

if (-not $proc -or $proc.HasExited) {
    Write-Host "      [LỖI] Không thể khởi chạy tiến trình Python runtime." -ForegroundColor Red
    exit 1
}

$runtimePid = $proc.Id
Set-Content -LiteralPath $pidFile -Value $runtimePid -Force

$launchInfo = @{
    pid = $runtimePid
    start_time = (Get-Date -Format "o")
    project_root = $repoRoot
    command = "$pythonExe scripts\run_asus_a17_demo.py"
    port = 8000
} | ConvertTo-Json
Set-Content -LiteralPath $launchFile -Value $launchInfo -Force

# Health readiness poll loop
$timeoutSeconds = 60
$startTime = Get-Date
$ready = $false

while (((Get-Date) - $startTime).TotalSeconds -lt $timeoutSeconds) {
    if ($proc.HasExited) {
        break
    }
    if (Test-ExamGuardHealth -TimeoutMs 1000) {
        $ready = $true
        break
    }
    Start-Sleep -Milliseconds 800
}

# [4/4] Open Dashboard or Report Failure
if ($ready) {
    Write-Host ""
    Write-Host "[4/4] Mở bảng điều khiển..." -ForegroundColor White
    Write-Host "      ✓ ExamGuard Vision đã sẵn sàng" -ForegroundColor Green
    Write-Host "      Địa chỉ: $dashboardUrl" -ForegroundColor Cyan
    Start-Process $dashboardUrl
    Start-Sleep -Seconds 2
    exit 0
} else {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Red
    Write-Host " [LỖI] ExamGuard Vision chưa thể khởi động." -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Red
    Write-Host " Vị trí file nhật ký (log):" -ForegroundColor Yellow
    Write-Host "   $runtimeLog" -ForegroundColor White
    Write-Host ""
    if (Test-Path -LiteralPath $runtimeLog) {
        Write-Host " 15 dòng nhật ký gần nhất:" -ForegroundColor DarkGray
        Get-Content -LiteralPath $runtimeLog -Tail 15 -ErrorAction SilentlyContinue | ForEach-Object {
            Write-Host "   $_" -ForegroundColor DarkGray
        }
        Write-Host ""
    }
    Write-Host " Để kiểm tra chẩn đoán chi tiết phần cứng, chạy lệnh sau:" -ForegroundColor Cyan
    Write-Host "   .\.venv\Scripts\python.exe scripts\laptop_preflight.py" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}
