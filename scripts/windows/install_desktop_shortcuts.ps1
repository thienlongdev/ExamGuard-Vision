# ==============================================================================
# ExamGuard Vision - Desktop Shortcut Installer
# Dynamically creates Desktop shortcuts for starting and stopping ExamGuard Vision.
# Does not require administrator privileges.
# ==============================================================================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path "$scriptDir\..\..").Path
Set-Location -LiteralPath $repoRoot

$desktopPath = [Environment]::GetFolderPath("Desktop")
if (-not (Test-Path -LiteralPath $desktopPath)) {
    $desktopPath = Join-Path $env:USERPROFILE "Desktop"
}

Clear-Host
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   CÀI ĐẶT SHORTCUT EXAMGUARD VISION        " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Thư mục dự án:  $repoRoot" -ForegroundColor DarkGray
Write-Host "Thư mục Desktop: $desktopPath" -ForegroundColor DarkGray
Write-Host ""

try {
    $wshShell = New-Object -ComObject WScript.Shell

    # 1. Shortcut: Start ExamGuard Vision
    $startBat = Join-Path $repoRoot "Start ExamGuard Vision.bat"
    $startLnkPath = Join-Path $desktopPath "ExamGuard Vision.lnk"
    $startLnk = $wshShell.CreateShortcut($startLnkPath)
    $startLnk.TargetPath = $startBat
    $startLnk.WorkingDirectory = $repoRoot
    $startLnk.Description = "ExamGuard Vision - Khởi động giám sát phòng thi"
    $startLnk.Save()

    # 2. Shortcut: Stop ExamGuard Vision
    $stopBat = Join-Path $repoRoot "Stop ExamGuard Vision.bat"
    $stopLnkPath = Join-Path $desktopPath "Stop ExamGuard Vision.lnk"
    $stopLnk = $wshShell.CreateShortcut($stopLnkPath)
    $stopLnk.TargetPath = $stopBat
    $stopLnk.WorkingDirectory = $repoRoot
    $stopLnk.Description = "ExamGuard Vision - Dừng hệ thống an toàn"
    $stopLnk.Save()

    Write-Host "✓ Đã tạo thành công 2 lối tắt trên Desktop:" -ForegroundColor Green
    Write-Host "   1. ExamGuard Vision          (Khởi động hệ thống)" -ForegroundColor Cyan
    Write-Host "   2. Stop ExamGuard Vision     (Dừng an toàn)" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Từ bây giờ, bạn chỉ cần double-click biểu tượng ngoài Desktop để sử dụng." -ForegroundColor White
    Write-Host ""
} catch {
    Write-Host "[LỖI] Không thể tạo shortcut: $_" -ForegroundColor Red
    exit 1
}
