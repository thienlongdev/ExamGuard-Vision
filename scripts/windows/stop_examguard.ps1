[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$runtimeDir = Join-Path $repoRoot '.runtime'
$pidFile = Join-Path $runtimeDir 'examguard.pid'
$launchFile = Join-Path $runtimeDir 'examguard-launch.json'
$stopSignalFile = Join-Path $runtimeDir 'stop.signal'
$targetPid = 0
try { $targetPid = [int](Get-Content -LiteralPath $pidFile -Raw).Trim() } catch {}
$proc = if ($targetPid) { Get-CimInstance Win32_Process -Filter "ProcessId = $targetPid" -ErrorAction SilentlyContinue } else { $null }
if ($proc) {
    try { $launch = Get-Content -LiteralPath $launchFile -Raw | ConvertFrom-Json } catch { Write-Host 'Missing ownership metadata; refusing to stop process.'; exit 1 }
    $expectedScript = Join-Path $repoRoot 'scripts\run_examguard.py'
    $expectedPython = Join-Path $repoRoot '.venv\Scripts\python.exe'
    $cmd = $proc.CommandLine
    $created = ([DateTimeOffset]$proc.CreationDate).ToUnixTimeMilliseconds() / 1000.0
    if ($launch.project_root -ne $repoRoot -or $launch.pid -ne $targetPid -or
        [Math]::Abs($created - $launch.create_time) -gt 2 -or
        ($cmd -notlike '*scripts\run_examguard.py*' -and $cmd -notlike "*$expectedScript*") -or
        $cmd -notlike "*$expectedPython*") {
        Write-Host "PID $targetPid ownership mismatch; process was left untouched."
        exit 1
    }
    Set-Content -LiteralPath $stopSignalFile -Value 'STOP' -Encoding UTF8
    $deadline = (Get-Date).AddSeconds(60)
    do {
        Start-Sleep -Milliseconds 500
        $alive = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
    } while ($alive -and (Get-Date) -lt $deadline)
    if ($alive) { Write-Host 'Graceful stop timed out. Process left intact to protect evidence; see runtime log.'; exit 1 }
} else {
    $starting = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*$(Join-Path $repoRoot '.venv\Scripts\python.exe')*" -and $_.CommandLine -like '*scripts\run_examguard.py*'
    }
    if ($starting) {
        Set-Content -LiteralPath $stopSignalFile -Value 'STOP' -Encoding UTF8
        Write-Host 'Stop requested during initialization. Resources will be released after the current operation.'
        exit 0
    }
}
Remove-Item -LiteralPath $pidFile,$launchFile,$stopSignalFile -Force -ErrorAction SilentlyContinue
Write-Host 'Đã dừng ExamGuard Vision an toàn.'
