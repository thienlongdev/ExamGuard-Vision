param([switch]$NoBrowser, [switch]$Reconfigure, [int]$Port = 8765)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot
$runtimeDir = Join-Path $repoRoot '.runtime'
New-Item -ItemType Directory -Force -Path (Join-Path $runtimeDir 'logs') | Out-Null
$pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
$launchFile = Join-Path $runtimeDir 'examguard-launch.json'
$stopSignalFile = Join-Path $runtimeDir 'stop.signal'
$dashboardUrl = "http://127.0.0.1:$Port/"
$mutexName = 'Local\ExamGuardStart-' + ([System.BitConverter]::ToString([System.Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($repoRoot)))).Replace('-','')
$mutex = New-Object Threading.Mutex($false, $mutexName)
$locked = $false
function Get-Ready {
    try {
        $ready = Invoke-RestMethod "http://127.0.0.1:$Port/ready" -TimeoutSec 2
        $launch = Get-Content -LiteralPath $launchFile -Raw | ConvertFrom-Json
        if ($ready.application -eq 'ExamGuard Vision' -and $ready.models_verified -and
            $ready.state -in @('READY','DEGRADED') -and $ready.instance -eq $launch.instance -and $ready.pid -eq $launch.pid) { return $ready }
    } catch {}
    return $null
}
function Open-Dashboard {
    if (-not $NoBrowser -and $env:EXAMGUARD_LAUNCHER_TEST_MODE -ne '1') { Start-Process $dashboardUrl }
}
try {
    $locked = $mutex.WaitOne(0)
    if (-not $locked) { Write-Host 'ExamGuard is already initializing. Please wait.'; exit 0 }
    Write-Host 'ExamGuard Vision - Auto runtime'
    Write-Host 'First use: Setup ExamGuard Vision.bat. Change source: Configure ExamGuard Source.bat.'
    Write-Host 'Create admin at /setup; sign in; start a monitoring session to enable AI.'
    Write-Host 'Stop: Stop ExamGuard Vision.bat. Missing models: git lfs pull. Diagnostics: .runtime/logs.'
    if (Get-Ready) { Write-Host 'ExamGuard Vision đang chạy.'; Open-Dashboard; exit 0 }
    $tcp = New-Object Net.Sockets.TcpClient
    try { $occupied = $tcp.ConnectAsync('127.0.0.1',$Port).Wait(500) -and $tcp.Connected } catch { $occupied = $false } finally { $tcp.Dispose() }
    if ($occupied) { Write-Host "cổng $Port đang được chương trình khác sử dụng. Existing process was left untouched."; exit 1 }
    if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'SETUP REQUIRED: Run Setup ExamGuard Vision.bat once. Python .venv is missing.' }
    Write-Host 'Checking hardware, camera and model execution. First setup can take longer.'
    Remove-Item -LiteralPath $stopSignalFile -Force -ErrorAction SilentlyContinue
    $arguments = @('scripts\windows\launch_examguard_detached.py', '--port', $Port)
    if ($Reconfigure) { $arguments += '--reconfigure' }
    $pidText = & $pythonExe @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Detached server launch failed.' }
    $launchedPid = [int]($pidText | Select-Object -Last 1)
    $proc = Get-Process -Id $launchedPid -ErrorAction Stop
    $deadline = (Get-Date).AddSeconds(180)
    do {
        $ready = Get-Ready
        if ($ready) {
            Write-Host "ExamGuard Vision đã sẵn sàng. Device: $($ready.device); state: $($ready.state)"
            if ($ready.reason) { Write-Host $ready.reason }
            Write-Host 'Sign in and start a monitoring session to enable camera + AI.'
            Open-Dashboard
            exit 0
        }
        if ($proc.HasExited) { break }
        Start-Sleep -Milliseconds 700
    } while ((Get-Date) -lt $deadline)
    Set-Content -LiteralPath $stopSignalFile -Value 'STOP' -Encoding UTF8
    $statusPath = Join-Path $runtimeDir 'status.json'
    if (Test-Path -LiteralPath $statusPath) { Write-Host (Get-Content -LiteralPath $statusPath -Raw) }
    Get-Content -LiteralPath (Join-Path $runtimeDir 'logs\startup.stderr.log') -Tail 12 -ErrorAction SilentlyContinue | Write-Host
    throw 'Startup blocked. See .runtime\logs\examguard-runtime.log. No automatic dependency installation was attempted.'
} catch { Write-Host $_ -ForegroundColor Red; exit 1 }
finally { if ($locked) { $mutex.ReleaseMutex() }; $mutex.Dispose() }
