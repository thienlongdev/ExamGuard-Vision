# Requires Windows 11 x64, Git + Git LFS, standard Python 3.13 x64 with py.
# Install without opening hardware or changing shortcuts: -Backend CPU -NoLaunch
param([ValidateSet('Auto','CPU','CUDA128')][string]$Backend = 'Auto', [switch]$NoLaunch)
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot
try {
    Write-Host 'ExamGuard Vision | Windows 11 x64 | Python 3.13 x64 + Git LFS'
    Write-Host 'Setup installs dependencies and verifies 8 checkpoints, including optional pose.'
    Write-Host 'First run: http://127.0.0.1:8765/setup -> create admin (no default password).'
    Write-Host 'Configure ExamGuard Source.bat: USB webcam, video file or RTSP. Stop ExamGuard Vision.bat: safe shutdown.'
    Write-Host 'CUDA driver/install trouble: retry with -Backend CPU. CPU inference is slower.'
    Write-Host 'Allow Windows camera access; close other camera apps. Multiple webcams require FFmpeg in PATH.'
    Write-Host 'Diagnostics: .runtime/logs. AI events require human review; body pose provides conservative 2D cues.'
    $pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        & py -3.13 -m venv .venv
        if ($LASTEXITCODE) { throw 'Install standard Python 3.13 x64 (not the free-threaded build), then retry.' }
    }
    if (Test-Path -LiteralPath (Join-Path $repoRoot '.git')) {
        & git lfs install --local
        if ($LASTEXITCODE) { throw 'Install Git LFS, reopen terminal and retry.' }
        & git lfs pull
        if ($LASTEXITCODE) { throw 'Model download failed. Check Git LFS/network connection.' }
    } else {
        Write-Host 'Archive detected: ordinary GitHub ZIP may contain LFS pointers; all real weights are required.'
    }
    & $pythonExe tools/validation/verify_checkpoint_hashes.py
    if ($LASTEXITCODE) { throw 'Missing/invalid model: use Git clone + git lfs pull, then retry.' }
    if ($Backend -eq 'Auto') { $Backend = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { 'CUDA128' } else { 'CPU' } }
    $torchIndex = if ($Backend -eq 'CPU') { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
    & $pythonExe -m pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url $torchIndex
    if ($LASTEXITCODE) { throw 'PyTorch installation failed.' }
    & $pythonExe -m pip install -r requirements-runtime.txt
    if ($LASTEXITCODE) { throw 'Dependency installation failed.' }
    & $pythonExe -m pip check
    if ($LASTEXITCODE) { throw 'Dependency compatibility check failed.' }
    if (-not $NoLaunch) {
        & "$PSScriptRoot\install_desktop_shortcuts.ps1"
        & "$PSScriptRoot\start_examguard.ps1" -Reconfigure
        if ($LASTEXITCODE) { throw 'Startup failed. Check .runtime/logs and camera configuration.' }
    } else {
        Write-Host 'Dependencies and checkpoints verified. No camera opened or Desktop shortcuts changed.'
        Write-Host 'Next: Start ExamGuard Vision.bat; create admin; sign in; start a monitoring session.'
    }
} catch { Write-Host "SETUP REQUIRED: $_" -ForegroundColor Red; exit 1 }
