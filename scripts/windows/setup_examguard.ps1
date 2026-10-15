param([ValidateSet('Auto','CPU','CUDA128')][string]$Backend = 'Auto')
$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $repoRoot
try {
    $pythonExe = Join-Path $repoRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        & py -3.13 -m venv .venv
        if ($LASTEXITCODE) { throw 'Install standard Python 3.13 x64 (not the free-threaded build), then retry.' }
    }
    & git lfs install --local
    & git lfs pull
    if ($LASTEXITCODE) { throw 'Model download failed. Check Git LFS/network connection.' }
    if ($Backend -eq 'Auto') { $Backend = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { 'CUDA128' } else { 'CPU' } }
    $torchIndex = if ($Backend -eq 'CPU') { 'https://download.pytorch.org/whl/cpu' } else { 'https://download.pytorch.org/whl/cu128' }
    & $pythonExe -m pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url $torchIndex
    if ($LASTEXITCODE) { throw 'PyTorch installation failed.' }
    & $pythonExe -m pip install -r requirements-runtime.txt
    if ($LASTEXITCODE) { throw 'Dependency installation failed.' }
    & $pythonExe -m pip check
    if ($LASTEXITCODE) { throw 'Dependency compatibility check failed.' }
    & "$PSScriptRoot\install_desktop_shortcuts.ps1"
    & "$PSScriptRoot\start_examguard.ps1" -Reconfigure
} catch { Write-Host "SETUP REQUIRED: $_" -ForegroundColor Red; exit 1 }
