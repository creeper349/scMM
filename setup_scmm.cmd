@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title SCMM environment setup
set "SCMM_SETUP_FILE=%~f0"
set "SCMM_SETUP_PREFIX=%~dp0.scmm-env"
set "SCMM_SETUP_CHECK=0"
set "SCMM_SETUP_NOPAUSE=0"
set "SCMM_SETUP_RAW_ONLY=0"
set "SCMM_SETUP_ACCEPT_RAW=0"
set "SCMM_SETUP_PYTHON="
:arguments
if "%~1"=="" goto run_setup
if /i "%~1"=="--check" set "SCMM_SETUP_CHECK=1"
if /i "%~1"=="--no-pause" set "SCMM_SETUP_NOPAUSE=1"
if /i "%~1"=="--raw-only" set "SCMM_SETUP_RAW_ONLY=1"
if /i "%~1"=="--accept-thermo-license" set "SCMM_SETUP_ACCEPT_RAW=1"
if /i "%~1"=="--prefix" (
  set "SCMM_SETUP_PREFIX=%~2"
  shift
)
if /i "%~1"=="--python" (
  set "SCMM_SETUP_PYTHON=%~2"
  shift
)
shift
goto arguments
:run_setup
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; try { $text=[IO.File]::ReadAllText($env:SCMM_SETUP_FILE); $marker='# SCMM_INSTALLER_BODY'; & ([scriptblock]::Create($text.Substring($text.LastIndexOf($marker)+$marker.Length))); } catch { Write-Host ('[SCMM] ERROR: '+$_) -ForegroundColor Red; Write-Host $_.ScriptStackTrace; exit 1 }"
set "SCMM_SETUP_EXIT=%errorlevel%"
echo.
if not "%SCMM_SETUP_EXIT%"=="0" echo [SCMM] Setup failed with exit code %SCMM_SETUP_EXIT%. Review the error above.
if "%SCMM_SETUP_NOPAUSE%"=="0" pause
exit /b %SCMM_SETUP_EXIT%

# SCMM_INSTALLER_BODY
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $env:SCMM_SETUP_FILE
$prefix = [IO.Path]::GetFullPath($env:SCMM_SETUP_PREFIX)
$checkOnly = $env:SCMM_SETUP_CHECK -eq '1'
$rawOnly = $env:SCMM_SETUP_RAW_ONLY -eq '1'
Set-Location -LiteralPath $projectRoot

function Find-Conda {
    $command = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $paths = @($env:CONDA_EXE,
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\SCMM\Miniconda3\Scripts\conda.exe",
        'C:\ProgramData\Miniconda3\Scripts\conda.exe')
    foreach ($drive in Get-PSDrive -PSProvider FileSystem) {
        $paths += Join-Path $drive.Root 'Miniconda3\Scripts\conda.exe'
    }
    foreach ($path in $paths) {
        if ($path -and (Test-Path -LiteralPath $path)) { return $path }
    }
}

function Find-Python {
    $candidates = @($env:SCMM_SETUP_PYTHON)
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $ErrorActionPreference = 'Continue'
        $candidate = & $launcher.Source -3.11 -c 'import sys; print(sys.executable)' 2>$null
        if ($LASTEXITCODE -eq 0) { $candidates += $candidate }
        $ErrorActionPreference = 'Stop'
    }
    foreach ($candidate in $candidates) {
        if (-not $candidate) { continue }
        $ErrorActionPreference = 'Continue'
        $path = & $candidate -c "import sys,struct; print(sys.executable) if sys.version_info[:2] == (3,11) and struct.calcsize('P') == 8 else sys.exit(1)" 2>$null
        $success = $LASTEXITCODE -eq 0
        $ErrorActionPreference = 'Stop'
        if ($success) { return $path }
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'pyproject.toml'))) {
    throw 'pyproject.toml is missing. Keep the complete SCMM project next to setup_scmm.cmd.'
}
$python = @((Join-Path $prefix 'python.exe'), (Join-Path $prefix 'Scripts\python.exe')) |
    Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $python) {
    if ($checkOnly -or $rawOnly) { throw "The environment is missing: $prefix. Run setup_scmm.cmd first." }
    if (Test-Path -LiteralPath $prefix) { throw "The existing directory has no usable Python: $prefix. Rename it as a backup before creating a new environment." }
    $conda = if ($env:SCMM_SETUP_PYTHON) { $null } else { Find-Conda }
    $basePython = if (-not $conda) { Find-Python } else { $null }
    if (-not $conda -and -not $basePython) {
        Write-Host '[SCMM] Install 64-bit Python 3.11 or Miniconda, then run setup_scmm.cmd again.' -ForegroundColor Yellow
        Write-Host 'Python: https://www.python.org/downloads/windows/'
        Write-Host 'Miniconda: https://docs.conda.io/projects/miniconda/en/latest/'
        Write-Host 'If Miniconda requests acceptance of Anaconda channel terms, open Miniconda Prompt and run:'
        foreach ($channel in @('main', 'r', 'msys2')) {
            Write-Host "conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/$channel"
        }
        throw 'Python/Conda is unavailable. Complete the installation above, reopen the terminal and retry.'
    }
    Write-Host "[SCMM] Creating environment: $prefix" -ForegroundColor Cyan
    if ($conda) {
        & $conda create --prefix $prefix --override-channels --channel conda-forge python=3.11 pip -y
        if ($LASTEXITCODE -ne 0) { throw "Conda environment creation failed (exit $LASTEXITCODE)." }
        $python = Join-Path $prefix 'python.exe'
    } else {
        & $basePython -m venv $prefix
        if ($LASTEXITCODE -ne 0) { throw "Python environment creation failed (exit $LASTEXITCODE)." }
        $python = Join-Path $prefix 'Scripts\python.exe'
    }
}
& $python -c "import sys,struct; print(sys.version); sys.exit(0 if sys.version_info[:2] == (3,11) and struct.calcsize('P') == 8 else 1)"
if ($LASTEXITCODE -ne 0) { throw 'SCMM requires 64-bit Python 3.11. Check the environment path above.' }

# Read the dependency list once; installed distributions are left in place.
$inspect = @'
import importlib.metadata as metadata, json, pathlib, re, sys, sysconfig, tomllib
project = tomllib.loads((pathlib.Path(sys.argv[1]) / 'pyproject.toml').read_text(encoding='utf-8'))['project']
required = project['dependencies'] + project.get('optional-dependencies', {}).get('web', []) + project.get('optional-dependencies', {}).get('ml', [])
normalize = lambda name: re.sub(r'[-_.]+', '-', name).lower()
installed = {normalize(d.metadata['Name']): d for d in metadata.distributions(path=[sysconfig.get_path('purelib')]) if d.metadata['Name']}
missing = [req for req in required if normalize(re.split(r'[<>=!~\[; ]', req, maxsplit=1)[0]) not in installed]
print(json.dumps({'missing': missing, 'scmm': 'scmm' in installed, 'pythonnet': 'pythonnet' in installed,
                  'scmm_version': installed['scmm'].version if 'scmm' in installed else None,
                  'project_version': project['version']}))
'@
$inspection = & $python -c $inspect $projectRoot
if ($LASTEXITCODE -ne 0) { throw 'Could not inspect Python dependencies.' }
$packages = $inspection | ConvertFrom-Json
$missing = @($packages.missing)
$dllDir = Join-Path $prefix 'Library\bin\ThermoRawFileReader'
$dlls = @('ThermoFisher.CommonCore.Data.dll', 'ThermoFisher.CommonCore.RawFileReader.dll')
$missingDlls = @($dlls | Where-Object { -not (Test-Path -LiteralPath (Join-Path $dllDir $_)) })
$rawReady = $packages.pythonnet -and $missingDlls.Count -eq 0
Write-Host "[SCMM] Python: $python"
Write-Host "[SCMM] Missing required packages: $($missing.Count)"
Write-Host "[SCMM] Optional Thermo RAW reader ready: $rawReady"
if ($checkOnly) {
    if ($missing.Count -gt 0 -or -not $packages.scmm) { throw 'Required packages are missing. Run setup_scmm.cmd to install them.' }
    Write-Host '[SCMM] Check complete; no packages were installed.' -ForegroundColor Green
    exit 0
}

$installRaw = $false
if (-not $rawReady) {
    Write-Host 'Optional Thermo RAW direct reader license:'
    Write-Host 'https://github.com/thermofisherlsms/RawFileReader/blob/main/License.doc'
    $installRaw = $env:SCMM_SETUP_ACCEPT_RAW -eq '1'
    if (-not $installRaw) { $installRaw = (Read-Host 'Install the RAW reader and accept the license? [y/N]') -match '^[Yy]' }
}
if (-not $rawOnly -and $missing.Count -gt 0) {
    Write-Host "[SCMM] Installing missing packages: $($missing -join ', ')" -ForegroundColor Cyan
    & $python -m pip install --only-binary=:all: --no-binary=matplotlib-venn,upsetplot @missing
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed (exit $LASTEXITCODE). The pip error is shown above." }
}
if (-not $rawOnly -and (-not $packages.scmm -or $packages.scmm_version -ne $packages.project_version)) {
    & $python -m pip install --no-deps --editable $projectRoot
    if ($LASTEXITCODE -ne 0) { throw "SCMM installation failed (exit $LASTEXITCODE)." }
}
if ($installRaw) {
    if (-not $packages.pythonnet) {
        & $python -m pip install 'pythonnet>=3.0,<4.0'
        if ($LASTEXITCODE -ne 0) { throw "pythonnet installation failed (exit $LASTEXITCODE)." }
    }
    New-Item -ItemType Directory -Path $dllDir -Force | Out-Null
    foreach ($dll in $missingDlls) {
        Write-Host "[SCMM] Downloading $dll"
        $destination = Join-Path $dllDir $dll
        $temporary = "$destination.download"
        Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/thermofisherlsms/RawFileReader/main/Libs/Net471/$dll" -OutFile $temporary
        Move-Item -LiteralPath $temporary -Destination $destination
    }
    & $python -c "from scMM.file.thermo_raw import _load_api; _load_api(); print('Thermo RAW reader ready')"
    if ($LASTEXITCODE -ne 0) { throw 'Thermo RAW reader could not load. Review the traceback above.' }
}
& $python -c "import scMM,streamlit; print('SCMM web environment ready')"
if ($LASTEXITCODE -ne 0) { throw 'Final import check failed. Review the Python traceback above.' }
Write-Host '[SCMM] Setup completed. Start with start_scmm.cmd.' -ForegroundColor Green
exit 0
