param([switch]$AcceptThermoLicense)
# Compatibility entry point for the named scmm environment.
$ErrorActionPreference = 'Stop'
$conda = Get-Command conda.exe -ErrorAction Stop
$environments = & $conda.Source env list --json | ConvertFrom-Json
$prefix = $environments.envs | Where-Object { (Split-Path $_ -Leaf) -eq 'scmm' } | Select-Object -First 1
if (-not $prefix) { throw "Conda environment 'scmm' was not found." }
$setupArgs = @('--prefix', $prefix, '--raw-only')
if ($AcceptThermoLicense) { $setupArgs += '--accept-thermo-license' }
& (Join-Path $PSScriptRoot 'setup_scmm.cmd') @setupArgs
exit $LASTEXITCODE
