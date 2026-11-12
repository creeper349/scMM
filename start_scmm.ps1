# Compatibility entry point. Launch and error handling live in start_scmm.cmd.
& (Join-Path $PSScriptRoot 'start_scmm.cmd') @args
exit $LASTEXITCODE
