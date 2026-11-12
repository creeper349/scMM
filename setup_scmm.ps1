# Compatibility entry point. Installation is implemented in setup_scmm.cmd.
& (Join-Path $PSScriptRoot 'setup_scmm.cmd') @args
exit $LASTEXITCODE
