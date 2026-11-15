param(
    [Parameter(Mandatory=$true)][string]$Target,
    [switch]$VerifyOnly,
    [switch]$SkipSetup,
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
$targetRoot = [IO.Path]::GetFullPath($Target).TrimEnd('\')
$prefix = $targetRoot + [IO.Path]::DirectorySeparatorChar
if (-not (Test-Path -LiteralPath (Join-Path $targetRoot 'pyproject.toml')) -or
    -not (Test-Path -LiteralPath (Join-Path $targetRoot 'scMM\webapp.py'))) {
    throw 'Extract this patch into the existing SCMM software folder and run its apply_patch CMD file there.'
}
$manifest = Get-Content -Raw -LiteralPath (Join-Path $PSScriptRoot 'manifest.json') | ConvertFrom-Json
$project = [IO.File]::ReadAllText((Join-Path $targetRoot 'pyproject.toml'))
$versionMatch = [regex]::Match($project, '(?m)^version\s*=\s*"([^"]+)"')
$version = $versionMatch.Groups[1].Value
if ($version -ne $manifest.source_version -and $version -ne $manifest.target_version) {
    throw "This patch is for SCMM $($manifest.source_version). Detected version: $version"
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [IO.Compression.ZipFile]::OpenRead((Join-Path $PSScriptRoot 'payload.zip'))
try {
    $entries = @($archive.Entries | Where-Object { $_.Name })
    $names = @($entries | ForEach-Object { $_.FullName })
    if (@(Compare-Object @($manifest.files) $names).Count) { throw 'Patch payload does not match the manifest.' }
    foreach ($entry in $entries) {
        $destination = [IO.Path]::GetFullPath((Join-Path $targetRoot $entry.FullName))
        if (-not $destination.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Invalid patch path: $($entry.FullName)"
        }
    }
} finally { $archive.Dispose() }
if ($VerifyOnly) {
    Write-Host "[SCMM] Patch verified: $version -> $($manifest.target_version). $($manifest.files.Count) files."
    exit 0
}
$backup = Join-Path $targetRoot ('.scmm-update-backups\' + [DateTime]::Now.ToString('yyyyMMdd-HHmmss-fff'))
$stage = Join-Path $backup 'new-version'
New-Item -ItemType Directory -Path $stage -Force | Out-Null
[IO.Compression.ZipFile]::ExtractToDirectory((Join-Path $PSScriptRoot 'payload.zip'), $stage)
$rootFiles = @($manifest.files | Where-Object { -not $_.StartsWith('scMM/') })
$previousFiles = @()
$writtenFiles = @()
$packageMoved = $false
try {
    Move-Item -LiteralPath (Join-Path $targetRoot 'scMM') -Destination (Join-Path $backup 'scMM')
    $packageMoved = $true
    Move-Item -LiteralPath (Join-Path $stage 'scMM') -Destination (Join-Path $targetRoot 'scMM')
    foreach ($relative in $rootFiles) {
        $destination = Join-Path $targetRoot $relative
        if (Test-Path -LiteralPath $destination) {
            $saved = Join-Path (Join-Path $backup 'previous') $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force | Out-Null
            Copy-Item -LiteralPath $destination -Destination $saved
            $previousFiles += $relative
        }
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        $writtenFiles += $relative
        Copy-Item -LiteralPath (Join-Path $stage $relative) -Destination $destination -Force
    }
} catch {
    if ($packageMoved) {
        $current = Join-Path $targetRoot 'scMM'
        if (Test-Path -LiteralPath $current) { Move-Item -LiteralPath $current -Destination (Join-Path $backup 'failed-scMM') }
        Move-Item -LiteralPath (Join-Path $backup 'scMM') -Destination $current
    }
    foreach ($relative in $writtenFiles) {
        $destination = Join-Path $targetRoot $relative
        if ($previousFiles -contains $relative) {
            Copy-Item -LiteralPath (Join-Path (Join-Path $backup 'previous') $relative) -Destination $destination -Force
        } elseif (Test-Path -LiteralPath $destination) {
            $failed = Join-Path (Join-Path $backup 'failed-root') $relative
            New-Item -ItemType Directory -Path (Split-Path -Parent $failed) -Force | Out-Null
            Move-Item -LiteralPath $destination -Destination $failed
        }
    }
    throw
}
Write-Host "[SCMM] Program files updated. Previous program and settings: $backup" -ForegroundColor Green
if ($SkipSetup) {
    Write-Host '[SCMM] Dependency setup skipped. Run setup_scmm.cmd before launching.' -ForegroundColor Yellow
    exit 0
}
$setupArguments = @('--no-pause')
if ($PythonPath) { $setupArguments += @('--python', $PythonPath) }
& (Join-Path $targetRoot 'setup_scmm.cmd') @setupArguments
if ($LASTEXITCODE -ne 0) {
    throw 'Program files are updated; environment setup needs attention. Read the error above, then rerun setup_scmm.cmd.'
}
Write-Host '[SCMM] Update complete. Start with start_scmm.cmd.' -ForegroundColor Green
exit 0
