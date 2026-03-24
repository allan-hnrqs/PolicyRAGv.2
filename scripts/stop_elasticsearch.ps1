Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pidPath = Join-Path $repoRoot ".cache\elasticsearch\elasticsearch.pid"

if (-not (Test-Path $pidPath)) {
    Write-Output "No Elasticsearch PID file found."
    exit 0
}

$pidValue = (Get-Content -Path $pidPath -Raw).Trim()
if (-not $pidValue) {
    Remove-Item -Path $pidPath -Force
    Write-Output "PID file was empty and has been removed."
    exit 0
}

$process = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
if ($null -eq $process) {
    Remove-Item -Path $pidPath -Force
    Write-Output "Elasticsearch process $pidValue is not running."
    exit 0
}

Stop-Process -Id $process.Id -Force
Remove-Item -Path $pidPath -Force
Write-Output "Stopped Elasticsearch process $($process.Id)."
