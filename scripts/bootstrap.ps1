Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot\..
try {
    pip install -e .[dev]
    bgrag inspect
} finally {
    Pop-Location
}
