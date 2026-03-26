param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 4173
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$envFile = Join-Path $repoRoot ".env"
$activeIndexPointer = Join-Path $repoRoot "datasets\index\active_index.json"

function Resolve-BootstrapPython {
    if (Test-Path $venvPython) {
        return $venvPython
    }

    $candidates = @(
        @{
            Name = "py -3.11"
            Command = { & py -3.11 -c "import sys; print(sys.executable)" }
        },
        @{
            Name = "python"
            Command = { & python -c "import sys; print(sys.executable)" }
        }
    )

    foreach ($candidate in $candidates) {
        try {
            $resolved = & $candidate.Command
            if ($LASTEXITCODE -eq 0 -and $resolved) {
                return $resolved.Trim()
            }
        } catch {
        }
    }

    throw "No usable Python 3.11 interpreter was found. Install Python 3.11, then rerun this script."
}

function Ensure-Venv {
    $bootstrapPython = Resolve-BootstrapPython
    if (-not (Test-Path $venvPython)) {
        Write-Output "Creating repo-local virtual environment at $venvDir"
        & $bootstrapPython -m venv $venvDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create the repo-local virtual environment."
        }
    }
}

function Ensure-EditableInstall {
    $packageImportOk = $false
    try {
        & $venvPython -c "import bgrag" | Out-Null
        $packageImportOk = ($LASTEXITCODE -eq 0)
    } catch {
        $packageImportOk = $false
    }

    if (-not $packageImportOk) {
        Write-Output "Installing repo dependencies into .venv"
        Push-Location $repoRoot
        try {
            & $venvPython -m pip install -e '.[dev]'
            if ($LASTEXITCODE -ne 0) {
                throw "pip install failed"
            }
        } finally {
            Pop-Location
        }
    }
}

function Ensure-EnvFile {
    if (-not (Test-Path $envFile)) {
        throw "Missing .env file at $envFile. Add COHERE_API_KEY before starting the live demo."
    }
    $envContents = Get-Content $envFile -Raw
    if ($envContents -notmatch "(?m)^COHERE_API_KEY=\S+") {
        throw "COHERE_API_KEY is missing or empty in $envFile."
    }
}

function Ensure-Elasticsearch {
    try {
        $health = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:9200" -TimeoutSec 2
        if ($health.StatusCode -ge 200 -and $health.StatusCode -lt 500) {
            Write-Output "Elasticsearch already responding at http://127.0.0.1:9200"
            return
        }
    } catch {
    }

    Write-Output "Starting local Elasticsearch"
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "start_elasticsearch.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start Elasticsearch."
    }
}

function Test-ActiveIndexReady {
    if (-not (Test-Path $activeIndexPointer)) {
        return $false
    }

    try {
        $payload = Get-Content $activeIndexPointer | ConvertFrom-Json
        $namespace = [string]$payload.namespace
    } catch {
        return $false
    }
    if (-not $namespace) {
        return $false
    }

    $manifestPath = Join-Path $repoRoot ("datasets\index\" + $namespace + "\index_manifest.json")
    $embeddingsPath = Join-Path $repoRoot ("datasets\index\" + $namespace + "\chunk_embeddings.json")
    return (Test-Path $manifestPath) -and (Test-Path $embeddingsPath)
}

function Ensure-BaselineIndex {
    if (Test-ActiveIndexReady) {
        Write-Output "Active baseline index already present"
        return
    }

    Write-Output "Building baseline index for the live demo"
    Push-Location $repoRoot
    try {
        & $venvPython -c "from bgrag.demo_server import build_demo_settings; from bgrag.pipeline import run_build_index; import json; settings = build_demo_settings(); print(json.dumps(run_build_index(settings, 'baseline'), indent=2))"
        if ($LASTEXITCODE -ne 0) {
            throw "baseline index build failed"
        }
    } finally {
        Pop-Location
    }
}

Ensure-Venv
Ensure-EditableInstall
Ensure-EnvFile
Ensure-Elasticsearch
Ensure-BaselineIndex

Write-Output "Launching PolicyAI live demo at http://$HostAddress`:$Port"
Push-Location $repoRoot
try {
    & $venvPython -m bgrag.demo_server --host $HostAddress --port $Port
} finally {
    Pop-Location
}
