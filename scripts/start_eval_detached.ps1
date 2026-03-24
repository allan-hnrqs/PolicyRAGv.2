param(
    [ValidateSet("eval", "eval-ragas", "eval-pairwise")][string]$Mode = "eval",
    [string]$EvalPath = "",
    [string]$Profile = "baseline",
    [string]$IndexNamespace = "",
    [string]$ControlRun = "",
    [string]$CandidateRun = "",
    [string]$Label = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ActiveIndexNamespace {
    param(
        [Parameter(Mandatory = $true)][string]$RepoRoot
    )

    $pointerPath = Join-Path $RepoRoot "datasets\index\active_index.json"
    if (-not (Test-Path $pointerPath)) {
        throw "No active index pointer found at $pointerPath. Pass -IndexNamespace explicitly or activate a snapshot first."
    }

    $payload = Get-Content $pointerPath -Raw | ConvertFrom-Json
    $namespace = [string]$payload.namespace
    if (-not $namespace.Trim()) {
        throw "Active index pointer at $pointerPath does not contain a namespace value."
    }
    return $namespace
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
$launchId = "{0}_{1}" -f $timestamp, ([guid]::NewGuid().ToString("N").Substring(0, 8))
$safeLabel = if ($Label) { ($Label -replace '[^A-Za-z0-9_-]', '_') } else { $Mode }
$logDir = Join-Path $repoRoot "datasets\runs\background"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stdoutPath = Join-Path $logDir "${launchId}_${safeLabel}.out.log"
$stderrPath = Join-Path $logDir "${launchId}_${safeLabel}.err.log"
$metaPath = Join-Path $logDir "${launchId}_${safeLabel}.meta.json"

$resolvedIndexNamespace = ""
$resolvedIndexNamespaceSource = ""
$cmd = @("-m", "bgrag.cli")

switch ($Mode) {
    "eval" {
        if (-not $EvalPath) {
            throw "-EvalPath is required for -Mode eval."
        }
        $resolvedIndexNamespace = if ($IndexNamespace) { $IndexNamespace } else { Get-ActiveIndexNamespace -RepoRoot $repoRoot }
        $resolvedIndexNamespaceSource = if ($IndexNamespace) { "explicit" } else { "active_index_pointer" }
        $cmd += @("eval", $EvalPath, "--profile", $Profile, "--index-namespace", $resolvedIndexNamespace)
    }
    "eval-ragas" {
        if (-not $EvalPath) {
            throw "-EvalPath is required for -Mode eval-ragas."
        }
        $resolvedIndexNamespace = if ($IndexNamespace) { $IndexNamespace } else { Get-ActiveIndexNamespace -RepoRoot $repoRoot }
        $resolvedIndexNamespaceSource = if ($IndexNamespace) { "explicit" } else { "active_index_pointer" }
        $cmd += @("eval-ragas", $EvalPath, "--profile", $Profile, "--index-namespace", $resolvedIndexNamespace)
    }
    "eval-pairwise" {
        if (-not $ControlRun) {
            throw "-ControlRun is required for -Mode eval-pairwise."
        }
        if (-not $CandidateRun) {
            throw "-CandidateRun is required for -Mode eval-pairwise."
        }
        $cmd += @("eval-pairwise", $ControlRun, $CandidateRun)
    }
}

$meta = @{
    schema_version = 1
    launch_id = $launchId
    launched_at = (Get-Date).ToString("o")
    mode = $Mode
    repo_root = $repoRoot
    eval_path = if ($EvalPath) { $EvalPath } else { $null }
    profile = if ($Mode -like "eval*") { $Profile } else { $null }
    requested_index_namespace = if ($IndexNamespace) { $IndexNamespace } else { $null }
    resolved_index_namespace = if ($resolvedIndexNamespace) { $resolvedIndexNamespace } else { $null }
    resolved_index_namespace_source = if ($resolvedIndexNamespaceSource) { $resolvedIndexNamespaceSource } else { $null }
    control_run = if ($ControlRun) { $ControlRun } else { $null }
    candidate_run = if ($CandidateRun) { $CandidateRun } else { $null }
    command = @("python") + $cmd
    stdout = $stdoutPath
    stderr = $stderrPath
}
$meta | ConvertTo-Json -Depth 6 | Set-Content -Path $metaPath -Encoding utf8

$env:BGRAG_LAUNCH_ID = $launchId
$env:BGRAG_LAUNCH_MANIFEST = $metaPath
try {
    $proc = Start-Process `
        -FilePath "python" `
        -ArgumentList $cmd `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
}
finally {
    Remove-Item Env:\BGRAG_LAUNCH_ID -ErrorAction SilentlyContinue
    Remove-Item Env:\BGRAG_LAUNCH_MANIFEST -ErrorAction SilentlyContinue
}

$meta.pid = $proc.Id
$meta.started_at = (Get-Date).ToString("o")
$meta | ConvertTo-Json -Depth 6 | Set-Content -Path $metaPath -Encoding utf8

Write-Output "Started detached run. PID=$($proc.Id)"
Write-Output "mode:   $Mode"
if ($resolvedIndexNamespace) {
    Write-Output "index:  $resolvedIndexNamespace ($resolvedIndexNamespaceSource)"
}
Write-Output "stdout: $stdoutPath"
Write-Output "stderr: $stderrPath"
Write-Output "meta:   $metaPath"
