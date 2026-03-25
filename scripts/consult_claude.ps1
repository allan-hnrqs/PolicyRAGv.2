param(
    [Parameter(Position = 0)]
    [string]$Prompt,
    [string]$PromptFile,
    [string]$SessionName = "codex-peer",
    [double]$MaxBudgetUsd = 2.0,
    [switch]$RawJson
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$localDir = Join-Path $repoRoot ".claude/session_local"
$transcriptDir = Join-Path $localDir "transcripts"
$sessionIdPath = Join-Path $localDir ("{0}-session-id.txt" -f $SessionName)

New-Item -ItemType Directory -Force -Path $transcriptDir | Out-Null

if ($PromptFile) {
    $Prompt = Get-Content -Path $PromptFile -Raw
}

if ([string]::IsNullOrWhiteSpace($Prompt)) {
    throw "Provide -Prompt or -PromptFile."
}

$claudeArgs = @(
    "-p",
    "--model", "claude-opus-4-6",
    "--effort", "max",
    "--add-dir", ".",
    "--output-format", "json",
    "--max-budget-usd", $MaxBudgetUsd
)

if (Test-Path $sessionIdPath) {
    $sessionId = (Get-Content -Path $sessionIdPath -Raw).Trim()
    $claudeArgs += @("--resume", $sessionId)
} else {
    $sessionId = [guid]::NewGuid().ToString()
    Set-Content -Path $sessionIdPath -Value $sessionId
    $claudeArgs += @("--session-id", $sessionId)
}

$resultJson = $Prompt | & claude @claudeArgs
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$transcriptPath = Join-Path $transcriptDir ("claude_consult_{0}_{1}.json" -f $SessionName, $stamp)
Set-Content -Path $transcriptPath -Value $resultJson

if ($RawJson) {
    Write-Output $resultJson
    return
}

$result = $resultJson | ConvertFrom-Json
if ($result.session_id) {
    Set-Content -Path $sessionIdPath -Value $result.session_id
}
Write-Output $result.result
Write-Output ""
Write-Output ("session_id={0}" -f $result.session_id)
Write-Output ("transcript={0}" -f $transcriptPath)
