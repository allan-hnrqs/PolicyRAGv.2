param(
    [string]$Version = "9.3.1",
    [string]$ElasticUrl = "http://127.0.0.1:9200",
    [int]$HeapMb = 1024
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$cacheRoot = Join-Path $repoRoot ".cache\elasticsearch"
$installRoot = Join-Path $cacheRoot "install"
$dataRoot = Join-Path $cacheRoot "data"
$logsRoot = Join-Path $cacheRoot "logs"
$downloadUrl = "https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-$Version-windows-x86_64.zip"
$zipPath = Join-Path $cacheRoot "elasticsearch-$Version-windows-x86_64.zip"
$esHome = Join-Path $installRoot "elasticsearch-$Version"
$pidPath = Join-Path $cacheRoot "elasticsearch.pid"

function Test-ZipArchive {
    param([string]$Path)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    try {
        $zip = [System.IO.Compression.ZipFile]::OpenRead($Path)
        $zip.Dispose()
        return $true
    } catch {
        return $false
    }
}

foreach ($path in @($cacheRoot, $installRoot, $dataRoot, $logsRoot)) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

try {
    $health = Invoke-WebRequest -UseBasicParsing -Uri $ElasticUrl -TimeoutSec 2
    if ($health.StatusCode -ge 200 -and $health.StatusCode -lt 500) {
        Write-Output "Elasticsearch already responding at $ElasticUrl"
        exit 0
    }
} catch {
}

if (-not (Test-Path $esHome)) {
    if ((Test-Path $zipPath) -and -not (Test-ZipArchive -Path $zipPath)) {
        Write-Output "Removing incomplete Elasticsearch archive at $zipPath"
        Remove-Item -Path $zipPath -Force
    }
    if (-not (Test-Path $zipPath)) {
        Write-Output "Downloading Elasticsearch $Version from $downloadUrl"
        & curl.exe -L --fail --retry 8 --retry-all-errors --output $zipPath $downloadUrl
        if ($LASTEXITCODE -ne 0) {
            throw "curl.exe failed to download Elasticsearch from $downloadUrl"
        }
    }
    Write-Output "Extracting Elasticsearch to $installRoot"
    Expand-Archive -Path $zipPath -DestinationPath $installRoot -Force
}

$stdoutPath = Join-Path $logsRoot "stdout.log"
$stderrPath = Join-Path $logsRoot "stderr.log"
$env:ES_JAVA_OPTS = "-Xms${HeapMb}m -Xmx${HeapMb}m"

$arguments = @(
    "-Ediscovery.type=single-node"
    "-Expack.security.enabled=false"
    "-Ehttp.host=127.0.0.1"
    "-Etransport.host=127.0.0.1"
    "-Epath.data=$dataRoot"
    "-Epath.logs=$logsRoot"
)

$process = Start-Process `
    -FilePath (Join-Path $esHome "bin\elasticsearch.bat") `
    -ArgumentList $arguments `
    -WorkingDirectory $esHome `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru

$process.Id | Set-Content -Path $pidPath -Encoding ascii

$ready = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    Start-Sleep -Seconds 2
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri $ElasticUrl -TimeoutSec 3
        if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
            $ready = $true
            break
        }
    } catch {
    }
    if ($process.HasExited) {
        break
    }
}

if (-not $ready) {
    if ($process.HasExited) {
        throw "Elasticsearch exited during startup. Check $stdoutPath and $stderrPath"
    }
    throw "Elasticsearch did not become ready at $ElasticUrl. Check $stdoutPath and $stderrPath"
}

Write-Output "Elasticsearch started. PID=$($process.Id) URL=$ElasticUrl"
