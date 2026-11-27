<#
.SYNOPSIS
    Start (or stop) the Hexagonal Geometric Test Engine in Docker and open it in your browser.

.EXAMPLE
    .\start.ps1          # build if needed, start, wait until healthy, open http://localhost:8501
    .\start.ps1 -Logs    # follow the app's log output (Ctrl+C to stop watching; the app keeps running)
    .\start.ps1 -Stop    # stop and remove the container (your data\ folder is kept)

.NOTES
    Works in Windows PowerShell 5.1 and PowerShell 7. If Windows blocks the script, run once:
        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>
[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$Logs,
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = 'Stop'
$Url = 'http://localhost:8501'
$Container = 'hex-engine'
Set-Location -Path $PSScriptRoot

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }

# Run a native command with its stderr discarded. In Windows PowerShell 5.1, redirecting a native command's
# stderr while $ErrorActionPreference is 'Stop' turns the first stderr line into a terminating error, so the
# friendly messages below would never be shown. Lower the preference just for the call; $LASTEXITCODE is kept.
function Invoke-Quiet([scriptblock]$Command) {
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Command 2>$null } finally { $ErrorActionPreference = $old }
}
function Fail($msg) {
    Write-Host ""
    Write-Host $msg -ForegroundColor Red
    exit 1
}

# 1. Is Docker installed?
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Fail ("Docker was not found.`n" +
          "Install Docker Desktop from https://www.docker.com/products/docker-desktop/ (free for personal use),`n" +
          "restart your PC if asked, start Docker Desktop, then run .\start.ps1 again.")
}

# 2. Is the Docker engine running? ('docker info' fails when Docker Desktop hasn't been started.)
Invoke-Quiet { & docker info } | Out-Null
if ($LASTEXITCODE -ne 0) {
    Fail ("Docker is installed but not running.`n" +
          "Start Docker Desktop from the Start menu, wait until it says 'Engine running', then run .\start.ps1 again.")
}

if ($Stop) {
    Write-Step 'Stopping the hex engine...'
    & docker compose down
    if ($LASTEXITCODE -ne 0) { Fail 'docker compose down failed (see the messages above).' }
    Write-Host 'Stopped. Your data\ folder has been kept.' -ForegroundColor Green
    exit 0
}

if ($Logs) {
    Write-Step 'Showing logs (Ctrl+C to stop watching; the app keeps running)...'
    & docker compose logs -f --tail 200 $Container
    exit 0
}

# 3. Build (only rebuilds what changed) and start in the background.
if (-not (Test-Path -Path (Join-Path $PSScriptRoot 'data'))) {
    New-Item -ItemType Directory -Path (Join-Path $PSScriptRoot 'data') | Out-Null
}
Write-Step 'Building and starting the container (the first build downloads ~1 GB and can take a few minutes)...'
& docker compose up -d --build
if ($LASTEXITCODE -ne 0) { Fail 'docker compose up failed (see the messages above).' }

# 4. Wait until the container's health check passes.
Write-Step "Waiting for the app to become healthy (up to $TimeoutSeconds s)..."
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$status = ''
while ((Get-Date) -lt $deadline) {
    $status = (Invoke-Quiet { & docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' $Container })
    if ($status -eq 'healthy') { break }
    if ($status -eq 'exited' -or $status -eq 'dead') { break }
    Write-Host '.' -NoNewline
    Start-Sleep -Seconds 2
}
Write-Host ''

if ($status -ne 'healthy') {
    Write-Host "The app did not become healthy (status: '$status'). Last log lines:" -ForegroundColor Yellow
    & docker compose logs --tail 40 $Container
    Fail 'Fix the error above, then run .\start.ps1 again. (.\start.ps1 -Logs follows the log live.)'
}

Write-Host "Hex engine is running at $Url" -ForegroundColor Green
Write-Host 'Stop it with .\start.ps1 -Stop   |   watch logs with .\start.ps1 -Logs'
Start-Process $Url
