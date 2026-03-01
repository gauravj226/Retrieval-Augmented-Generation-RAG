#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Registers three Windows Task Scheduler tasks:
    1. Start Docker Desktop on boot
    2. Start RAG Platform after Docker is ready
    3. Start the background file watcher
#>
param(
    [string]$InstallDir = "C:\rag-platform",
    [string]$ProjectDir = "$PSScriptRoot\..\.."
)

$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path $ProjectDir).Path

Write-Host "Registering Task Scheduler tasks..." -ForegroundColor Cyan
Write-Host "   Project: $ProjectDir"
Write-Host ""

function Register-RAGTask {
    param(
        [string]$Name,
        [string]$Description,
        [string]$ScriptPath,
        [int]$DelaySeconds,
        [string]$WorkingDir
    )

    Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue |
        Unregister-ScheduledTask -Confirm:$false

    $trigger = New-ScheduledTaskTrigger -AtStartup
    $trigger.Delay = "PT${DelaySeconds}S"

    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ScriptPath`"" `
        -WorkingDirectory $WorkingDir

    $principal = New-ScheduledTaskPrincipal `
        -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
        -LogonType S4U `
        -RunLevel Highest

    $settings = New-ScheduledTaskSettingsSet `
        -Compatibility Win8 `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 2) `
        -ExecutionTimeLimit ([TimeSpan]::Zero)

    Register-ScheduledTask `
        -TaskName $Name `
        -Description $Description `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Force | Out-Null

    Write-Host "   Task registered: $Name (starts ${DelaySeconds}s after boot)" -ForegroundColor Green
}

$dockerScript = Join-Path $InstallDir "deploy\windows\_start-docker.ps1"
@"
# Auto-generated: start Docker Desktop
`$dockerExe = "`$Env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
if (Test-Path `$dockerExe) {
    Start-Process `$dockerExe
}
"@ | Set-Content -Path $dockerScript -Encoding ASCII

Register-RAGTask `
    -Name "RAG-Platform-DockerDesktop" `
    -Description "Starts Docker Desktop for the RAG Platform on boot" `
    -ScriptPath $dockerScript `
    -DelaySeconds 30 `
    -WorkingDir $InstallDir

$startScript = Join-Path $InstallDir "deploy\windows\_start-rag.ps1"
@"
# Auto-generated: wait for Docker then start RAG Platform
`$maxWait = 180
`$waited = 0
while (`$waited -lt `$maxWait) {
    try {
        docker info 2>`$null | Out-Null
        if (`$LASTEXITCODE -eq 0) { break }
    } catch {}
    Start-Sleep -Seconds 5
    `$waited += 5
}
if (`$waited -ge `$maxWait) {
    exit 1
}
Set-Location "$ProjectDir"
docker compose up -d *>> "`$Env:TEMP\rag-platform.log"
if (`$LASTEXITCODE -ne 0) { exit `$LASTEXITCODE }
"@ | Set-Content -Path $startScript -Encoding ASCII

Register-RAGTask `
    -Name "RAG-Platform-Start" `
    -Description "Starts RAG Platform containers after Docker Desktop is ready" `
    -ScriptPath $startScript `
    -DelaySeconds 90 `
    -WorkingDir $ProjectDir

$watchScript = Join-Path $InstallDir "deploy\windows\_start-watcher.ps1"
@"
# Auto-generated: start background file watcher
Start-Sleep -Seconds 120
`$watcherScript = "$ProjectDir\deploy\windows\file-watcher.ps1"
if (Test-Path `$watcherScript) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden `
        -File `$watcherScript -ProjectDir "$ProjectDir"
}
"@ | Set-Content -Path $watchScript -Encoding ASCII

Register-RAGTask `
    -Name "RAG-Platform-FileWatcher" `
    -Description "Watches for file changes and triggers container rebuilds" `
    -ScriptPath $watchScript `
    -DelaySeconds 120 `
    -WorkingDir $ProjectDir

Write-Host ""
Write-Host "All tasks registered. Boot sequence on next startup:" -ForegroundColor Green
Write-Host "   0:30 -> Docker Desktop starts"
Write-Host "   1:30 -> RAG Platform containers start (waits for Docker)"
Write-Host "   2:00 -> File watcher starts"
Write-Host ""
Write-Host "To see tasks: Open Task Scheduler > Task Scheduler Library > search 'RAG'"
