<#
.SYNOPSIS
    Watches backend/app and frontend directories.
    On file change: triggers targeted docker compose rebuild with debouncing.
    Runs as a background process started by Task Scheduler.
.USAGE
    powershell -ExecutionPolicy Bypass -File deploy\windows\file-watcher.ps1
#>
param(
    [string]$ProjectDir  = (Resolve-Path "$PSScriptRoot\..\.." ).Path,
    [int]   $DebounceMs  = 3000,    # wait 3s before acting to batch rapid saves
    [int]   $CooldownSec = 15       # min seconds between rebuilds of same service
)

# ── Logging ───────────────────────────────────────────────────────────────────
$logFile = "$Env:TEMP\rag-filewatcher.log"
function Log([string]$msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $msg"
    Add-Content -Path $logFile -Value $line
    Write-Host $line
}

Log "=== RAG Platform File Watcher started ==="
Log "    Project : $ProjectDir"
Log "    Debounce: ${DebounceMs}ms | Cooldown: ${CooldownSec}s"
Log ""

# ── Cooldown tracking ─────────────────────────────────────────────────────────
$lastRebuild = @{
    backend  = [DateTime]::MinValue
    frontend = [DateTime]::MinValue
}

# ── Rebuild function ──────────────────────────────────────────────────────────
function Invoke-Rebuild([string]$Service) {
    $now = [DateTime]::Now
    if (($now - $lastRebuild[$Service]).TotalSeconds -lt $CooldownSec) {
        Log "   [$Service] Skipped — cooldown active"
        return
    }
    $lastRebuild[$Service] = $now
    Log "   [$Service] Rebuilding..."

    $proc = Start-Process `
        -FilePath         "docker" `
        -ArgumentList     "compose up -d --build $Service" `
        -WorkingDirectory $ProjectDir `
        -PassThru `
        -NoNewWindow `
        -RedirectStandardOutput "$Env:TEMP\rag-rebuild-$Service.log" `
        -RedirectStandardError  "$Env:TEMP\rag-rebuild-$Service-err.log"

    $proc.WaitForExit(120000)  # max 2 minutes per rebuild
    if ($proc.ExitCode -eq 0) {
        Log "   [$Service] ✅ Rebuild complete"
    } else {
        Log "   [$Service] ❌ Rebuild failed (exit $($proc.ExitCode)) — check $Env:TEMP\rag-rebuild-$Service-err.log"
    }
}

# ── Watcher setup ─────────────────────────────────────────────────────────────
$watchPaths = @{
    backend  = Join-Path $ProjectDir "backend\app"
    frontend = Join-Path $ProjectDir "frontend"
}

$watchers   = @{}
$timers     = @{}
$pending    = @{ backend = $false; frontend = $false }

foreach ($service in $watchPaths.Keys) {
    $path = $watchPaths[$service]
    if (-not (Test-Path $path)) {
        Log "   WARNING: Watch path not found: $path"
        continue
    }

    $w = New-Object System.IO.FileSystemWatcher
    $w.Path                  = $path
    $w.IncludeSubdirectories = $true
    $w.EnableRaisingEvents   = $true
    $w.NotifyFilter          = [System.IO.NotifyFilters]::LastWrite `
                             -bor [System.IO.NotifyFilters]::FileName `
                             -bor [System.IO.NotifyFilters]::DirectoryName

    # ── Ignore noise files ────────────────────────────────────────────────────
    $ignore = @('*.pyc', '*.pyo', '__pycache__', '.git', '*.tmp',
                '*.log', 'node_modules', '*.swp', '~*')

    $serviceRef = $service   # capture for closure

    $handler = Register-ObjectEvent -InputObject $w -EventName "Changed" -Action {
        $changed = $Event.SourceEventArgs.FullPath
        $svc     = $Event.MessageData

        # Skip ignored patterns
        foreach ($pat in $using:ignore) {
            if ($changed -like "*$pat*") { return }
        }

        if (-not $using:pending[$svc]) {
            $using:pending[$svc] = $true
            Log "   [$svc] Change detected: $changed (rebuilding in $($using:DebounceMs)ms)"

            # Debounce via Start-Sleep in a job
            Start-Job -ScriptBlock {
                param($svc, $debounce, $pending, $projectDir, $cooldown, $lastRebuild)
                Start-Sleep -Milliseconds $debounce
                $pending[$svc] = $false
                Invoke-Rebuild $svc
            } -ArgumentList $svc, $DebounceMs, $pending, $ProjectDir, $CooldownSec, $lastRebuild | Out-Null
        }
    } -MessageData $serviceRef

    Register-ObjectEvent -InputObject $w -EventName "Created" -Action $handler.Action -MessageData $serviceRef | Out-Null
    Register-ObjectEvent -InputObject $w -EventName "Deleted" -Action $handler.Action -MessageData $serviceRef | Out-Null
    Register-ObjectEvent -InputObject $w -EventName "Renamed" -Action $handler.Action -MessageData $serviceRef | Out-Null

    $watchers[$service] = $w
    Log "   👁  Watching [$service]: $path"
}

# ── Also watch requirements.txt for dependency changes ───────────────────────
$reqPath = Join-Path $ProjectDir "backend\requirements.txt"
if (Test-Path $reqPath) {
    $reqDir = Split-Path $reqPath
    $reqWatcher = New-Object System.IO.FileSystemWatcher
    $reqWatcher.Path   = $reqDir
    $reqWatcher.Filter = "requirements.txt"
    $reqWatcher.EnableRaisingEvents = $true

    Register-ObjectEvent -InputObject $reqWatcher -EventName "Changed" -Action {
        Log "   [backend] requirements.txt changed — full image rebuild triggered"
        $proc = Start-Process "docker" `
            -ArgumentList     "compose build --no-cache backend" `
            -WorkingDirectory $using:ProjectDir `
            -PassThru -NoNewWindow
        $proc.WaitForExit(300000)
        if ($proc.ExitCode -eq 0) {
            docker compose -f $using:ProjectDir/docker-compose.yml up -d backend
            Log "   [backend] ✅ Full rebuild complete"
        }
    } | Out-Null
    Log "   👁  Watching [requirements.txt]: $reqDir"
}

Log ""
Log "File watcher running. Press Ctrl+C to stop."
Log "Log file: $logFile"
Log ""

# ── Keep process alive ────────────────────────────────────────────────────────
try {
    while ($true) {
        # Clean up completed background jobs every 30 seconds
        Get-Job -State Completed | Remove-Job
        Start-Sleep -Seconds 30
    }
} finally {
    Log "File watcher stopping..."
    foreach ($w in $watchers.Values) {
        $w.EnableRaisingEvents = $false
        $w.Dispose()
    }
    Get-EventSubscriber | Unregister-Event
    Log "File watcher stopped."
}
