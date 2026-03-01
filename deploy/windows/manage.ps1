<#
.SYNOPSIS  RAG Platform management for Windows
.USAGE     powershell -ExecutionPolicy Bypass -File deploy\windows\manage.ps1 [command]
#>
param([string]$Command = "help", [string]$Model = "mistral")

$ProjectDir = (Resolve-Path "$PSScriptRoot\..\.." ).Path
Set-Location $ProjectDir

switch ($Command) {
    "start"   { docker compose up -d; Write-Host "✅ Started" -ForegroundColor Green }
    "stop"    { docker compose down;  Write-Host "🛑 Stopped" -ForegroundColor Yellow }
    "restart" { docker compose down; docker compose up -d; Write-Host "🔄 Restarted" -ForegroundColor Cyan }
    "status"  { docker compose ps }
    "logs"    { docker compose logs -f --tail=100 }
    "update"  {
        Write-Host "🔄 Rebuilding all services..."
        docker compose build --pull
        docker compose up -d
        Write-Host "✅ Updated" -ForegroundColor Green
    }
    "backup"  {
        $ts = Get-Date -Format "yyyyMMdd_HHmmss"
        $dir = "C:\rag-platform-backups\$ts"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        docker cp rag_backend:/app/data/rag_system.db "$dir\rag_system.db"
        docker cp rag_backend:/app/uploads "$dir\uploads"
        Write-Host "✅ Backup saved: $dir" -ForegroundColor Green
    }
    "pull-model" {
        Write-Host "📥 Pulling model: $Model"
        docker exec rag_ollama ollama pull $Model
    }
    "list-models" { docker exec rag_ollama ollama list }
    "watcher-log" { Get-Content "$Env:TEMP\rag-filewatcher.log" -Tail 50 }
    "watcher-start" {
        Start-Process powershell -ArgumentList `
            "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$ProjectDir\deploy\windows\file-watcher.ps1`"" `
            -WorkingDirectory $ProjectDir
        Write-Host "✅ File watcher started in background" -ForegroundColor Green
    }
    default {
        @"

  RAG Platform — Windows Management
  Usage: powershell -File deploy\windows\manage.ps1 [command]

  start           Start all containers
  stop            Stop all containers
  restart         Restart all containers
  status          Show container status
  logs            Tail live logs
  update          Pull + rebuild all images
  backup          Backup database and uploads to C:\rag-platform-backups
  pull-model      Pull Ollama model  (e.g. -Model mistral)
  list-models     List downloaded models
  watcher-log     View last 50 lines of file watcher log
  watcher-start   Start file watcher manually (background)
"@
    }
}
