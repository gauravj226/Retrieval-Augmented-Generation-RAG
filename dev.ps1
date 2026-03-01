<#
.SYNOPSIS  Start RAG Platform in development mode with hot reload on Windows
.USAGE     powershell -ExecutionPolicy Bypass -File dev.ps1 [-Build] [-Down] [-Logs]
#>
param(
    [switch]$Build,
    [switch]$Down,
    [switch]$Logs
)

$compose = "docker compose -f docker-compose.yml -f docker-compose.dev.yml"

if ($Down)  { Invoke-Expression "$compose down"; exit }
if ($Logs)  { Invoke-Expression "$compose logs -f backend frontend"; exit }
if ($Build) {
    Write-Host "🔨 Building dev images..." -ForegroundColor Cyan
    Invoke-Expression "$compose build"
}

Write-Host ""
Write-Host "🚀 RAG Platform — DEVELOPMENT MODE" -ForegroundColor Magenta
Write-Host "   Frontend  → http://localhost:3000"
Write-Host "   API docs  → http://localhost:8000/docs"
Write-Host ""
Write-Host "   File watching ACTIVE:" -ForegroundColor Green
Write-Host "   • backend\app\**   → synced → uvicorn auto-reloads"
Write-Host "   • frontend\**      → synced → refresh browser to see"
Write-Host "   • requirements.txt → full image rebuild"
Write-Host ""
Write-Host "   Press Ctrl+C to stop"
Write-Host ""

Invoke-Expression "$compose up --watch"
