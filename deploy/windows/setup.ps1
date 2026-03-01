#Requires -RunAsAdministrator
<#
.SYNOPSIS
    One-shot setup for RAG Platform on Windows.
    Run from the project root: powershell -ExecutionPolicy Bypass -File deploy\windows\setup.ps1
#>
param(
    [string]$Hostname   = "rag-platform.example.org",
    [string]$InstallDir = "C:\rag-platform",
    [string]$SslDir     = "C:\rag-platform\ssl"
)

$ProjectDir = (Resolve-Path "$PSScriptRoot\..\.." ).Path
$ErrorActionPreference = "Stop"

function Get-PrimaryIPv4 {
    $cfg = Get-NetIPConfiguration |
        Where-Object {
            $_.IPv4Address -and
            $_.IPv4DefaultGateway -and
            $_.NetAdapter.Status -eq "Up"
        } |
        Select-Object -First 1

    if ($cfg -and $cfg.IPv4Address) {
        return $cfg.IPv4Address.IPAddress
    }

    return (Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object {
            $_.IPAddress -notlike "169.*" -and
            $_.IPAddress -notlike "172.16.*" -and
            $_.IPAddress -notlike "172.17.*" -and
            $_.IPAddress -notlike "172.18.*" -and
            $_.IPAddress -notlike "172.19.*" -and
            $_.InterfaceAlias -notlike "*Loopback*" -and
            $_.InterfaceAlias -notlike "*vEthernet*" -and
            $_.InterfaceAlias -notlike "*WSL*" -and
            $_.InterfaceAlias -notlike "*Docker*" -and
            $_.InterfaceAlias -notlike "*Hyper-V*" -and
            $_.InterfaceAlias -notlike "*VirtualBox*" -and
            $_.InterfaceAlias -notlike "*VMware*"
        } |
        Select-Object -First 1).IPAddress
}

if ([string]::IsNullOrWhiteSpace($Hostname)) {
    $domain = $env:USERDNSDOMAIN
    if ([string]::IsNullOrWhiteSpace($domain)) {
        $domain = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().DomainName
    }

    if (-not [string]::IsNullOrWhiteSpace($domain)) {
        $Hostname = "$env:COMPUTERNAME.$domain".ToLower()
    } else {
        $Hostname = "$env:COMPUTERNAME.local".ToLower()
        Write-Warn "No AD/DNS domain detected. Using '$Hostname'. Pass -Hostname explicitly for production DNS."
    }
}

function Write-Step([int]$n, [string]$msg) {
    Write-Host ""
    Write-Host "  [$n/7] $msg" -ForegroundColor Cyan
}
function Write-OK([string]$msg)   { Write-Host "         $msg" -ForegroundColor Green  }
function Write-Warn([string]$msg) { Write-Host "          $msg" -ForegroundColor Yellow }
function Write-Fail([string]$msg) { Write-Host "         $msg" -ForegroundColor Red; exit 1 }

Clear-Host
Write-Host ""
Write-Host "  " -ForegroundColor Magenta
Write-Host "     RAG Platform  Windows Production Setup   " -ForegroundColor Magenta
Write-Host "  " -ForegroundColor Magenta
Write-Host ""
Write-Host "  Hostname   : $Hostname"
Write-Host "  Project    : $ProjectDir"
Write-Host "  Install to : $InstallDir"
Write-Host ""

#  1. Check Docker Desktop 
Write-Step 1 "Checking Docker Desktop..."
$dockerExe = "$Env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
if (-not (Test-Path $dockerExe)) {
    Write-Warn "Docker Desktop not found. Opening download page..."
    Start-Process "https://docs.docker.com/desktop/install/windows-install/"
    Write-Fail "Install Docker Desktop with WSL2 backend, then re-run this script."
}
Write-OK "Docker Desktop found"

# Ensure Docker is running
try {
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { throw }
    Write-OK "Docker Engine is running"
} catch {
    Write-Warn "Docker Engine not running  starting Docker Desktop..."
    Start-Process $dockerExe
    $waited = 0
    while ($waited -lt 120) {
        Start-Sleep 5; $waited += 5
        docker info 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { Write-OK "Docker Engine ready"; break }
        Write-Host "        Waiting for Docker... (${waited}s)" -NoNewline
        Write-Host "`r" -NoNewline
    }
    if ($LASTEXITCODE -ne 0) { Write-Fail "Docker did not start. Start Docker Desktop manually and retry." }
}

#  2. Copy project files 
Write-Step 2 "Copying project to $InstallDir..."
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# robocopy handles large trees; exit codes 0-7 are success
$rc = (robocopy $ProjectDir $InstallDir /E /XD ".git" "__pycache__" /XF "*.pyc" "*.pyo" /NFL /NDL /NJH /NJS).ExitCode
if ($LASTEXITCODE -le 7) { Write-OK "Files copied to $InstallDir" }
else { Write-Fail "robocopy failed ($LASTEXITCODE)" }

#  3. Generate SSL certificate 
Write-Step 3 "Generating SSL certificate..."
$sslCrt = Join-Path $SslDir "server.crt"
$sslKey = Join-Path $SslDir "server.key"
if ((Test-Path $sslCrt) -and (Test-Path $sslKey)) {
    Write-Warn "SSL cert/key already exist  skipping (delete $SslDir to regenerate)"
} else {
    Write-Warn "SSL files incomplete or missing  regenerating cert and key..."
    & powershell -ExecutionPolicy Bypass -File "$InstallDir\deploy\windows\ssl-gen.ps1" `
        -Hostname $Hostname -OutputDir $SslDir
    if ($LASTEXITCODE -ne 0) { Write-Fail "SSL generation failed. Check deploy\\windows\\ssl-gen.ps1 output." }
    Write-OK "SSL certificate generated at $SslDir"
}

#  4. Configure .env 
Write-Step 4 "Configuring .env..."
$envFile = Join-Path $InstallDir ".env"
if (-not (Test-Path $envFile)) {
    $secret    = -join ((1..32) | ForEach-Object { '{0:x}' -f (Get-Random -Max 16) })

    @"
SECRET_KEY=$secret

CHROMA_HOST=chromadb
CHROMA_PORT=8000

OLLAMA_HOST=ollama
OLLAMA_PORT=11434
DEFAULT_LLM_MODEL=llama3.2:3b

DATABASE_URL=sqlite:////app/data/rag_system.db
UPLOAD_DIR=/app/uploads

# First registered user becomes admin automatically

# Windows-specific
SSL_DIR=$($SslDir -replace '\\', '/')
WATCHFILES_FORCE_POLLING=true
WATCHFILES_FORCE_POLLING_DELAY=500
"@ | Set-Content $envFile

    Write-OK ".env created"
} else {
    Write-Warn ".env already exists  skipping"
}

#  5. Register Task Scheduler tasks 
Write-Step 5 "Registering auto-start tasks..."
& powershell -ExecutionPolicy Bypass -File "$InstallDir\deploy\windows\install-tasks.ps1" `
    -InstallDir $InstallDir -ProjectDir $InstallDir
if ($LASTEXITCODE -ne 0) { Write-Fail "Task registration failed. Check deploy\\windows\\install-tasks.ps1 output." }
Write-OK "Task Scheduler tasks registered"

#  6. Configure Windows Firewall 
Write-Step 6 "Configuring Windows Firewall..."
$rules = @(
    @{ Name="RAG Platform HTTP";  Port=80;  Proto="TCP" },
    @{ Name="RAG Platform HTTPS"; Port=443; Proto="TCP" }
)
foreach ($rule in $rules) {
    $existing = Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue
    if ($existing) { Remove-NetFirewallRule -DisplayName $rule.Name }
    New-NetFirewallRule `
        -DisplayName $rule.Name `
        -Direction   Inbound `
        -Protocol    $rule.Proto `
        -LocalPort   $rule.Port `
        -Action      Allow `
        -Profile     Any | Out-Null
    Write-OK "Firewall: port $($rule.Port) open"
}

#  7. Build and start 
Write-Step 7 "Building and starting RAG Platform..."
Set-Location $InstallDir
docker compose build
if ($LASTEXITCODE -ne 0) { Write-Fail "docker compose build failed. Fix docker-compose.yml and retry." }
docker compose up -d
if ($LASTEXITCODE -ne 0) { Write-Fail "docker compose up -d failed. Services were not started." }

$serverIP = Get-PrimaryIPv4

Write-Host ""
Write-Host "  " -ForegroundColor Green
Write-Host "      RAG Platform is running!                         " -ForegroundColor Green
Write-Host "  " -ForegroundColor Green
Write-Host ""
Write-Host "    Local URL    : https://localhost" -ForegroundColor White
Write-Host "    Network URL  : https://$serverIP" -ForegroundColor White
Write-Host "    Org URL      : https://$Hostname  (after DNS is set)" -ForegroundColor White
Write-Host "    Admin panel  : https://localhost/admin.html" -ForegroundColor White
Write-Host "    API docs     : https://localhost/api/docs" -ForegroundColor White
Write-Host ""
Write-Host "    DNS record to request from IT team:" -ForegroundColor Yellow
Write-Host "      Type: A    Name: $Hostname    Value: $serverIP" -ForegroundColor Yellow
Write-Host ""
Write-Host "    Models downloading in background (~3 GB):"
Write-Host "      docker logs -f rag_ollama"
Write-Host ""
Write-Host "    Auto-start:  Platform starts automatically at every Windows boot"
Write-Host "     File watch: Changes in backend/app or frontend trigger auto-rebuild"
Write-Host ""


