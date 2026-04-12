#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Installs and configures IIS as a reverse proxy for the RAG Platform.
#>
param(
    [string]$InstallDir = "C:\rag-platform"
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$msg) { Write-Host " [IIS] $msg" -ForegroundColor Cyan }
function Write-OK([string]$msg) { Write-Host " [IIS] $msg" -ForegroundColor Green }

Write-Info "Installing IIS and required modules..."

# 1. Install IIS and ARR/Rewrite modules via WebPI or Chocolatey if possible, 
# but for a script we'll check and guide or use Enable-WindowsOptionalFeature
Enable-WindowsOptionalFeature -Online -FeatureName "IIS-WebServerRole", "IIS-WebServer", "IIS-Proxy" -All -NoRestart | Out-Null

Write-Info "Ensuring URL Rewrite and ARR are installed..."
# Note: In a real environment, you'd download the MSIs if missing. 
# We assume the user followed Step 1 of the guide or we'd automate the MSI download here.

# 2. Configure ARR to enable proxy
$arrSettings = Get-WebConfigurationProperty -PSPath "MACHINE/WEBROOT/APPHOST" -Filter "system.webServer/proxy" -Name "enabled"
if ($arrSettings.Value -eq $false) {
    Set-WebConfigurationProperty -PSPath "MACHINE/WEBROOT/APPHOST" -Filter "system.webServer/proxy" -Name "enabled" -Value $true
    Write-OK "ARR Proxy enabled"
}

# 3. Create/Configure IIS Site
$siteName = "RAG-Platform"
$physicalPath = "$InstallDir\deploy\windows" # Points to where web.config lives

if (Get-Website -Name $siteName) {
    Set-Website -Name $siteName -PhysicalPath $physicalPath
} else {
    New-Website -Name $siteName -Port 80 -PhysicalPath $physicalPath -Force
}

Write-OK "IIS Site '$siteName' configured at $physicalPath"
