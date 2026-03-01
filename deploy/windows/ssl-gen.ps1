#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Generates a self-signed SSL certificate for the RAG Platform.
    Uses built-in Windows CryptoAPI (no OpenSSL required).
.USAGE
    powershell -ExecutionPolicy Bypass -File deploy\windows\ssl-gen.ps1
    powershell -ExecutionPolicy Bypass -File deploy\windows\ssl-gen.ps1 -Hostname "myserver.org.ac.uk"
#>
param(
    [string]$Hostname   = "hostname.sample.ac.uk",
    [string]$OutputDir  = "C:\rag-platform\ssl",
    [int]   $ValidYears = 10,
    [string]$Org        = "Sample Organisation",
    [string]$Country    = "GB"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "Generating SSL certificate" -ForegroundColor Cyan
Write-Host "   Hostname  : $Hostname"
Write-Host "   Output    : $OutputDir"
Write-Host "   Valid for : $ValidYears years"
Write-Host ""

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$existing = Get-ChildItem "Cert:\LocalMachine\My" |
    Where-Object { $_.Subject -like "*$Hostname*" }
if ($existing) {
    Write-Host "   Removing old certificate..." -ForegroundColor Yellow
    $existing | Remove-Item -Force
}

$cert = New-SelfSignedCertificate `
    -DnsName              $Hostname, "localhost", "127.0.0.1" `
    -CertStoreLocation    "Cert:\LocalMachine\My" `
    -NotAfter             (Get-Date).AddYears($ValidYears) `
    -KeyAlgorithm         RSA `
    -KeyLength            4096 `
    -HashAlgorithm        SHA256 `
    -KeyUsage             DigitalSignature, KeyEncipherment `
    -TextExtension        @("2.5.29.37={text}1.3.6.1.5.5.7.3.1") `
    -Subject              "CN=$Hostname, O=$Org, C=$Country"

Write-Host "   Certificate created: $($cert.Thumbprint)" -ForegroundColor Green

$pfxPath = Join-Path $OutputDir "server.pfx"
$pwd = ConvertTo-SecureString -String "temp_export_pwd_rag" -Force -AsPlainText
Export-PfxCertificate -Cert $cert -FilePath $pfxPath -Password $pwd | Out-Null

Add-Type -AssemblyName System.Security
$certPath = Join-Path $OutputDir "server.crt"
$keyPath = Join-Path $OutputDir "server.key"
$dockerOutputDir = $OutputDir -replace '\\', '/'
$opensslScript = @"
set -e
apk add --no-cache openssl >/dev/null
openssl pkcs12 -in /work/server.pfx -clcerts -nokeys -passin pass:temp_export_pwd_rag -out /work/server.crt.tmp
openssl pkcs12 -in /work/server.pfx -nocerts -nodes -passin pass:temp_export_pwd_rag -out /work/server.key.tmp
sed '/^Bag Attributes/d;/^subject=/d;/^issuer=/d;/^localKeyID:/d' /work/server.crt.tmp > /work/server.crt
sed '/^Bag Attributes/d;/^subject=/d;/^issuer=/d;/^localKeyID:/d' /work/server.key.tmp > /work/server.key
rm -f /work/server.crt.tmp /work/server.key.tmp
"@
docker run --rm -v "${dockerOutputDir}:/work" alpine:3.20 sh -lc $opensslScript | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Failed to extract PEM files from PFX using Docker/OpenSSL."
}

$distPath = Join-Path $OutputDir "$Hostname.crt"
Copy-Item $certPath $distPath -Force

$store = [System.Security.Cryptography.X509Certificates.X509Store]::new("Root", "LocalMachine")
$store.Open("ReadWrite")
$store.Add($cert)
$store.Close()

Remove-Item $pfxPath -Force

Write-Host ""
Write-Host "SSL files ready:" -ForegroundColor Green
Write-Host "   Private key  : $keyPath"
Write-Host "   Certificate  : $certPath"
Write-Host "   Distributable: $distPath"
Write-Host ""
Write-Host "Share '$distPath' with users so their browser trusts the site:" -ForegroundColor Yellow
Write-Host "   Windows : Double-click .crt > Install Certificate > Local Machine > Trusted Root CAs"
Write-Host "   macOS   : Double-click .crt > Keychain Access > Always Trust"
Write-Host "   Linux   : sudo cp $Hostname.crt /usr/local/share/ca-certificates/ && sudo update-ca-certificates"
Write-Host ""
