# Test SSH verso il server come fa il deploy GitHub Actions (senza eseguire pip).
# Uso (PowerShell):
#   $env:SSH_HOST = "1.2.3.4"
#   $env:SSH_USER = "deploy"
#   $env:SSH_KEY_PATH = "C:\path\to\private_key"
#   $env:SSH_PORT = "22"   # opzionale
#   .\scripts\test-ssh-deploy.ps1

param(
    [string]$Host = $env:SSH_HOST,
    [string]$User = $env:SSH_USER,
    [string]$KeyPath = $env:SSH_KEY_PATH,
    [int]$Port = $(if ($env:SSH_PORT) { [int]$env:SSH_PORT } else { 22 })
)

$ErrorActionPreference = "Stop"

if (-not $Host -or -not $User -or -not $KeyPath) {
    Write-Host "Imposta SSH_HOST, SSH_USER, SSH_KEY_PATH (variabili d'ambiente) oppure passa -Host -User -KeyPath" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path -LiteralPath $KeyPath)) {
    Write-Host "File chiave non trovato: $KeyPath" -ForegroundColor Red
    exit 1
}

$ssh = Get-Command ssh -ErrorAction SilentlyContinue
if (-not $ssh) {
    Write-Host "Comando 'ssh' non trovato. Installa OpenSSH Client (Windows Optional Features)." -ForegroundColor Red
    exit 1
}

Write-Host "Test SSH: ${User}@${Host}:${Port} ..." -ForegroundColor Cyan
& ssh -i $KeyPath -p $Port -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new "${User}@${Host}" "echo SSH_OK && hostname && uname -a"
$code = $LASTEXITCODE
if ($code -ne 0) {
    Write-Host "SSH fallito (exit $code). Controlla firewall, chiave in authorized_keys, utente e porta." -ForegroundColor Red
    exit $code
}
Write-Host "OK: sessione SSH funzionante." -ForegroundColor Green
