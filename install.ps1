# Cassi installieren oder reparieren (Windows). Aus dem Checkout: install.cmd (Doppelklick).
# Ohne Checkout (laedt Cassi nach $env:CASSI_HOME bzw. %USERPROFILE%\cassi), in PowerShell:
#   irm https://raw.githubusercontent.com/minzink/local-assistant/main/install.ps1 | iex
# Danach oeffnet sich die Verwaltung (Reparieren, Updaten, Deinstallieren): cassi manage
# ASCII only: Windows PowerShell 5.1 reads scripts without BOM as ANSI.
$ErrorActionPreference = "Stop"
$Repo = "https://github.com/minzink/local-assistant.git"

if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "scripts\setup.ps1"))) {
    $Root = $PSScriptRoot
} else {
    $Root = if ($env:CASSI_HOME) { $env:CASSI_HOME } else { Join-Path $HOME "cassi" }
    if (-not (Test-Path (Join-Path $Root ".git"))) {
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            Write-Host "Git wird installiert (winget) ..."
            winget install --id Git.Git --exact --silent --accept-source-agreements --accept-package-agreements
            $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User") + ";$env:ProgramFiles\Git\cmd"
        }
        Write-Host "Lade Cassi nach $Root ..."
        git clone $Repo $Root
        if ($LASTEXITCODE -ne 0) { throw "git clone fehlgeschlagen." }
    }
}

& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root "scripts\setup.ps1") @args
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nDie Installation ist nicht vollstaendig - das Log steht oben. Spaeter erneut starten." -ForegroundColor Red
    exit 1
}
$env:Path = "$env:USERPROFILE\.local\bin;" + $env:Path
Write-Host "`nOeffne die Verwaltung ..."
Push-Location $Root
try { & uv run --quiet cassi manage } finally { Pop-Location }
