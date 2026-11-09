# Flay-Stimme trainieren: Datensatz erzeugen (Qwen3-TTS spricht die Saetze, Whisper prueft).
# Kann abgebrochen und erneut gestartet werden; fertige Saetze werden uebersprungen.
#   powershell -ExecutionPolicy Bypass -File tools\voice-training\dataset.ps1 [-Voice ramona]
#   ... dataset.ps1 -Recorded <Ordner>   # eigene Aufnahmen aus "flay voice dataset" statt Qwen3-TTS
param([string]$Voice = "ramona", [string]$Recorded)
$ErrorActionPreference = "Stop"
# uv from setup.ps1, also when this shell was opened before it was installed
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
Push-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))
try {
    if ($Recorded) {
        uv run python tools\voice-training\make_dataset.py --recorded $Recorded
    } else {
        uv run python tools\voice-training\make_dataset.py --voice $Voice
    }
    if ($LASTEXITCODE -ne 0) { throw "Datensatz fehlgeschlagen." }
} finally {
    Pop-Location
}
Write-Host "`nWeiter mit: tools\voice-training\train.ps1" -ForegroundColor Green
