# Cassi: die Stimme in einem Durchgang trainieren - einrichten, Datensatz, Training, Export und
# in den Stimmen-Ordner kopieren. Gestartet von der Verwaltung (Tab "Stimme") in einem eigenen
# Fenster; geht auch von Hand. Abgebrochen? Erneut starten: jeder Schritt setzt fort.
#   powershell -ExecutionPolicy Bypass -File tools\voice-training\run.ps1 -VoicesDir <Ordner>
#   ... run.ps1 -Quality high -VoicesDir <Ordner>
# Der Stand steht in work\status.json (liest die Verwaltung), die Ausgabe in work\run.log.
param(
    [ValidateSet("medium", "high")][string]$Quality = "medium",
    [int]$Epochs = 2000,
    # Wohin die fertige Stimme kommt: der Stimmen-Ordner des Programms (downloads\voices).
    [Parameter(Mandatory = $true)][string]$VoicesDir,
    [string]$Name = "cassi-ramona"
)
$ErrorActionPreference = "Stop"
$Here = $PSScriptRoot
$Work = Join-Path $Here "work"
$StatusFile = Join-Path $Work "status.json"
New-Item -ItemType Directory -Force $Work | Out-Null

# Kein Ruhezustand, solange dieses Fenster laeuft (ohne die Energieeinstellungen zu aendern).
Add-Type -Namespace Cassi -Name Power -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
"@
[Cassi.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null  # CONTINUOUS | SYSTEM_REQUIRED

$Started = (Get-Date).ToString("o")
function Status($step, $text, [bool]$done = $false, [bool]$ok = $false) {
    [ordered]@{
        pid = $PID; quality = $Quality; epochs = $Epochs; name = "$Name-$Quality"
        step = $step; text = $text; started = $Started; updated = (Get-Date).ToString("o")
        done = $done; ok = $ok
    } | ConvertTo-Json | Set-Content -Encoding UTF8 $StatusFile
    Write-Host "`n=== $text" -ForegroundColor Cyan
}

function Run($script, [string[]]$arguments = @()) {
    $file = Join-Path $Here $script
    & powershell -NoProfile -ExecutionPolicy Bypass -File $file @arguments
    if ($LASTEXITCODE -ne 0) { throw "$script ist fehlgeschlagen (Code $LASTEXITCODE)." }
}

Start-Transcript -Path (Join-Path $Work "run.log") -Append | Out-Null
try {
    Status "setup" "1/5 Einrichten (beim ersten Mal ~20-30 min)"
    Run "setup.ps1"

    Status "dataset" "2/5 Datensatz erzeugen (beim ersten Mal ~1 h)"
    Run "dataset.ps1"

    Status "train" "3/5 Trainieren ($Quality, $Epochs Epochen - grob ein Tag)"
    $Runs = Join-Path $Work $(if ($Quality -eq "high") { "runs-high" } else { "runs" })
    $TrainArgs = @("-Epochs", $Epochs, "-Quality", $Quality, "-Name", $Name)
    $Last = if (Test-Path $Runs) { Get-ChildItem $Runs -Recurse -Filter "last.ckpt" } else { $null }
    if ($Last) {
        Write-Host "Setze das angefangene Training fort."
        $TrainArgs += "-Resume"
    }
    Run "train.ps1" $TrainArgs

    Status "export" "4/5 Exportieren"
    Run "export.ps1" @("-Quality", $Quality, "-Name", $Name)

    Status "install" "5/5 In den Stimmen-Ordner kopieren"
    New-Item -ItemType Directory -Force $VoicesDir | Out-Null
    foreach ($Suffix in @(".onnx", ".onnx.json")) {
        $File = "$Name-$Quality$Suffix"
        $Target = Join-Path $VoicesDir $File
        if (Test-Path $Target) {
            # Die bisherige Stimme bleibt erhalten (kein .onnx am Ende: nicht als Stimme gelistet).
            Copy-Item $Target "$Target.vor-training" -Force
        }
        Copy-Item (Join-Path $Work "output\$File") $Target -Force
    }

    Status "done" "Fertig: $Name-$Quality liegt in $VoicesDir" $true $true
    Write-Host "In der Verwaltung: 'Trainierte Stimme verwenden', dann neu starten." -ForegroundColor Green
} catch {
    Status "failed" "Fehlgeschlagen: $($_.Exception.Message)" $true $false
    Write-Host "Erneut starten setzt beim letzten Schritt fort." -ForegroundColor Yellow
    exit 1
} finally {
    Stop-Transcript | Out-Null
    [Cassi.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}
