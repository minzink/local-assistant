# Cassi: einen Trainingsstand als Piper-Stimme exportieren -> output\cassi-ramona-medium.onnx(.json)
#   tools\voice-training\export.ps1                          # neuester Stand (last.ckpt)
#   tools\voice-training\export.ps1 -Checkpoint <Pfad.ckpt>  # bestimmter Stand, z.B. aus runs\...\checkpoints
#   tools\voice-training\export.ps1 -Quality high            # aus runs-high\ -> cassi-ramona-high.onnx
param(
    [string]$Checkpoint,
    [string]$Name = "cassi-ramona",
    [ValidateSet("medium", "high")][string]$Quality = "medium"
)
$ErrorActionPreference = "Stop"
$Root = Join-Path $PSScriptRoot "work"
$Suffix = if ($Name -eq "cassi-ramona") { "" } else { "-$Name" }
if ($Quality -eq "high") { $Suffix += "-high" }
$Py = Join-Path $Root "train-venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $Root "piper1-gpl\src"
if (-not $Checkpoint) {
    $Checkpoint = (Get-ChildItem (Join-Path $Root "runs$Suffix") -Recurse -Filter "last.ckpt" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}
if (-not $Checkpoint) { throw "Kein Checkpoint gefunden." }
$Out = Join-Path $Root "output\$Name-$Quality.onnx"
& $Py -m piper.train.export_onnx --checkpoint $Checkpoint --output-file $Out
if ($LASTEXITCODE -ne 0) { throw "Export fehlgeschlagen." }
Write-Host "`nExportiert: $Out (aus $Checkpoint)" -ForegroundColor Green
Write-Host "Beide Dateien aus tools\voice-training\work\output nach %LOCALAPPDATA%\cassi\voices kopieren."
