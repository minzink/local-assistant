# Flay: einen Trainingsstand als Piper-Stimme exportieren -> output\flay-ramona-medium.onnx(.json)
#   tools\voice-training\export.ps1                          # neuester Stand (last.ckpt)
#   tools\voice-training\export.ps1 -Checkpoint <Pfad.ckpt>  # bestimmter Stand, z.B. aus runs\...\checkpoints
param([string]$Checkpoint, [string]$Name = "flay-ramona")
$ErrorActionPreference = "Stop"
$Root = Join-Path $PSScriptRoot "work"
$Suffix = if ($Name -eq "flay-ramona") { "" } else { "-$Name" }
$Py = Join-Path $Root "train-venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $Root "piper1-gpl\src"
if (-not $Checkpoint) {
    $Checkpoint = (Get-ChildItem (Join-Path $Root "runs$Suffix") -Recurse -Filter "last.ckpt" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}
if (-not $Checkpoint) { throw "Kein Checkpoint gefunden." }
$Out = Join-Path $Root "output\$Name-medium.onnx"
& $Py -m piper.train.export_onnx --checkpoint $Checkpoint --output-file $Out
if ($LASTEXITCODE -ne 0) { throw "Export fehlgeschlagen." }
Write-Host "`nExportiert: $Out (aus $Checkpoint)" -ForegroundColor Green
Write-Host "Beide Dateien aus tools\voice-training\work\output nach %LOCALAPPDATA%\flay\voices kopieren."
