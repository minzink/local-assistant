# Flay: Piper-Stimme "Ramona" (ein Modell fuer Deutsch und Englisch) trainieren.
# Startet vom Kristin-Checkpoint (weiblich, gemeinfrei). Zwischenstaende landen in runs\.
#   tools\voice-training\train.ps1                     # 2000 Epochen (Empfehlung der Piper-Doku fuer < 5 h Audio)
#   tools\voice-training\train.ps1 -Epochs 1000        # kuerzer
#   tools\voice-training\train.ps1 -Resume             # nach Abbruch fortsetzen (gleiche -Epochs angeben!)
param(
    [int]$Epochs = 2000,
    [int]$BatchSize = 16,
    [int]$SegmentSize = 8192,
    [switch]$Resume,
    # Name der Stimme, z.B. flay-mia fuer die eigene; ergibt <Name>-medium.onnx
    [string]$Name = "flay-ramona"
)
$ErrorActionPreference = "Stop"
$Root = Join-Path $PSScriptRoot "work"
# Ramona keeps its folders; any other voice gets its own (dataset-own from -Recorded).
$Suffix = if ($Name -eq "flay-ramona") { "" } else { "-$Name" }
$Dataset = if ($Name -eq "flay-ramona") { "dataset" } else { "dataset-own" }
$Py = Join-Path $Root "train-venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $Root "piper1-gpl\src"
New-Item -ItemType Directory -Force (Join-Path $Root "output") | Out-Null

$TrainArgs = @(
    "-m", "piper.train", "fit",
    "--data.voice_name", $Name,
    "--data.csv_path", (Join-Path $Root "$Dataset\metadata.csv"),
    "--data.audio_dir", (Join-Path $Root "$Dataset\wavs"),
    "--data.dataset_type", "phoneme_ids",
    "--data.espeak_voice", "de",
    "--data.cache_dir", (Join-Path $Root "cache$Suffix"),
    "--data.config_path", (Join-Path $Root "output\$Name-medium.onnx.json"),
    "--data.batch_size", $BatchSize,
    "--model.sample_rate", "22050",
    "--model.segment_size", $SegmentSize,
    "--model.mos_metric", "none",
    "--trainer.accelerator", "gpu",
    "--trainer.devices", "1",
    "--trainer.max_epochs", $Epochs,
    "--trainer.check_val_every_n_epoch", "10",
    "--trainer.log_every_n_steps", "10",
    "--trainer.default_root_dir", (Join-Path $Root "runs$Suffix")
)
if ($Resume) {
    $Last = Get-ChildItem (Join-Path $Root "runs$Suffix") -Recurse -Filter "last.ckpt" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $Last) { throw "Kein last.ckpt in runs\ gefunden." }
    Write-Host "Setze fort ab $($Last.FullName)"
    $TrainArgs += @("--ckpt_path", $Last.FullName)
} else {
    $TrainArgs += @("--model.warmstart_ckpt",
        (Join-Path $Root "ckpt\en\en_US\kristin\medium\kristin-2000.ckpt"))
}
& $Py @TrainArgs
exit $LASTEXITCODE
