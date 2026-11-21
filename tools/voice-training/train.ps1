# Cassi: Piper-Stimme "Ramona" (ein Modell fuer Deutsch und Englisch) trainieren.
# Startet vom Kristin-Checkpoint (weiblich, gemeinfrei). Zwischenstaende landen in runs\.
#   tools\voice-training\train.ps1                     # 2000 Epochen (Empfehlung der Piper-Doku fuer < 5 h Audio)
#   tools\voice-training\train.ps1 -Epochs 1000        # kuerzer
#   tools\voice-training\train.ps1 -Resume             # nach Abbruch fortsetzen (gleiche -Epochs angeben!)
#   tools\voice-training\train.ps1 -Quality high       # groesserer Decoder: klingt natuerlicher, braucht
#                                                       # je Satz laenger (Mac-CPU 0,55-0,8 s statt ~0,1 s);
#                                                       # Start ist LJSpeech high (weiblich, gemeinfrei)
param(
    [int]$Epochs = 2000,
    [int]$BatchSize = 16,
    [int]$SegmentSize = 8192,
    [switch]$Resume,
    # Name der Stimme, z.B. cassi-mia fuer die eigene; ergibt <Name>-<Quality>.onnx
    [string]$Name = "cassi-ramona",
    [ValidateSet("medium", "high")][string]$Quality = "medium"
)
$ErrorActionPreference = "Stop"
$Root = Join-Path $PSScriptRoot "work"
# Ramona keeps its folders; any other voice gets its own (dataset-own from -Recorded).
$Suffix = if ($Name -eq "cassi-ramona") { "" } else { "-$Name" }
# high has another decoder: its own runs and cache, never resumed from a medium run.
if ($Quality -eq "high") { $Suffix += "-high" }
$Dataset = if ($Name -eq "cassi-ramona") { "dataset" } else { "dataset-own" }
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
    "--data.config_path", (Join-Path $Root "output\$Name-$Quality.onnx.json"),
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
if ($Quality -eq "high") {
    # Piper's high quality (piper_train ModelAudioConfig.high_quality), matching the checkpoint.
    $TrainArgs += @(
        "--model.resblock", "1",
        "--model.resblock_kernel_sizes", "[3, 7, 11]",
        "--model.resblock_dilation_sizes", "[[1, 3, 5], [1, 3, 5], [1, 3, 5]]",
        "--model.upsample_rates", "[8, 8, 2, 2]",
        "--model.upsample_initial_channel", "512",
        "--model.upsample_kernel_sizes", "[16, 16, 4, 4]"
    )
}
if ($Resume) {
    $Last = Get-ChildItem (Join-Path $Root "runs$Suffix") -Recurse -Filter "last.ckpt" |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $Last) { throw "Kein last.ckpt in runs\ gefunden." }
    Write-Host "Setze fort ab $($Last.FullName)"
    $TrainArgs += @("--ckpt_path", $Last.FullName)
} else {
    $Start = if ($Quality -eq "high") {
        "en/en_US/ljspeech/high/ljspeech-2000.ckpt"
    } else {
        "en/en_US/kristin/medium/kristin-2000.ckpt"
    }
    $StartCkpt = Join-Path $Root "ckpt\$($Start -replace '/', '\')"
    if (-not (Test-Path $StartCkpt)) {
        Write-Host "Lade Start-Checkpoint $Start (~1 GB, gemeinfrei) ..."
        & $Py -c "from huggingface_hub import hf_hub_download; hf_hub_download('rhasspy/piper-checkpoints', '$Start', repo_type='dataset', local_dir=r'$Root\ckpt')"
        if ($LASTEXITCODE -ne 0) { throw "Checkpoint-Download fehlgeschlagen." }
    }
    $TrainArgs += @("--model.warmstart_ckpt", $StartCkpt)
}
& $Py @TrainArgs
exit $LASTEXITCODE
