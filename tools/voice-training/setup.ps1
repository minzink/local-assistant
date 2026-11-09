# Flay-Stimme trainieren (Windows, NVIDIA): alles einrichten, nur aus dem Git-Repo.
# Braucht den NVIDIA-Treiber und Internet; uv, Python, PyTorch usw. holt das Skript selbst.
# Aufruf im Repo: powershell -ExecutionPolicy Bypass -File tools\voice-training\setup.ps1
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest ist sonst sehr langsam
$Here = $PSScriptRoot
$Repo = Resolve-Path (Join-Path $Here "..\..")
$Work = Join-Path $Here "work"
$Commit = "304d5dac29c57a30d43fad09ef4b3a70a795b1bc"   # piper1-gpl, damit getestet
$Py = Join-Path $Work "train-venv\Scripts\python.exe"
New-Item -ItemType Directory -Force $Work | Out-Null
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"   # where uv installs itself

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

Step "NVIDIA-GPU pruefen"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi fehlt: zuerst den NVIDIA-Treiber installieren (https://www.nvidia.com/drivers)."
}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

Step "uv installieren (Python-Paketmanager, bringt Python 3.12 mit)"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
uv --version

Step "Flay installieren (fuer den Datensatz: Qwen3-TTS, Whisper, Piper)"
Push-Location $Repo
uv sync
if ($LASTEXITCODE -ne 0) { Pop-Location; throw "uv sync fehlgeschlagen." }
Pop-Location

Step "Piper-Trainingscode laden (piper1-gpl $($Commit.Substring(0, 7)))"
$Piper = Join-Path $Work "piper1-gpl"
if (-not (Test-Path $Piper)) {
    $Zip = Join-Path $Work "piper1-gpl.zip"
    Invoke-WebRequest "https://codeload.github.com/OHF-Voice/piper1-gpl/zip/$Commit" -OutFile $Zip
    Expand-Archive $Zip -DestinationPath $Work
    Rename-Item (Join-Path $Work "piper1-gpl-$Commit") "piper1-gpl"
    Remove-Item $Zip
}

Step "Trainingsumgebung mit PyTorch (CUDA) anlegen"
if (-not (Test-Path $Py)) { uv venv -p 3.12 (Join-Path $Work "train-venv") }
uv pip install -p $Py torch --index-url https://download.pytorch.org/whl/cu130
uv pip install -p $Py "lightning>=2,<3" "tensorboard>=2,<3" "tensorboardX>=2,<3" `
    "jsonargparse[signatures]>=4.27.7" "onnx>=1,<2" onnxscript "pysilero-vad>=2.1,<3" `
    "librosa<1" numba "onnxruntime>=1,<2" "pathvalidate>=3,<4" huggingface_hub
if ($LASTEXITCODE -ne 0) { throw "Installation fehlgeschlagen." }

Step "Anpassungen fuer Flay einspielen"
& $Py (Join-Path $Here "patch_piper.py") $Piper
if ($LASTEXITCODE -ne 0) { throw "Anpassungen fehlgeschlagen." }

Step "espeak-ng aus dem offiziellen piper-tts-1.8.0-Paket uebernehmen (kein Compiler noetig)"
$Target = Join-Path $Piper "src\piper"
if (-not (Test-Path (Join-Path $Target "espeakbridge.pyd"))) {
    $Wheel = (Invoke-RestMethod https://pypi.org/pypi/piper-tts/1.8.0/json).urls |
        Where-Object { $_.filename -like "*win_amd64.whl" } | Select-Object -First 1
    $WheelZip = Join-Path $Work "piper-tts.zip"
    $WheelDir = Join-Path $Work "piper-tts-wheel"
    Invoke-WebRequest $Wheel.url -OutFile $WheelZip
    Expand-Archive $WheelZip -DestinationPath $WheelDir -Force
    Copy-Item (Join-Path $WheelDir "piper\espeakbridge.pyd") $Target
    Copy-Item (Join-Path $WheelDir "piper\espeak-ng-data") $Target -Recurse -Force
    Remove-Item $WheelZip
    Remove-Item $WheelDir -Recurse
}

Step "Start-Checkpoint Kristin medium laden (846 MB, gemeinfrei)"
$Ckpt = Join-Path $Work "ckpt\en\en_US\kristin\medium\kristin-2000.ckpt"
if (-not (Test-Path $Ckpt)) {
    & $Py -c "from huggingface_hub import hf_hub_download; hf_hub_download('rhasspy/piper-checkpoints', 'en/en_US/kristin/medium/kristin-2000.ckpt', repo_type='dataset', local_dir=r'$Work\ckpt')"
    if ($LASTEXITCODE -ne 0) { throw "Checkpoint-Download fehlgeschlagen." }
}

Step "Selbsttest"
$env:PYTHONPATH = Join-Path $Piper "src"
& $Py -c @"
import torch
assert torch.cuda.is_available(), 'PyTorch sieht keine CUDA-GPU'
print('GPU:', torch.cuda.get_device_name(0), '| PyTorch', torch.__version__)
from piper.phonemize_espeak import EspeakPhonemizer
print('espeak:', ''.join(EspeakPhonemizer().phonemize('de', 'Hallo Welt.')[0]))
from piper.train.vits.monotonic_align import maximum_path
import piper.train.vits.lightning
print('Training importierbar')
"@
if ($LASTEXITCODE -ne 0) { throw "Selbsttest fehlgeschlagen." }
Write-Host "`nFertig. Weiter mit: tools\voice-training\dataset.ps1" -ForegroundColor Green
