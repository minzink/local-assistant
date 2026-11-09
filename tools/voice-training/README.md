# Eigene Piper-Stimme trainieren (Windows, NVIDIA)

Trainiert **ein** Piper-Modell, das Deutsch und Englisch mit derselben Stimme spricht – schnell
auf der CPU, ohne Grafikspeicher neben LLM und Whisper. Die Stimme (Standard: Ramona) kommt
von Qwen3-TTS: Es spricht die geprüften Sätze in `src/flay/data/sentences/` (482 deutsch, 483 englisch),
Whisper sortiert Missverstandenes aus. Start ist der Piper-Checkpoint Kristin (weiblich,
gemeinfrei). Alles Erzeugte landet in `tools/voice-training/work/` (nicht im Git).

## Voraussetzungen

- Windows 10/11 mit NVIDIA-GPU (ausgelegt für RTX 4070 Ti, 12 GB) und aktuellem Treiber –
  `nvidia-smi` muss in PowerShell funktionieren
- das Flay-Repo, Internet (~10 GB Downloads), ~20 GB freier Platz
- Python, Compiler usw. sind nicht nötig – das Setup holt alles selbst

## Schritte

Alle Befehle in PowerShell (kein Administrator) im Repo-Ordner, z.B. `cd C:\flay`.

1. **Einrichten** (einmalig, ~20–30 min): installiert uv, Flay (`uv sync`), den
   Piper-Trainingscode und den Start-Checkpoint, am Ende ein Selbsttest.
   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\voice-training\setup.ps1
   ```
2. **Datensatz erzeugen** (~1 h): lädt beim ersten Mal Qwen3-TTS und Whisper.
   Unterbrechen ist kein Problem – erneut starten setzt fort.
   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\voice-training\dataset.ps1
   ```
   Am Ende: `Datensatz: … Sätze, … min`. Erwartet sind gut 900 Sätze, ~80 min.
   Eine andere Stimme: `-Voice hokuspokus`, `-Voice own` (eigene Aufnahme) oder ein WAV-Pfad.
3. **Ruhezustand abschalten**, damit das Training durchläuft:
   ```powershell
   powercfg /change standby-timeout-ac 0
   ```
4. **Trainieren:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\voice-training\train.ps1
   ```
   - Standard: 2000 Epochen (Empfehlung der Piper-Doku für diese Datenmenge), grob ein Tag.
     Das echte Tempo zeigt die erste Epoche (Sekunden je Epoche × 2000).
   - Kürzer: `-Epochs 1000`. Die Zahl vorher festlegen – die Lernrate richtet sich danach.
   - „CUDA out of memory“: `-BatchSize 8`.
   - Abgebrochen? Mit derselben Epochenzahl fortsetzen: `-Resume` (bzw. `-Epochs 1000 -Resume`).
5. **Zwischenstand exportieren** (auch während des Trainings, zweites Fenster):
   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\voice-training\export.ps1
   ```
   Ergibt `work\output\flay-ramona-medium.onnx` und `.onnx.json`. Gespeichert werden der
   neueste Stand (`last.ckpt`) und die 5 besten nach `val_mel` unter
   `work\runs\lightning_logs\version_*\checkpoints\`; einen davon exportiert
   `export.ps1 -Checkpoint <Pfad>`.

## Eigene Stimme aus Aufnahmen

Statt Qwen3-TTS liest du die Sätze selbst – das klingt am meisten nach dir.

1. **Aufnehmen** (Mac oder Windows, ruhiger Raum, immer gleicher Abstand zum Mikrofon):
   ```sh
   uv run flay voice dataset          # beide Sprachen; --language de für nur Deutsch
   ```
   Jede Aufnahme wird gleich geprüft (Stille, Übersteuerung, Tempo); `r` wiederholt, `q`
   beendet – beim nächsten Start geht es beim nächsten Satz weiter. 30–60 Minuten genügen
   für eine brauchbare Stimme, alle 965 Sätze (~80 min) sind besser. Die Aufnahmen liegen im
   Datenordner unter `voice_dataset/` (Mac `~/Library/Application Support/flay`, Windows
   `%LOCALAPPDATA%\flay`); für das Training auf den Windows-Rechner kopieren.
2. **Datensatz** aus den Aufnahmen (Whisper sortiert Versprecher aus, `work\dataset-own`):
   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\voice-training\dataset.ps1 -Recorded <Ordner>
   ```
3. **Trainieren und exportieren** mit eigenem Namen (eigene Ordner, Ramona bleibt unberührt):
   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\voice-training\train.ps1 -Name flay-ich
   powershell -ExecutionPolicy Bypass -File tools\voice-training\export.ps1 -Name flay-ich
   ```
   Start ist auch hier Kristin (weiblich). Für eine tiefe Stimme ist ein männlicher
   Piper-Checkpoint als Start günstiger (z. B. Thorsten, CC0) – dann in `train.ps1` den
   `--model.warmstart_ckpt` darauf zeigen lassen.
4. In `flay.yaml` wie unten, mit `flay-ich-medium` statt `flay-ramona-medium`.

## Mit Git Bash statt PowerShell

Die Skripte bleiben PowerShell-Skripte; Git Bash ruft sie über `powershell.exe` auf
(Pfade mit `/`). Nur `powercfg` braucht `MSYS_NO_PATHCONV=1`, sonst macht Git Bash aus
`/change` einen Dateipfad:

```bash
cd /c/flay   # wo das Repo liegt
powershell.exe -ExecutionPolicy Bypass -File tools/voice-training/setup.ps1
powershell.exe -ExecutionPolicy Bypass -File tools/voice-training/dataset.ps1
MSYS_NO_PATHCONV=1 powercfg.exe /change standby-timeout-ac 0
powershell.exe -ExecutionPolicy Bypass -File tools/voice-training/train.ps1
powershell.exe -ExecutionPolicy Bypass -File tools/voice-training/export.ps1
```

## In Flay verwenden

Ein fertig trainiertes Modell liegt in `voices/` (Git LFS); siehe `voices/README.md`.

1. Beide Dateien aus `work\output\` in den Stimmen-Ordner kopieren:
   Windows `%LOCALAPPDATA%\flay\voices`, Mac `~/Library/Application Support/flay/voices`.
2. In `flay.yaml`:
   ```yaml
   tts:
     backend: piper
     voices:
       de: flay-ramona-medium
       en: flay-ramona-medium
   ```
   Steht dasselbe Modell bei beiden Sprachen, lädt Flay es einmal und spricht jede Sprache
   mit ihrer espeak-Aussprache (`de` bzw. `en-us`).

## Technisches

- `make_dataset.py` läuft in Flays Umgebung: Qwen3-TTS (unter Windows PyTorch/CUDA),
  Whisper, und die Phonem-IDs mit dem Piper, das Flay auch zum Sprechen nutzt. Deutsche
  Sätze werden mit espeak `de`, englische mit `en-us` umgeschrieben – so lernt ein Modell
  beide Sprachen. `metadata.csv`: `Datei|Text|Phonem-IDs`.
- Das Training läuft in einer eigenen Umgebung (`work\train-venv`) mit piper1-gpl
  (Commit 304d5da). `patch_piper.py` passt vier Stellen an:
  1. Die Cython-Ausrichtung ist durch `numba_core.py` ersetzt (gegen das Original geprüft) –
     kein C-Compiler nötig. espeak-ng kommt fertig aus dem piper-tts-1.8.0-Paket.
  2. Der Checkpoint-Speicher nach `val_mos` fällt weg (die Bewertung ist aus, Lightning
     bräche sonst ab).
  3. Der ONNX-Export nutzt den klassischen Exporter (der neue scheitert mit PyTorch ≥ 2.9).
  4. Unter Windows werden Linux-Pfade im Kristin-Checkpoint als Windows-Pfade geladen
     (sonst: „cannot instantiate 'PosixPath' on your system“).
- Lizenzen: Piper-Trainingscode GPL-3.0 (nur Werkzeug, nicht Teil von Flay);
  Kristin-Checkpoint und Ramona (LibriVox) gemeinfrei; Qwen3-TTS Apache 2.0.
