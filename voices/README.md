# Trainierte Stimmen

| Datei | Stimme | Sprachen | Lizenz |
|---|---|---|---|
| `flay-ramona-medium.onnx` (+ `.onnx.json`) | Ramona, Piper medium, 22 kHz | Deutsch und Englisch | gemeinfrei (Basis) |

**flay-ramona-medium:** feinjustiert mit `tools/voice-training/` (90 Epochen, Mac M2 Pro).
Basis ist der Piper-Checkpoint Kristin (gemeinfrei). Die Trainingsdaten sind rund 950 Sätze
(~1,5 h), gesprochen von Qwen3-TTS (Apache 2.0) mit der Stimme Ramona Deininger-Schnabel
(LibriVox, gemeinfrei). Ein Modell spricht beide Sprachen; Flay wählt die espeak-Aussprache
je Sprache. Auf der CPU kommt der erste Ton nach ~0,08 s.

Die `.onnx`-Dateien liegen in Git LFS. Ohne LFS enthält der Klon nur Platzhalter:
`git lfs install` und dann `git lfs pull`.

## Verwenden

1. Beide Dateien in den Stimmen-Ordner kopieren:
   Windows `%LOCALAPPDATA%\flay\voices`, Mac `~/Library/Application Support/flay/voices`.
2. In `flay.yaml`:
   ```yaml
   tts:
     backend: piper
     voices:
       de: flay-ramona-medium
       en: flay-ramona-medium
   ```
