# Trainierte Stimmen

| Datei | Stimme | Sprachen | Lizenz |
|---|---|---|---|
| `cassi-ramona-medium.onnx` (+ `.onnx.json`) | Ramona, Piper medium, 22 kHz | Deutsch und Englisch | gemeinfrei (Basis) |

**cassi-ramona-medium:** feinjustiert mit `tools/voice-training/` (90 Epochen, Mac M2 Pro). Das ist weit unter den 2000 Epochen, die Piper für
diese Datenmenge empfiehlt; ein volles Training (`train.ps1`, Windows) klingt deutlich sauberer.
Basis ist der Piper-Checkpoint Kristin (gemeinfrei). Die Trainingsdaten sind rund 950 Sätze
(~1,5 h), gesprochen von Qwen3-TTS (Apache 2.0) mit der Stimme Ramona Deininger-Schnabel
(LibriVox, gemeinfrei). Ein Modell spricht beide Sprachen; Cassi wählt die espeak-Aussprache
je Sprache. Auf der CPU kommt der erste Ton nach ~0,08 s.

Die `.onnx`-Dateien liegen in Git LFS. Ohne LFS enthält der Klon nur Platzhalter:
`git lfs install` und dann `git lfs pull`.

## Verwenden

Cassi nimmt `cassi-ramona-medium` für Piper standardmäßig (`tts.voices`), unter Windows also
immer, auf dem Mac mit `tts.backend: piper`. Beim Start bzw. mit `cassi setup` kopiert Cassi
beide Dateien selbst in den Stimmen-Ordner (Windows `%LOCALAPPDATA%\cassi\voices`, Mac
`~/Library/Application Support/cassi/voices`). Sind hier nur LFS-Zeiger, spricht Cassi mit
Kerstin (de) und Kristin (en) und schreibt einen Hinweis ins Log.
