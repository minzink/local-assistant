# Roadmap

## Rahmenbedingungen

| Thema | Entscheidung |
|---|---|
| Plattformen | macOS Apple Silicon (M2 Pro, 32 GB) und Windows (RTX 4070 Ti, 12 GB) |
| Sprachen | Deutsch und Englisch, Erkennung pro Äußerung |
| Eingabe | Push-to-Talk und dauerhaftes Zuhören (VAD); kein Aktivierungswort (kein freies Modell) |
| LLM | Austauschbar, nicht an Ollama gebunden. Standard: beliebiger Server mit OpenAI-kompatibler API |
| Backends | LLM, STT und TTS liegen hinter Protokollen und werden per Config gewählt |
| Multimodal | Pflicht: das auswertende LLM verarbeitet mindestens Text + Bild |
| Lizenzen | Alle Modelle frei für jeden Zweck nutzbar (Apache 2.0, MIT, CC0, CC BY o. ä.); keine Llama-/Gemma-Lizenzen |
| Lokal | Alles läuft lokal; nur `flay setup` lädt herunter, zur Laufzeit gehen keine Daten nach draußen |
| Latenzziel | < 1 s vom Loslassen der Taste bis zum ersten Ton |

## Phase 0 – Aufräumen & Gerüst ✅

- [x] Alte Stände als Tags archiviert, Legacy-Code entfernt
- [x] `pyproject.toml` mit uv, Python 3.12, ruff, mypy (strict), pytest
- [x] Paketstruktur mit Protokollen je Stufe (input, audio, stt, llm, tts)
- [x] Multimodales Nachrichtenmodell + `ModelCapabilities`
- [x] Validierte Konfiguration (`flay.yaml`, `$FLAY_CONFIG`), `flay info`
- [x] CI-Matrix macOS + Windows
- [x] `main` als einzige Linie; `master` (mit privaten Logs) nur noch als lokaler Tag

## Phase 1 – Ende-zu-Ende, lokal (Push-to-Talk) ✅

- [x] Audio-I/O mit `sounddevice`: Mikrofon dauerhaft offen mit 250 ms Vorlauf, Resampling auf
      16 kHz, gestreamte Wiedergabe mit sofortigem Stopp
- [x] Push-to-Talk per globalem Hotkey (pynput) oder Enter im Terminal
- [x] STT: `mlx-whisper` (Apple Silicon) / `faster-whisper` (CUDA/CPU); erkennt de/en, andere
      Sprachen werden in der Standardsprache neu dekodiert
- [x] LLM: OpenAI-kompatibler Streaming-Client, Factories für STT/LLM/TTS
- [x] TTS: Piper mit je einer deutschen und englischen Stimme, automatischer Download
- [x] Orchestrator: Antwort in der erkannten Sprache, Verlauf der letzten 20 Nachrichten,
      neue Äußerung bricht laufende Antwort ab (Wiedergabe + HTTP-Stream)
- [x] Modelle werden beim Start vorgewärmt
- [x] Zeitmessung pro Stufe im Log, `flay bench` für LLM-Latenz
- ~~Benchmark weiterer Server/Modelle (LM Studio/MLX, llama.cpp, mlx-lm) auf Mac und Windows~~ (verworfen bzw. siehe unten)
- ~~Manueller Test auf Windows (Hotkey, faster-whisper mit CUDA)~~ (verworfen bzw. siehe unten)

### Messwerte Phase 1 (M2 Pro, 32 GB)

| Stufe | Messung |
|---|---|
| STT `large-v3-turbo` (mlx), 3 s Sprache | 0,6 s inkl. Spracherkennung (vorher 1,2 s mit `mlx_whisper.transcribe`) |
| LLM `llama3.2` via Ollama | erstes Token 0,09 s, ~230 Zeichen/s |
| TTS Piper `high`, ein Satz (~3 s Audio) | 0,55–0,8 s für den ganzen Satz |
| Ausgabestream öffnen | ~0,3–0,4 s (Funk-Headset) |

Daraus folgt grob 2–2,5 s bis zum ersten Ton. Die größten Hebel für Phase 2 sind:
Satz-Streaming statt kompletter Antwort, ein dauerhaft offener Ausgabestream und ggf. Piper
`medium`-Stimmen.

## Phase 2 – Latenz & Gesprächsfluss ✅ (Ziel < 1 s noch offen)

- [x] LLM → Sätze → TTS → Wiedergabe laufen parallel; der erste Satz wird gesprochen, während das
      LLM weiterschreibt und TTS die folgenden Sätze vorab synthetisiert
- [x] Satz-Splitter mit Abkürzungen/Ordinalzahlen, entfernt Markdown/Emojis; lange Sätze werden
      an Kommas geteilt (erstes Stück ab 60 Zeichen), da kleine Modelle oft einen langen Satz bilden
- [x] Silero VAD (ONNX): schneidet Stille ab, verwirft Aufnahmen ohne Sprache (~10 ms pro Aufnahme)
- [x] Stumme Aufnahmen (< −60 dBFS) mit Hinweis auf Berechtigung verwerfen, `flay mic-test`
- [x] Ausgabestream bleibt offen: 0,3 s Audio in 0,31 s statt 0,73 s, Abbruch in 0,09 s
- [x] Piper `medium` als Standard: erstes Audio ~0,1 s nach dem ersten Satz (`high`: bis 0,7 s)
- [x] Antwortsprache zuverlässig: Hinweis in der aktuellen Nutzer-Nachricht (llama3.2 nach
      Sprachwechsel 10/10 statt 7/10)
- ~~Ziel < 1 s: aktuell 1,1–1,7 s vom Loslassen bis zum ersten Ton~~ (verworfen bzw. siehe unten)

### Messwerte Phase 2 (M2 Pro, llama3.2 via Ollama, Piper medium)

| Stufe ab Loslassen | Zeit |
|---|---|
| VAD | 0,01 s |
| STT | 0,63–0,69 s |
| erstes LLM-Token | 0,80–1,04 s |
| erster Satz fertig | 1,05–1,49 s |
| erster Ton | 1,13–1,67 s |

Größter Block ist jetzt STT. Geprüft und verworfen: Whisper mit gekürztem Audiofenster
(`audio_ctx`) ist 4× schneller, gerät aber in Wiederholungsschleifen. Kandidaten für die nächste
Runde, dank austauschbarem Backend ohne Umbau der Pipeline:

- STT-Alternativen (zurückgestellt, Modelloptimierung erst nach dem Grundgerüst und nur mit
  Windows): Ein erster Mac-Test (FLEURS dev, je 40 Aufnahmen de/en, M2 Pro) ergab:

  | Modell | WER de / en | 2,5-s-Befehl | Sprache erkannt |
  |---|---|---|---|
  | Whisper `large-v3-turbo` (aktuell) | 3,6 % / 3,2 % | 582 ms | 80/80 |
  | Whisper `small` | 9,6 % / 5,0 % | 145 ms | 80/80 |
  | Parakeet TDT 0.6B v3 (MLX) | 4,4 % / 4,3 % | 81 ms | – (liefert keine Sprache) |

  Parakeet wäre ~7× schneller bei kurzen Befehlen, bräuchte aber eine eigene de/en-Erkennung
  und ein Windows-Backend (ONNX/NeMo).
- LLM-Benchmark (Phase-1-Punkt): kleinere Antwortlatenz und besseres Deutsch als llama3.2
- Ollama entlädt Modelle nach 5 min Leerlauf (nächste Antwort dann +2–3 s):
  `OLLAMA_KEEP_ALIVE` setzen oder Server mit dauerhaft geladenem Modell nutzen

## Phase 3 – Persönlichkeit & Gedächtnis ✅

- [x] Persona per Config: System-Prompt, frei benannte Stimmungen (0–10) statt EmotionEncoder,
      aktuelles Datum/Uhrzeit im Prompt
- [x] `Memory`-Protokoll; Implementierungen `PersistentMemory` (SQLite) und `ShortTermMemory` (RAM)
- [x] Sitzungen: nach 30 min Pause neuer Kontext; ältere Nachrichten der Sitzung werden vom LLM
      zusammengefasst, die letzten 12 gehen wörtlich mit
- [x] Langzeit-Fakten: LLM extrahiert/entfernt Fakten im Hintergrund (Prompt mit Beispiel;
      llama3.2 erfasst „merk dir …“ damit zuverlässig), Duplikate werden übersprungen
- [x] Hintergrundpflege nur in Sprechpausen, Abbruch bei neuer Eingabe, Wiederaufnahme später
- [x] `flay memory` (anzeigen, `forget`, `clear`), `memory.enabled: false` schaltet Speichern ab
- ~~Semantische Suche über Fakten (Embeddings), sobald es mehr als ~50 werden~~ (verworfen bzw. siehe unten)
- [x] Gezieltes Merken/Vergessen per Sprachbefehl über Tool-Calling (Phase 4)

## Zwischenschritt – Web-UI ✅

- [x] Lokale Web-UI (aiohttp, nur 127.0.0.1, Origin-Prüfung gegen fremde Webseiten)
- [x] Chatverlauf mit live gestreamten Antworten, Hinweisen und Statusanzeige
      (bereit / hört zu / denkt nach / spricht); Verlauf der Sitzung beim Öffnen
- [x] Texteingabe (Sprache de/en per Wortliste erkannt), ersetzt laufende Antworten wie Sprache
- [x] Getrennte Schalter: Mikrofon stumm (Aufnahmen verworfen), Sprachausgabe stumm
      (Antwort nur als Text, auch mitten in einer Antwort)
- [x] `EventBus` im Assistenten, damit weitere Oberflächen (Tray, Phase 5) andocken können

## Phase 4 – Aktionen auf dem Rechner ✅ (Windows-Test offen)

- [x] Tool-Calling über die OpenAI-kompatible API (gestreamte Aufrufe), bis zu 3 Runden pro Antwort
- [x] `ToolRegistry`: nur Werkzeuge aus `tools.enabled` existieren für das Modell
- [x] `open_project`: PhpStorms zuletzt geöffnete Projekte, unscharfe Namenssuche, Toolbox-Starter
- [x] `send_to_claude`: schreibt in die Claude-Code-Sitzung, die per `flay claude` in einer
      tmux-Sitzung im PhpStorm-Terminal läuft (ohne Fokuswechsel, ohne Enter); sonst
      Zwischenablage + PhpStorm nach vorn. Tastatur-Automation verworfen: das Claude-Plugin
      (0.1.14 beta) reagiert in PhpStorm 2026.2 auf kein Kürzel
- [x] `claude_status`: liest Claudes Terminal (tmux) und fasst den Zustand zusammen
- [x] Sitzungswahl: genanntes Projekt, sonst die im Terminal geöffnete Sitzung (tmux
      „attached“); PhpStorms recentProjects.xml ist dafür zu selten aktualisiert
- [x] Autostart über PhpStorms MCP-Server: temporäres `.flay-claude.sh` per `create_new_file`,
      Start per `execute_run_configuration` (Shell-Scripts laufen in einem echten Terminal-Tab),
      danach löschen. Verworfen: `execute_terminal_command` (kein TTY), `Terminal.NewTab`/„Open
      in Terminal“ (über MCP nicht verfügbar), Plugin-Aktion „Claude Code“ (tut nichts).
      Offene Projekte live vom MCP-Server bzw. aus den Lock-Dateien des Claude-Plugins; nicht
      geöffnetes Projekt → Hintergrund-Sitzung. Vertrauensabfrage wird nie beantwortet
- [x] Claude-Aufträge werden erkannt und in frischem Kontext als Werkzeug-Aufruf erzwungen
      (der Verlauf brachte das Modell sonst zum bloßen Behaupten; Ollama ignoriert `tool_choice`)
- [x] Bildschirm-Erwähnungen hängen den Screenshot direkt an (Modell wählte das Werkzeug nicht
      zuverlässig)
- [x] `look_at_screen`: lokaler Screenshot an das Bildmodell; leeres Bild → Hinweis auf Berechtigung
- [x] Modell auf Qwen3-VL 8B (Apache-2.0) umgestellt, englische Stimme auf Kristin (gemeinfrei)
- [x] Werkzeug-Aufrufe in der Web-UI sichtbar; Debug-Log-Fenster (Button unten rechts) mit allen
      Logs, Aktionen hervorgehoben und filterbar
- [x] Ollama auf 0.40.11 aktualisiert, `qwen3-vl:8b-instruct` geladen (die Standard-Variante
      `qwen3-vl:8b` denkt immer erst 6–10 s, auch mit `think=false`)
- [x] Werkzeuge im echten Betrieb getestet (macOS): Projekt öffnen, Bildschirm, Auftrag an
      Claude inkl. Öffnen des PhpStorm-Terminals
- [x] Claude-Eingabe wird vor dem Tippen gelesen: nur leere Eingabe oder eigener, nicht
      abgeschickter Text von Flay wird beschrieben; fremder Text bleibt unangetastet
- ~~Windows: Zwischenablage-Weg und Fensteraktivierung testen~~ (verworfen bzw. siehe unten)
- [x] Merken/Vergessen als Werkzeuge (`remember`, `forget`), per Satz erkannt und erzwungen;
      so behandelte Gesprächszüge werden nicht noch einmal per Hintergrund-Extraktion ausgewertet
- [x] Offline-Garantie: Downloads nur über `flay setup`; im Betrieb strikt offline (geprüft in
      einer macOS-Sandbox ohne Internet), Warnung bei nicht-lokalem LLM

## Phase 5 – Dauerhaftes Zuhören ✅

- [x] `ContinuousListener`: Mikrofon-Strom mit Silero VAD (Block für Block) in Äußerungen zerlegt;
      Vorlauf 300 ms, Satzende nach 600 ms Pause, `flay run --continuous` bzw. `input.mode`
- [x] Unterbrechen per Sprache (Barge-in): während Flay spricht, zählt nur klare, anhaltende
      Sprache; `input.continuous.barge_in: false` ohne Headset (Flay hört sich sonst selbst)
- [x] Menüleisten-/Tray-Symbol (pystray): Zustand als Farbe, Mikrofon/Sprache stumm, Web-UI,
      Beenden; auf macOS läuft Flay dafür in einem Arbeitsthread
- [x] Autostart: `flay autostart on|off|status` (macOS LaunchAgent, Windows Autostart-Ordner)

- [x] Dauerhaftes Zuhören zur Laufzeit umschaltbar (Web-UI, Menüleiste); Push-to-Talk bleibt aktiv
- [x] Projekt schließen (`close_project`, PhpStorm-Aktion `CloseProject` über MCP, Erfolg geprüft);
      Öffnen/Schließen per Satz erkannt; Hinweis auf PhpStorms „Open project in: New window“

## Phase 6 – Multimodal (Teilumfang) ✅

- [x] Gezielter Bildausschnitt: aktives Fenster, Fenster einer App, bestimmter Monitor – direkt aus
      dem Satz erkannt; Berechtigung „Bildschirmaufnahme“ wird vorab bei macOS abgefragt (ohne sie
      liefert macOS nur Hintergrund und Menüleiste)

## Antwortzeit

- [x] Modell warm halten: Mini-Anfrage mit dem üblichen Prompt nach 4 min Leerlauf
- [x] Erstes Satzstück schon ab 40 Zeichen an die Sprachausgabe
- [x] Vorab-Erkennung beim dauerhaften Zuhören: Transkription startet bei der ersten Pause
- Gemessen (M2 Pro, Qwen3-VL 8B, synthetische Sprache): Push-to-Talk 1,35–1,4 s vom Loslassen
  bis zum ersten Ton (Whisper ~0,6 s); dauerhaftes Zuhören 1,55 s vom Sprechende (vorher 2,1 s)
- [x] Parakeet TDT 0.6B v3 (ONNX, CC BY 4.0, onnx-asr) als Standard-Spracherkenner, auf der CPU
      (schneller als CoreML: 70 vs. 168 ms pro Satz); Sprache de/en aus dem Text
- Ergebnis mit Parakeet: Push-to-Talk **0,85 s** (Median), dauerhaftes Zuhören 1,22 s vom
  Sprechende (davon 0,6 s Warten auf das Satzende). Whisper bleibt per `stt.backend` wählbar
- [x] Füllwort („Mhm.“, „Moment.“), wenn die Antwort nach 0,4 s noch nicht läuft (`tts.filler`)
- [x] Unterbrechungen bleiben im Kontext: gespeichert wird nur das Gesagte, das Modell sieht,
      wo es unterbrochen wurde; auch vor der Antwort abgebrochene Fragen bleiben erhalten

## Phase 7 – Zuverlässigkeit der Aktionen

Anlass: Flay führte zu oft falsche Aktionen aus oder behauptete Erfolge, die nicht stimmten.
Ursachen aus echten Gesprächen: das Modell formulierte Werkzeug-Ergebnisse frei (und falsch)
um, der Verlauf enthielt nur Text statt der tatsächlichen Aktionen, eigene falsche Antworten
wurden nachgeahmt, und die Spracherkennung verstümmelte Projektnamen und Befehlsverben.

### Schritt 1 – ohne neue Modelle

- [x] Feste Rückmeldungen für Aktionen: Werkzeuge liefern den gesprochenen Satz selbst (de/en);
      bei Aktionen formuliert das Modell nicht mehr um – richtiger und spart einen Modell-Durchlauf
- [x] Aktions-Protokoll im Verlauf: ausgeführte Werkzeuge und ihr Ergebnis werden gespeichert und
      dem Modell in späteren Runden als echte Werkzeug-Aufrufe gezeigt
- [x] Fehlgeschlagener Screenshot wird fest beantwortet statt vom Modell „beschrieben“; ein
      Fenster, das gerade die Flay-UI zeigt (Firefox-Tab), zählt bei ausdrücklicher Nennung mit
- [x] Mehrdeutiger Projektname beim Claude-Auftrag: Rückfrage statt stiller Zwischenablage
- [x] Spracherkennung: Korrektur gegen das eigene Vokabular (Projektnamen, Fachbegriffe);
      unscharf nur mit Projekt-Kontext, damit normale Wörter bleiben
- [x] Geräusch-Filter beim dauerhaften Zuhören (einzelne bedeutungslose Wörter verwerfen)
- [x] Aktueller Zustand in der Anfrage: offene Projekte, laufende Claude-Sitzungen (parallel zur
      Spracherkennung gelesen, höchstens 0,3 s Wartezeit, nur in der neuen Nachricht)

### Schritt 2 – Modell und Server ✅ (Mac; Windows-Messung offen)

- [x] Testsatz `eval/cases.yaml` (38 Fälle, viele aus echten Gesprächen inkl. Erkennungsfehlern)
      und `flay eval`: echter Assistent mit simulierten Werkzeugen, misst Trefferquote, falsche
      Aktionen, Rückfragen und Zeit bis Werkzeug/erster Antwort
- [x] Qwen3.5-9B (Q4_K_M, llama.cpp) gegen qwen3-vl:8b-instruct (Ollama) auf dem M2 Pro,
      je 38 Fälle × 3:

      | | qwen3-vl:8b-instruct (Ollama) | Qwen3.5-9B (llama.cpp) |
      |---|---|---|
      | Trefferquote | **93,9 %** | 90,4 % |
      | falsche Aktionen | **1** | 3 |
      | Gespräch ohne Aktion korrekt | **27/27** | 25/27 |
      | Zeit bis Werkzeug (Median) | **1,04 s** | 2,58 s |
      | Zeit bis erste Antwort (Gespräch) | 0,52 s | 0,58 s |

      Qwen3.5 mit Denken auf den schweren Fällen: nicht besser (60 % statt 70 %), dafür 3–13 s.
- [x] Entscheidung: qwen3-vl:8b-instruct auf Ollama bleibt Standard. llama.cpp bringt mit
      Qwen3.5 keinen Vorteil (langsamer, Hybrid-Architektur); per Config weiter möglich
      (`llm.extra_body` für server-spezifische Felder). Testmodell und llama.cpp wieder entfernt
- [x] Durch den Testsatz gefunden und behoben (modellunabhängig):
      - Ein-Satz-Behauptungen („Ich schließe das Projekt …“) rutschten am Schutz vorbei
      - System-Notiz mitten im Gespräch brach Qwen3.5-Vorlagen (HTTP 500)
      - Projekt-Aktionen nur mit einem Projekt, das der Nutzer (unscharf) genannt hat; sonst
        Rückfrage („Mach mal Moewe auf“ öffnete ein anderes, gerade offenes Projekt)
      - Zurückgenommene Anfragen („…, never mind“) lösen nichts aus
      - „Wie schließe ich ein Projekt?“ ist eine Frage, keine Aktion
      - Ergebnis qwen3-vl: 88,6 % → 93,9 %, Gespräch 23/27 → 27/27
- [ ] Gleiche Messung auf Windows (RTX 4070 Ti): `uv run flay eval --repeat 3`

### Schritt 3 – Weiterleitung und Denken ✅

- [x] Weiterleitung per Formular (JSON-Schema, `{request, action, project, text}`, Projekte nur
      aus der eigenen Liste) gebaut und gemessen – mit qwen3-vl:8b-instruct nicht besser:

      | Variante (38 × 3) | Treffer | falsche Aktionen | Zeit bis Werkzeug |
      |---|---|---|---|
      | Werkzeug-Aufruf + Plausibilitätsprüfung (bisher) | 93,9 % | 1 | 1,04 s |
      | Formular für alles mit Projektnamen | 89,5 % | 12 | 1,51 s |
      | Formular nur nach erkannter Aktion | 92,1 % | 2 | 1,70 s |

      Das Formular behebt keine Verständnisfehler (verstümmelte Verben → „open“) und nimmt dem
      Modell die Rückfrage. Bleibt abschaltbar (`llm.structured_routing`, `flay eval
      --structured-routing`) für spätere Modelle
- [x] Rückfrage vor dem Schließen, wenn das Projekt in den Worten nur ungefähr vorkommt
      („KEWO“ → „Meinst du project symfony kx ewo?“); „ja“ führt aus, „nein“ bricht ab, alles
      andere verwirft die Frage. Ergebnis: **94,7 %, keine falsche Aktion** (9 Rückfragen)
- [x] Optionales Denken (`llm.thinking`): nur auf ausdrücklichen Wunsch („denk mal nach“,
      „überleg …“, „Schritt für Schritt“); zuerst „Moment, ich überlege kurz.“, Denktext wird nie
      gesprochen (eigenes Feld oder `<think>`-Tags). Aus, solange kein Modell mit Denk-Modus
      eingerichtet ist
- [x] Denken gemessen mit qwen3-vl:8b-thinking (Apache 2.0, Text + Bild) auf
      `eval/thinking.yaml` (10 Rechen-, Zeit- und Logikaufgaben × 2):

      | | ohne Denken (8b-instruct) | mit Denken (8b-thinking) |
      |---|---|---|
      | richtig | 75 % | **100 %** |
      | Antwort fertig (Median) | 1,5 s | 13,5 s (7–30 s) |
      | erste Rückmeldung | 0,4 s | sofort („Moment, ich überlege kurz.“) |

      Ollama hält mit dem Standard-Kontext (32k, je 10 GB) nicht beide Modelle im Speicher;
      zurück zum normalen Modell kostet dann 5,8 s. Mit 8k Kontext (je 6,1 GB) bleiben beide
      geladen und der Wechsel kostet nichts (Ollama-Einstellung „Context length“ bzw.
      `OLLAMA_CONTEXT_LENGTH=8192`; Flays Prompts brauchen < 8k). Zusätzlich lädt Flay das
      normale Modell nach einer Denk-Antwort sofort im Hintergrund nach
- [ ] Windows (12 GB VRAM): zwei 8B-Modelle passen nicht gleichzeitig – dort kostet Denken
      den Modellwechsel; messen

### Schritt 4 – Spracherkennung vergleichen (läuft)

- [x] Testsatz `eval/speech.yaml` (25 Sätze, de/en, mit Pflichtbegriffen: Verben, Projektnamen,
      Fachbegriffe), `flay record` zum Einsprechen, `flay eval-stt` misst Wortfehlerrate,
      Begriffe, Spracherkennung und Zeit
- [x] Canary-1B-v2 (CC BY 4.0) als Backend `canary`; Whisper mit Begriffsvorgaben
      (`stt.vocabulary_prompt`)
- [x] Vorab mit synthetischer Sprache (Piper – spricht Namen selbst undeutlich, nur grob):

      | Modell | Wortfehler | Begriffe | Zeit/Satz |
      |---|---|---|---|
      | Parakeet TDT 0.6B v3 | 21,6 % | 42/54 | **0,07 s** |
      | Whisper large-v3-turbo | **17,9 %** | **44/54** | 0,60 s |
      | Whisper + Begriffsvorgaben | 18,5 % | 44/54 | 0,61 s |
      | Canary-1B-v2 | 33,8 % | 39/54 | 0,16 s |
      | Voxtral Mini 4B Realtime (4-bit, MLX) | 24,9 % | 36/54 | 2,26 s |

      Canary braucht die Sprache vorab und *übersetzt* englische Sätze, wenn es Deutsch
      erwartet – für gemischtes Deutsch/Englisch ungeeignet
- [x] Mit echten Aufnahmen verglichen (eigene Stimme, 25 Sätze, M2 Pro); „nach Korrektur“ = nach
      Flays Abgleich mit dem eigenen Vokabular, also das, womit Flay arbeitet:

      | Modell | Wortfehler | Begriffe nach Korrektur | Sätze komplett | Zeit/Satz |
      |---|---|---|---|---|
      | Parakeet TDT 0.6B v3 | 16,1 % | 46/54 | 16/25 | **0,08 s** |
      | Whisper large-v3-turbo | 11,8 % | 50/54 | 19/25 | 0,59 s |
      | **Whisper + Begriffsvorgaben** | **10,8 %** | **51/54** | **20/25** | 0,61 s |
      | Voxtral Mini 4B Realtime | 14,4 % | 50/54 | 20/25 | 2,46 s |
      | Canary-1B-v2 | 34,9 % | 41/54 | 14/25 | 0,16 s |

      Bei allen Modellen wurde „Sag Claude“ zu „Sagt Lord/Lot“ – jetzt korrigiert (nach
      „sag/frag/schreib/tell“ gilt „Lord“, „Lot“, „Claw“ als Claude). Canary und Voxtral
      wieder gelöscht
- [x] Umgestellt auf Whisper + Begriffsvorgaben (`stt.backend: auto`, `stt.vocabulary_prompt`)
- [x] Nachbearbeitung der Erkennung durch das Sprachmodell (qwen3-vl:8b, mit Vokabular) getestet
      und verworfen: Parakeet 46 → 49/54 Begriffe, Whisper 51 → 50/54, je +0,5 s, und
      Überkorrektur („Mach mal Moewe auf“ → „PhpStorm“, „Flay“ → „nexo-cms“, Englisch wird
      übersetzt)
- [x] Modell mit Audio-Eingang getestet: Gemma 4 E4B (Apache 2.0, Text + Bild + Audio + Tools,
      9,5 GB über Ollama; Audio nur über `input_audio` der OpenAI-Schnittstelle):

      | Verwendung | Ergebnis | Vergleich |
      |---|---|---|
      | Spracherkenner mit Begriffen als Kontext | 7,8 % WER, 50/54 Begriffe, 0,60 s | Whisper + Begriffe: 10,8 %, 51/54, 0,61 s |
      | Sprachmodell für Aktionen (Text) | 88,6 %, 5 falsche Aktionen, Werkzeug nach 0,49 s | qwen3-vl: 94,7 %, 0 falsche, 1,22 s |
      | Audio direkt → Aktion | 6/25 – versteht, beschreibt die Aktion aber nur in Worten | – |

      Kein Vorteil gegenüber Whisper + qwen3-vl, deutlich größer (passt auf Windows nicht neben
      das Sprachmodell), wieder gelöscht. Erkenntnis: Kontext („Entwickler spricht mit einem
      Assistenten für PhpStorm/Claude, Namen: …“) hilft einem hörenden Modell stark
- [x] Voxtral Mini 4B Realtime (Apache 2.0) als Backend `voxtral` (nur Mac, mlx-audio – MIT –
      nicht in den Abhängigkeiten: `uv run --with "mlx-audio[stt]" …`): nicht genauer und mit
      ~2,3 s pro Satz viel zu langsam für Flay

## Phase 8 – Langzeit-Gedächtnis (RAG) ✅

- [x] Erinnerungen statt Faktenliste: Fakten (über Nutzer, Projekte, Entscheidungen, Termine
      mit absoluten Daten) und Episoden (Zusammenfassung jedes beendeten Gesprächs, aus dem
      Gespräch selbst, dritte Person, Aktionen nur mit Beleg)
- [x] Hybride Suche je Anfrage: Embeddings (`qwen3-embedding:0.6b`, Apache 2.0, über Ollama,
      austauschbar über jede OpenAI-kompatible `/embeddings`-Schnittstelle) + Stichwörter
      (SQLite FTS5), zusammengeführt per Reciprocal Rank Fusion; nur Lokales, keine neue
      Abhängigkeit (Vektoren als BLOB in SQLite, Suche mit numpy)
- [x] Schwellen aus echten Daten: Fakten ab 0,30 und nahe am besten Treffer, Episoden ab 0,50;
      „letztes Mal“ holt die jüngste Episode. Suche ~20–30 ms, parallel zur Weiterleitung, max.
      0,3 s Wartezeit; ohne Embedding-Modell nur Stichwörter
- [x] Erinnerungen in der neuen Nachricht statt im System-Prompt: der bleibt gleich und kann
      vom Server zwischengespeichert werden
- [x] Jede Erinnerung mit Schreibdatum und Wichtigkeit (wichtig / normal / nebensächlich):
      wichtige (Regeln, Anweisungen, ausdrücklich Wichtiges) stehen immer im System-Prompt und
      in der Weiterleitung (Aufträge an Claude), nebensächliche (einmalige Ergebnisse,
      Gesprächszusammenfassungen) zählen beim Suchen weniger. Bewertet von der Extraktion und
      beim Merken; Formulierungen wie „in Zukunft“, „nie“, „immer“, „wichtig“ machen eine
      Erinnerung immer wichtig
- [x] `flay memory search`, `flay setup` prüft das Embedding-Modell
- [x] Nebenbei gefunden: der Wechsel vom Füllwort zur Antwort verschluckte eine Unterbrechung
      (Abbruch des Gesprächszugs) – behoben

## Phase 9 – Lernen aus dem Verlauf

- [x] Beispiele: erfolgreiche, nicht widersprochene Aktionen werden als „Satz wie gehört →
      Aktion“ gespeichert, „nein/falsch“ verwirft sie. Ähnliche Beispiele (nur nach Bedeutung,
      ab 0,43 – Gespräch erreichte höchstens 0,41, verstümmelte Befehle mindestens 0,46) gehen an
      Weiterleitung und Modell; bestätigte oder zweimal gleiche Sätze werden direkt ausgeführt
- [x] Gemessen mit 26 Beispielen aus den eigenen Aufnahmen (andere Sätze als im Testsatz),
      qwen3-vl:8b-instruct, 38 × 2:

      | | ohne Beispiele | mit Beispielen |
      |---|---|---|
      | Treffer | 92,1 % | **98,7 %** |
      | Aktionen richtig | 53/58 | **58/58** |
      | Rückfrage statt Aktion | 5 | **0** |
      | erste Antwort (Gespräch) | 0,52 s | 0,57 s |
      | Zeit bis Werkzeug | 1,23 s | 1,56 s |

- [x] `flay train-export`: Trainingsdaten für einen LoRA-Adapter aus den Beispielen
- [x] Korrekturen („nein, falsch“) als Fehler aufbewahren – nur Trainingsdaten (Antwort: nachfragen), nie im Prompt
- [x] Stand der Sammlung beim Start von `flay run` und in `flay memory` (LoRA lohnt sich ab etwa 300 Beispielen)
- [x] Fix: ein im Gespräch schon benutztes Werkzeug wurde für spätere Anfragen nicht mehr angeboten (nur der Behauptungs-Schutz rettete es, teils mit ~4 s)
- [x] Erzählungen („hab flay gestern geöffnet“) bekommen keine Befehlsbeispiele und keine Projektaktionen; `flay eval` prüft Folgeanfragen mit ausgeführten Aktionen in der Vorgeschichte
  (44 Fälle × 2, qwen3-vl:8b-instruct: ohne Beispiele 93,2 %, mit Beispielen 100 %, Gesprächssätze 24/24)
- [x] Verben lernen (`flay/verbs.py`): aus „She K Command“ → Schließen wird „she“ als Schließen-Verb für alle Projekte gelernt (nur kurze Bitte + Projektname)
- [x] Trennbare Verben: erstes Wort + Endung am Satzende (zu/auf/aus/an/ab/ein/weg/up/down/off) als ein Verb („mach … zu“ ≠ „mach … auf“); „mach“ allein wird nie gelernt
  (KX-Testsatz 6 Fälle × 3: ohne gelernte Verben 3/18 mit 3 falschen Aktionen, mit 18/18; voller Testsatz mit Beispielen 100 %)
- [x] Lernen nach Rückfrage: führte eine kurze Bitte zu keiner Aktion und öffnet/schließt die nächste (≤ 60 s) dasselbe Projekt, wird die erste mitgelernt – samt Verb („Sli flay“ → „Schließe flay“ lehrt „sli“); ein „nein“ verwirft beides
- [x] Buchstabierte Namen: „K K P F B“ findet kx-kpfb (ein verschluckter Buchstabe zählt fast wie exakt, ein klar besserer Treffer gewinnt allein)
  (KX NVS/KPFB/KPF nach einmal gelerntem „She/Slie“: 9/9 statt 0/9, jeweils mit Rückfrage „Meinst du …?“, da der Name nur ungefähr gehört wurde)
- [x] LoRA-Adapter getestet (alles frei: Qwen3-VL-8B Apache 2.0, mlx-vlm MIT; 4-bit, Rang 8,
      3 Epochen auf den 26 Beispielen, 28 min auf dem M2 Pro, 9,4 GB Spitze, Adapter 87 MB):

      | über mlx-vlm-Server, 38 Fälle | ohne Adapter | mit Adapter |
      |---|---|---|
      | Treffer | 94,7 % | 94,7 % |
      | falsche Aktionen | **0** | 2 |
      | Zeit bis Werkzeug | 2,7 s | 3,4 s |
      | erste Antwort | 5,3 s | 5,6 s |

      Kein Gewinn: mit so wenigen Beispielen eher auswendig gelernt (Trainingsfehler ~0,
      Prüffehler 0,55), aus Rückfragen wurden falsche Aktionen. Der mlx-vlm-Server ist zudem
      viel zu langsam (lauscht standardmäßig auf 0.0.0.0 – nur mit `--host 127.0.0.1`).
      Beispiele im Prompt sind besser und schnell. Wieder aufnehmen, wenn Hunderte echter
      Beispiele vorliegen und eine schnelle Ausführung mit Adapter möglich ist (llama.cpp
      `--lora` oder eingerechnet als GGUF in Ollama); Modell und Adapter wieder gelöscht

## Phase 10 – Mehr Werkzeuge

- [x] `ssh_connect`: „SSH auf …“ / „Verbinde dich mit dem Server …“ öffnet ein neues
  Terminalfenster mit `ssh <host>`; Server nur aus der ssh-Config (inkl. `Include`, ohne
  Platzhalter), Abgleich über Alias und HostName, nie ein ungenannter Server, nur der Alias
  geht an ssh (keine Optionen, keine Shell)
- [ ] `flay eval`: Fälle für `ssh_connect` (braucht simulierte Server-Liste)
- [ ] Windows: Windows Terminal / Konsole ausprobieren

## Phase 11 – Eine Stimme für beide Sprachen

- [x] Qwen3-TTS 0.6B (Apache 2.0) klont eine Stimme für Deutsch und Englisch; M2 Pro mit
  MLX 8 Bit (jetzt 4 Bit, gleich gut verständlich): erster Ton ~0,2 s, ~0,4 s Rechenzeit je Sekunde Sprache (PyTorch auf dem Mac:
  1,4–2,0 s je Sekunde, zu langsam)
- [x] Stimmen: eigene Aufnahme (`flay voice record`, je Sprache eine), sonst Ramona bzw.
  Hokuspokus (LibriVox, gemeinfrei), sonst Thorsten (CC0). Eine englische Referenz (Kristin) gab
  dem Deutschen einen englischen Akzent – deshalb deutsche Sprecherinnen
- [x] `tts.voice.pitch` (Halbtöne) und `tts.voice.speed`: einmal auf die Referenz angewendet
  (WSOLA); gemessen +3 Halbtöne → 174 → 207 Hz, Tempo 1,2 → ~15 % kürzer
- [ ] Windows: Qwen3-TTS mit PyTorch/CUDA messen (12 GB VRAM neben LLM und Whisper)
- [x] Hörtest durch den Nutzer: Ramona ist Standard
- [x] Erster Satzteil geht schon am ersten Komma an die Sprachausgabe
- [x] Piper ohne GPU: deutsche Stimme Kerstin (CC0, weiblich) statt Thorsten, passend zu
  Kristin; Kerstin gibt es nur mit 16 kHz, Piper rechnet auf die gemeinsame Rate (22 kHz) um
- [ ] Hörtest Kerstin/Kristin durch den Nutzer
- [ ] Langsam im Betrieb: Satz bis Ton ~1,2 s statt ~0,45 s, wenn der GPU-Speicher knapp wird
  (Ollama: LLM ~10 GB und Embedding-Modell ~4,8 GB, beide mit 32k Kontext, dazu Whisper
  und Qwen3-TTS)
- [x] Eigene Piper-Stimme aus Ramona trainiert (`tools/voice-training/`, Basis Kristin):
  ein Modell für Deutsch und Englisch, 90 Epochen auf dem M2 Pro; erster Ton ~0,08 s auf
  der CPU, Whisper-Fehlerquote 7,6 % (Kerstin/Kristin 5,3 %); liegt in `voices/` (Git LFS)
- [ ] Windows-Training (2000 Epochen voreingestellt; für eine Feinjustierung reichen
  vermutlich einige hundert) und Hörvergleich mit der Mac-Version

## Phase 12 – Bildschirm steuern und Webseiten lesen

Prototyp (M2 Pro, qwen3-vl:8b-instruct über Ollama, Prüfung über die macOS-
Bedienungshilfen: Rahmen jedes Elements und Element unter dem Cursor):

- [x] Koordinaten: Qwen3-VL antwortet mit `{"point_2d": [x, y]}`, 0–1000 relativ zum Bild;
  Bildschirmpunkt = Ausschnitt-Ursprung + x/1000 × Breite (unabhängig von Verkleinerung
  und Retina, mss liefert Monitore in Bildschirmpunkten)
- [x] Treffer auf 43 Elementen in PhpStorm: sichtbarer Text 18/20 direkt, 20/20 mit
  Nachschärfen (zweiter Blick auf einen Ausschnitt um den ersten Treffer); Symbole nur mit
  Tooltip-Namen 6/23 – Symbole müssen beschrieben werden („Glocke rechts oben“)
- [x] 6/6 echte Rechtsklicks (pynput) auf dem richtigen Element, Menü ging jeweils auf
- [x] Dauer untersucht: ein neues Bild kostet ~8 s, egal wie groß – Ollama (llama-server)
  macht aus jedem Bild ~1.040 Tokens (Mindestgröße für genaue Ortung); ~3,4 s Bild-Encoder,
  ~3,8 s Einlesen der Bild-Tokens (~270 Tokens/s, Grenze des M2 Pro für 8B Q4), 0,5 s
  Antwort. Nicht der Speicher (gleich schnell ohne Embedding-Modell). Dasselbe Bild erneut
  gefragt: ~0,8 s (Prompt-Cache)

Umsetzung:

- [x] Schneller orten: erst die macOS-Bedienungshilfen (Beschriftung + Rahmen, App für App,
  vorderste zuerst; 0,2–0,3 s je Klick statt 8–16 s), Qwen3-VL nur ohne Beschriftung, bei
  mehreren Treffern oder auf Windows. Steht die Beschriftung nur in der Beschreibung („UTF-8
  unten in der Statusleiste“), wird sie dort gefunden
- [x] Kleinere Modelle gemessen (44 Elemente, direkt / nachgeschärft; neues Bild):

  | | Text (Menü-, Statusleiste) | Symbole u. a. | neues Bild |
  |---|---|---|---|
  | qwen3-vl:8b-instruct | 16 / 19 von 22 | 3 / 3 von 22 | 7,8 s |
  | qwen3-vl:4b-instruct | 19 / 19 von 22 | 3 / 5 von 22 | 4,1 s |
  | qwen3-vl:2b-instruct | 18 / 19 von 22 | 0 / 6 von 22 | 2,5 s |

  4B ist genauer und doppelt so schnell – aber Ollama hält es nicht neben dem 8B (32k
  Kontext: 9,9 + 8,2 + Embedding 4,8 GB übersteigen den nutzbaren Speicher) und entlädt
  bei jedem Klick das Gesprächsmodell (+3 s Laden, +2,6 s bis zur nächsten Antwort).
  Deshalb bleibt `tools.pointer.model: null` (= 8B); Testmodelle wieder gelöscht
- [x] `click_on_screen(text, description, button, window)`: das LLM nennt nur Beschriftung
  bzw. Beschreibung; frischer Screenshot, eigener kurzer Aufruf zum Orten, Nachschärfen nur,
  wenn die Bedienungshilfen den Punkt nicht bestätigen (`tools.pointer.refine`); links,
  rechts, doppelt; Rückmeldung, was getroffen wurde. Echte Rechtsklicks: „Git“-Menü und
  „UTF-8“ über die Bedienungshilfen getroffen; „Glocken-Symbol rechts“ nur nach Bild
  daneben (Statusleiste statt Seitenleiste)
- [x] Sicherheit: nie in Flays eigenes Fenster; Linksklick auf „Löschen“, „Senden“, „Push“ u. ä.
  erst nach „ja“ (`ToolResult.confirm`); fehlende Berechtigung „Bedienungshilfen“ wird erklärt
  und von macOS abgefragt
- [x] `scroll_screen(direction, pages, window, look)`: Mausrad über dem aktiven oder genannten
  Fenster, 85 % einer Seite je Seite; auf Wunsch Screenshot danach
- [x] `open_url(url)`: Link im Standardbrowser
- [x] `read_webpage(url, pages)`: öffnen, warten bis der Bildschirm steht, Bildschirm für
  Bildschirm aufnehmen und weiterscrollen, bis sich nichts mehr ändert (Seitenende; ein
  blinkender Cursor zählt nicht) oder `tools.web.max_pages` (3) erreicht ist
- [x] Kette „Öffne diesen Link https://www.google.com/, fasse den Inhalt zusammen und gib es
  an Claude“: steuert der Assistent selbst (`Assistant._web`) – Zwischenmeldung „Ich öffne
  die Seite und lese sie.“ läuft, während gelesen wird; Zusammenfassung aus den Screenshots
  → `send_to_claude` (ohne Enter). Echt gemessen: 15,7 s. Ohne „an Claude“ antwortet das
  Modell aus den Screenshots. Werkzeugrunden 3 → 5; langsame Werkzeuge kündigen sich an,
  wenn das Modell sie selbst aufruft (`announcement`)
- [x] Links: getippt, gesprochen („heise punkt de“, „python dot org“), aus der Zwischenablage
  oder aus Claudes letzter Antwort; fehlt einer, fragt Flay nach
- [x] `flay eval`: 11 neue Fälle (Klicken, Scrollen, Webseiten inkl. Kette mit `then`),
  22/22; voller Satz mit Beispielen 100 % (84/84 Aktionen, 26/26 Gespräche)

Offen:

- [ ] Windows: UI Automation statt Bild für beschriftete Elemente (wie die Bedienungshilfen);
  DPI-Bewusstsein ist gesetzt, Klicks und Scroll-Schrittweite (Mausrad-Raster) auf dem
  Windows-Rechner prüfen; Bild-Orten mit RTX 4070 Ti messen
- [ ] Symbole ohne Beschriftung treffen das Bild-Orten selten; Ideen: Tooltip-Namen aus den
  Bedienungshilfen als Liste mitgeben, Kandidaten im Bild markieren und wählen lassen
- [ ] Ortungsmodell 4B nutzbar machen: mit kleinerem Kontext (eigene Ollama-Variante) passt
  es neben das 8B – zusammen mit dem offenen Speicherpunkt aus Phase 11
- [ ] Hörtest der Kette durch den Nutzer; Webseiten mit Cookie-Bannern und Logins

### Aufgaben auf Webseiten

Ziel: „Schick Claude die Installationsschritte von <Link> als Aufgabe“ – Flay öffnet die
Seite, scrollt und klickt, bis die gewünschte Information zu sehen ist, und gibt sie Claude
als Aufgabe.

- [x] `browsing.py`: Schleife aus Ansehen → eine Aktion (`scroll_screen` oder
  `click_on_screen`, nur im Browserfenster) → neuer Screenshot, höchstens
  `tools.web.max_steps` (4) Schritte; jeder Schritt wird angesagt („Ich klicke auf
  Installation.“). Danach schreibt das Modell das Ergebnis: für Claude eine Aufgabe im
  Imperativ mit allen Details von der Seite (Befehle, Versionen exakt) plus Quelle, sonst
  die gesprochene Antwort. Wichtige Regeln des Nutzers gelten auch hier
- [x] Ansehen und Schreiben sind zwei Züge desselben Gesprächs: in einem Zug gab das 8B zu
  früh auf („nicht auf der Seite“) – Eval 3/7 → 7/7. Der Schreib-Zug hängt nur eine
  Nachricht an, die Screenshots bleiben im Prompt-Cache
- [x] Schutz im Code statt nur im Prompt: keine Klicks auf Suche, Logo oder Anmelden (das
  8B klickte „Google Suche“, um Google zu „lesen“); Scrollen ohne Bewegung meldet das
  Seitenende; riskante Klicks werden mitten in der Aufgabe nicht gefragt, sondern
  ausgelassen
- [x] „diese Seite“: liest die im Browser offene Seite ohne Link (ab der aktuellen
  Position); ohne Browser vorn geht die Anfrage normal ans Modell
- [x] „schick/übergib/leite … weiter an Claude“ zählt nur bei Webseiten als Übergabe –
  allgemein brauchte „Schick das Gleiche an Claude“ den Verlauf (Eval-Fall fiel sonst um)
- [x] Riskante Beschriftungen nur am Wortanfang („Weitere Informationen“ ist nicht „format“),
  dazu Warenkorb, Kasse, Abonnieren, Registrieren
- [x] `flay eval`: 3 neue Fälle, Webseiten 14/14; voller Satz 96,6 %, wie vorher nur die
  genuschelten Befehle ohne Beispiele daneben
- [ ] Echter Test auf einer Doku-Seite (z. B. Installation hinter einem Menüpunkt) und
  Zeitmessung: jeder Schritt mit neuem Bild kostet ~8 s
- [ ] Mehrere Seiten in einer Aufgabe („vergleiche …“), Text eintippen (Suche auf der Seite)

### Webseiten als Text lesen (Mischform)

Prototyp ohne Modell (M2 Pro, Firefox als Standardbrowser):

| Weg | Seite lesen | Link „klicken“ | Logins/JS | Bildschirm |
|---|---|---|---|---|
| Screenshots (heute) | ~8 s je Bildschirm, max. 3 | Maus + neuer Screenshot ~9 s | ja | Browser muss vorn sein |
| Bedienungshilfen des Browsers | 0,2 s für die ganze Seite (uv-Doku: 755 Elemente, ~1500 Tokens Text, 88 Links mit Adresse) | `AXPress` auf den Link: sofort, Maus bewegt sich nicht | ja (Sitzung des Nutzers) | Fenster muss nicht vorn sein (Hintergrund noch zu bestätigen) |
| Direkt per HTTP holen | 0,1–0,25 s (uv, Python-Doku, heise, react.dev) | Link-Adresse direkt holen | nein (reine JS-Seiten leer, keine Logins) | gar nicht sichtbar |

- Text statt Bild: Befehle und Versionen exakt statt abgelesen; ~1500–1800 Tokens für eine
  ganze Doku-Seite statt ~3200 Bild-Tokens + Bild-Encoder für 3 Bildschirme. Geschätzt
  ~5–6 s statt ~23 s pro Seite (nicht gemessen); mit nur dem Hauptinhalt (ohne Navigation)
  weniger. Große Seiten (heise: ~8000 Tokens) müssen gekürzt werden
- Firefox und Chrome bauen den Baum der Webseite nur auf Anfrage (`AXEnhancedUserInterface`
  bzw. `AXManualAccessibility` setzen); Links liefern `AXURL`, Aktionen `AXPress`,
  `AXScrollToVisible`
- Direktes Holen ist eine neue Netzverbindung von Flay selbst (bisher öffnet nur der Browser
  Seiten) – bewusste Ausnahme nötig, sonst nur als Rückfall
- [x] Reihenfolge in `read_webpage`: direkt per HTTP (`pages.fetch_page`) → Browser im
  Hintergrund über die Bedienungshilfen (`browser_text.read_url`) → Screenshots, wenn die
  Seite kaum Text hat (< 300 Zeichen oder „JavaScript aktivieren“). „Diese Seite“: erst der
  Browser (Login), dann direkt
- [x] Direkt: Hauptinhalt (`main`/`article`) ohne Navigation, Skripte und versteckte Teile;
  Links absolut, ohne Sprünge auf derselben Seite; echte Seiten 0,15–0,5 s (uv, Python-Doku,
  heise, react.dev, Google). Fehler gefunden: `content.read(n)` liefert nur das bisher
  Angekommene – jetzt bis zum Ende (höchstens 5 MB)
- [x] Hintergrund geprüft: PhpStorm blieb vorn, Firefox las die Python-Doku (1,7 s) und
  drückte einen Link per `AXPress` (1,1 s), ohne Maus und ohne Fensterwechsel
- [x] `browse_text`: Text + nummerierte Links/Buttons (passend zur Anfrage zuerst, nur
  Texte, keine Adressen), `open_link(number)` folgt direkt bzw. drückt im Browser,
  `read_more` für lange Seiten (`tools.web.max_chars`, 12 000 Zeichen)
- [x] Tempo: gleiche Werkzeugliste in allen Zügen (sonst liest das Modell alles neu: 15–25 s
  je Zug), Text im Entscheidungszug bricht ab und gilt als fertig. Echt gemessen, Modell
  warm: Google → Claude 13,7 s; uv-Installation über einen Link → Claude 15–26 s (vorher
  mit Screenshots/ohne Cache bis 64 s); Python-Tutorial → Kapitel Module 31 s
- [x] Zusammenfassungen („Inhalt“, „Links von …“) öffnen keine weiteren Links – das 8B
  öffnete „Gmail“, um Google zusammenzufassen; Echo der Anfrage am Anfang der Nachricht an
  Claude wird entfernt; „slash“ in gesprochenen Adressen
- [x] Datenschutz: direktes Laden ist bewusste Ausnahme (README, `.ai-instruct`),
  abschaltbar mit `tools.web.direct: false`
- [x] `flay eval`: Simulation liest Text mit Links, neuer Fall mit Linkfolgen; Webseiten
  16/16, voller Satz 96,6 % (wie vorher nur die genuschelten Befehle)
- [x] Windows: `browser_uia.py` mit denselben Funktionen über UI Automation (comtypes, nur
  unter Windows installiert): Standardbrowser aus der Registry, Dokument-Element lesen (ein
  Aufruf mit Cache für die ganze Seite), Links/Buttons per `InvokePattern`, Suchfeld per
  `ValuePattern`, Tab über seinen Schließen-Knopf, Öffnen ohne Aktivieren
  (`SW_SHOWNOACTIVATE`). **Auf echtem Windows ungetestet** – jeder Fehler fällt auf direkt
  bzw. Screenshots zurück
- [x] Tabs: im Hintergrund geöffnete Tabs schließt Flay nach der Anfrage über ihren
  Schließen-Knopf (`tools.web.close_tabs`), nur wenn der ausgewählte Tab noch Flays Seite
  zeigt; bleibt offen, wenn er für Screenshots nach vorn kam. Geprüft: Tab zu, PhpStorm
  blieb vorn
- [x] Lange Seiten in Abschnitten (Überschriften), passende zur Anfrage zuerst, Rest als
  Überschriftenliste; `read_more` mit Überschrift liest gezielt nach (deutsche Frage,
  englische Seite: „Vererbung“ → „Inheritance“). `tools.web.max_chars` 12 000 → 8000.
  Python-Klassen-Kapitel, Frage nach Vererbung: 51 s → 23 s
- [x] Suchen: `search_page` nutzt ein GET-Suchformular direkt (nur Formulare mit genau einem
  Textfeld, nie Login/Kontakt), im Browser das Suchfeld + Suchknopf im Hintergrund. Leere
  Ergebnisse (per JavaScript gefüllt, z. B. Python-Doku) → Browser. Echt: TaskGroup in der
  Python-Doku gesucht und erklärt, 16 s
- [x] Vergleichen: mehrere Links in einer Anfrage (`find_urls`) werden zusammen gelesen,
  ohne Links zu folgen; uv vs. pip echt 11 s
- [x] `flay eval`: 2 neue Fälle (Vergleich, Suche), Webseiten 20/20; voller Satz 95,9 % – neu
  daneben nur einmal „Mach mal Moewe auf“, das schon früher ohne Beispiele schwankte
- [ ] Windows-Test auf dem RTX-Rechner: UI Automation mit Chrome/Edge/Firefox, Öffnen im
  Hintergrund, Tab schließen
- [ ] Zweimal kam Firefox in Tests doch nach vorn (sonst blieb PhpStorm vorn, 9 Läufe);
  Ursache nicht gefunden – beobachten
- [ ] Erstes Lesen einer neuen Seite kostet weiter ~9–11 s (bis 8000 Zeichen Prefill)

## Phase 13 – Formulare, Dateien, Abläufe, Aktivierungswort

### Eingaben auf Webseiten

- [x] `fill_web_form`: Felder einer Seite im Browser (Bedienungshilfen bzw. UI Automation,
  `Page.inputs`), Zuordnung über die Beschriftung wie beim Klicken; erst Rückfrage mit allen
  Werten („Abschicken musst du selbst“), nach „ja“ ausfüllen und Browser nach vorn; nie
  Passwortfelder, nie abschicken
- [x] Echt geprüft (httpbin.org/forms/post): Name und E-Mail eingetragen, nicht gefundene Felder
  und Passwort genannt; Formularseiten gelten mit Feldern als geladen (vorher 15,9 s Warten)
- [x] „Trag in deinem Gedächtnis ein …“ ist `remember`, kein Formular; Fragen („Wie viel
  Speicher hat mein Gedächtnis?“) lösen nichts aus
- [x] Fix: die Windows-Funktionen von `browser_text` wurden von `ruff --fix` als „unbenutzt“
  entfernt (schon gepusht) – jetzt feste Namensliste `PLATFORM_FUNCTIONS` mit Test

### Lokale Dateien

- [x] `read_file`: Projekte aus PhpStorm, Schreibtisch, Downloads, Dokumente; Name nach
  Wortanfängen (nicht „ai“ in „training“), „neueste“ nach Datum, Projektname in den Wörtern
  als Ort; Text/Code/Markdown, PDF (pypdf, BSD), Word (docx ohne Zusatzpaket); dieselbe
  Text-Schleife wie Webseiten (Abschnitte, an Claude mit Quelle)
- [x] Finden in ≤ 0,01 s (flay-README, Roadmap, `.ai-instruct`, `flay.example.yaml`); echt:
  README-Frage 52 s (las drei Abschnitte nach), Roadmap → Claude 35 s
- [x] Fix: Echo der Anfrage am Anfang der Nachricht an Claude auch nach zweitem Schreibversuch
  entfernt

### Abläufe zeigen und wiederholen

- [x] `workflow` (start/stop/run/list/delete): Klicks mit Beschriftung und App (über die
  Bedienungshilfen, auch Menüleiste), Stelle im Fenster als Rückfall, Doppelklick, Tastenkürzel,
  Sondertasten, Scrollen; kein getippter Text, nie Flays Fenster, nicht die Sprechtaste
- [x] Abspielen sucht jedes Element neu (bis 5 s), aktiviert die App; Enter oder „Löschen“ im
  Ablauf → Rückfrage; „Ablauf fertig“ benennt nicht um
- [x] Echt: Klick aufs Git-Menü + Esc aufgenommen und in 2 s wieder abgespielt
- [ ] Windows: Aufnahme klappt (pynput), Elemente nur über die Stelle im Fenster

### Kleineres Ortungsmodell

- [x] Ollama-Varianten mit kleinem Kontext (`llm/ollama.py`, `flay setup`, „auto“ in der Config):
  `qwen3-vl:4b-instruct-ctx4k` (Ortung) und `qwen3-embedding:0.6b-ctx2k` (Gedächtnis; Name
  behält „qwen3-embedding“ für die Suchanweisung). Nur bei ≥ 20 GB nutzbarem Speicher
  (Mac 32 GB: 22,9) – nicht auf der RTX 4070 Ti (12 GB)
- [x] Gemessen (M2 Pro): 8B 10,1 GB + 4B 3,7 GB + Embedding 1,0 GB bleiben gleichzeitig geladen
  (vorher 4,8 GB fürs Embedding allein); Menüleiste mit frischem Bild: 4B 3/4 Treffer in
  4,5 s, 8B 2/4 in 8,5 s

### Aktivierungswort „Hey Flay“

- [x] Eigener Erkenner (`wake.py`): openWakeWord-Merkmalsmodelle (Apache 2.0, `flay setup`,
  SHA-256 geprüft) → 16 Embeddings (1,28 s) → numpy-Netz 1536 → 128 → 1; fertiges Modell im
  Paket (`data/hey_flay.npz`, 1,6 MB), `flay train-wake` überschreibt es im Datenordner
- [x] Training (`wake_training.py`, ~1 h CPU): Piper-Stimmen mit freien Lizenzen – Kerstin,
  Kristin, Thorsten (3), Ramona und die Mehrsprecher-Modelle LibriTTS-R (904 Sprecher), MLS
  (236), VCTK (109), CC BY 4.0; Tempo, Tonhöhe, Hall, Rauschen; Gegenbeispiele: ähnliche
  Laute, Alltagssätze, „Hey Leute, …“/„Okay, …“-Anfänge, Rauschen, „Hey“ allein; eigene
  Aufnahmen (`flay voice wake`) zählen mit
- [x] Verlauf, geprüft mit 12 fremden macOS-Stimmen (36× „Hey Flay“, 96 andere Sätze):
  nur fünf Einzelstimmen 1/5 erkannt → mit Mehrsprecher-Stimmen gleich → mit 1 s Stille
  davor beim Erkennen (Aufnahmen beginnen direkt mit dem Sprechen, die Merkmale brauchen
  0,8 s Vorlauf) und Befehl nach „Hey Flay“ im Training 36/36, aber 26/96 Fehlauslösungen
  → mit „Hey …“-Gegenbeispielen **35/36 erkannt, 18/96 Fehlauslösungen** (fast nur Sätze,
  die mit „Hey“ oder „Okay“ beginnen)
- [x] Darum entscheidet der Text: der Erkenner filtert nur (Schwelle 0,7), dann erkennt die
  Spracherkennung – nebenher, die laufende Antwort geht weiter – und nur mit „Hey Flay“ am
  Anfang (hey/hei/ey + flay/fley/flei/flai, nicht „play“/„fly“) startet ein Zug; „Hey Flay“
  allein → „Ja?“, der nächste Satz braucht es nicht (8 s)
- [x] Fix beim Bauen: eine Äußerung ohne Aktivierungswort brach die laufende Antwort ab
  (jede Äußerung startete einen neuen Zug); die Vorab-Erkennung konnte überschrieben werden
- [x] Schalter in Web-UI und Menüleiste, jederzeit (das dauerhafte Zuhören startet dafür neu,
  erst nachdem der alte Mikrofon-Strom zu ist); das Modell wird immer geladen, wenn vorhanden
- [x] Laufende Ablauf-Aufnahme in Web-UI (Balken mit „Fertig“) und Menüleiste sichtbar
- [ ] Echter Test mit Mikrofon und eigener Stimme; danach `flay voice wake` + `flay train-wake`
- [ ] Mehr echte Gegenbeispiele (freie Sprachdaten, z. B. Common Voice CC0) gegen die
  „Hey …“-Fehlauslösungen

### Eigene Stimme

- [x] `flay voice dataset`: die 965 Sätze des Stimmen-Trainings (jetzt `data/sentences/`) zum
  Vorlesen, jede Aufnahme sofort geprüft (Stille, Übersteuerung, Tempo 7–22 Zeichen/s),
  wiederholen/überspringen, fortsetzbar (`recorded.jsonl`), 22,05 kHz
- [x] `make_dataset.py --recorded` / `dataset.ps1 -Recorded`: Datensatz aus den Aufnahmen statt
  Qwen3-TTS (Whisper-Prüfung, Trimmen wie bisher, eigener Ordner `dataset-own`);
  `train.ps1`/`export.ps1 -Name` mit eigenen Ordnern für Cache und Läufe
- [ ] Aufnehmen (30–60 min) und auf dem Windows-Rechner trainieren

### LoRA mit erzeugten Beispielen (nicht übernommen)

- [x] `flay train-export --synthetic N`: zusätzliche Zeilen mit verstümmelten Befehlen
  (buchstabierte/verhörte Projektnamen, „Slie“, „She“, „Cloud“) und Gesprächen ohne Auftrag,
  mit erfundenen Projekten – keines aus `eval/cases.yaml`, damit `flay eval` Gelerntes misst
- [x] Getestet: Qwen3-VL-8B 4-bit (mlx-vlm), Rang 8, 260 Zeilen, 1 Epoche (~3,5 h auf dem M2 Pro,
  Prüf-Loss 0,0013 – das Muster ist auswendig gelernt)
- [x] Ergebnis `flay eval` (je 2 Durchläufe, 140 Läufe): ohne Adapter 95,7 % (108/114 Aktionen,
  Fehler nur #10, #11, #30), mit Adapter 88,6 % (100/114, 8 falsche Aktionen). Behoben nur #10;
  neu kaputt u. a. #05/#06 (Projekt öffnen), #42 (Links an Claude), #51 (PDF), #53 (Ablauf),
  #59. Der Adapter verengt das Modell auf die drei geübten Werkzeuge → nicht übernommen,
  Modell und Adapter wieder gelöscht
- [ ] Nur mit echten Beispielen aus dem Verlauf über alle Werkzeuge erneut versuchen

## Verworfen (bei Bedarf wieder aufnehmen)

Kamera, Audio direkt ans Modell, Benchmark weiterer
LLM-Server, semantische Suche im Gedächtnis, Windows-Tests.
