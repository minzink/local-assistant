# Flay

Lokaler Sprachassistent für macOS (Apple Silicon) und Windows: Push-to-Talk → Speech-to-Text →
lokales LLM → Text-to-Speech. Deutsch und Englisch. Alles läuft offline auf dem eigenen Rechner.

> Status: **Phase 4** – Gespräch per Push-to-Talk oder Chat, Gedächtnis, Werkzeuge für
> PhpStorm, Claude Code und den Bildschirm. Nächste Schritte siehe
> [docs/roadmap.md](docs/roadmap.md).

## Setup

Voraussetzung: [uv](https://docs.astral.sh/uv/) (`brew install uv` bzw.
`winget install astral-sh.uv`). uv installiert auch die passende Python-Version (3.12).

```sh
uv sync
uv run flay setup       # lädt einmalig alle Modelle – der einzige Befehl, der etwas herunterlädt
uv run flay info        # erkannte Hardware + aufgelöste Konfiguration
```

Außerdem wird ein lokaler LLM-Server mit OpenAI-kompatibler API benötigt, z.B.
[Ollama](https://ollama.com) ≥ 0.30 mit `ollama pull qwen3-vl:8b-instruct` (Apache-2.0; versteht
Text und Bilder und kann Werkzeuge aufrufen; die Variante ohne `-instruct` denkt vor jeder
Antwort 6–10 s nach), alternativ LM Studio oder llama.cpp. Adresse und
Modell stehen in der Config (`llm.base_url`, `llm.model`).

## Starten

```sh
uv run flay run              # Push-to-Talk-Taste halten, sprechen, loslassen
uv run flay run --continuous # dauerhaft zuhören, einfach sprechen
uv run flay run --terminal   # Enter startet, Enter stoppt (ohne Hotkey-Berechtigung)
```

**Dauerhaftes Zuhören** (`--continuous` bzw. `input.mode: continuous`, oder jederzeit über den
Schalter „Zuhören“ in der Web-UI bzw. „Dauerhaft zuhören“ im Menüleisten-Symbol; Push-to-Talk
funktioniert immer zusätzlich): Silero VAD erkennt,
wann du sprichst; eine Pause von 600 ms beendet den Satz. Schon bei der ersten kurzen Pause
beginnt die Spracherkennung, damit die Antwort früher kommt. Sprichst du, während Flay redet,
verstummt es (Barge-in) – das braucht ein Headset; über Lautsprecher würde Flay sich selbst
hören, dann `input.continuous.barge_in: false` setzen.

**Aktivierungswort „Hey Flay“** (Schalter „Hey Flay“ in der Web-UI bzw. „Nur auf „Hey Flay“
reagieren“ im Menüleisten-Symbol, jederzeit; beim Start an mit
`input.continuous.wake_word: true`; braucht einmal `flay setup`): Beim dauerhaften Zuhören
zählt dann nur, was mit „Hey Flay“ beginnt („Hey Flay, öffne flay“); „Hey Flay“ allein
beantwortet Flay mit „Ja?“ und nimmt den nächsten Satz ohne. Ein kleiner, selbst trainierter
Erkenner filtert (auf den Merkmalsmodellen von openWakeWord, Apache 2.0), der erkannte Text
entscheidet: ohne „Hey Flay“ am Anfang wird nichts ausgeführt. Gespräche im Raum unterbrechen
Flay dann nicht mehr; „Hey Flay …“ schon. Mit eigenen Aufnahmen wird der Erkenner besser:
`flay voice wake` (20× „Hey Flay“), dann `flay train-wake`.

Unterbrichst du Flay (per Taste oder Sprache), merkt es sich nur, was es bis dahin gesagt hat,
und weiß in der nächsten Antwort, dass du es unterbrochen hast – auch wenn es noch gar nichts
gesagt hatte. So bleibt eine abgebrochene Frage im Zusammenhang.

**Menüleiste/Tray:** Ein Punkt zeigt den Zustand (grau bereit, orange hört zu, blau denkt
nach, grün spricht, roter Strich = Mikrofon stumm). Das Menü schaltet Mikrofon und Sprache
stumm, schaltet dauerhaftes Zuhören und „Hey Flay“, beendet eine laufende Ablauf-Aufnahme,
öffnet die Web-UI und beendet Flay. Abschalten: `--no-tray` bzw. `ui.tray: false`.

**Autostart:** `uv run flay autostart on` startet Flay bei jeder Anmeldung im Hintergrund
(`off` entfernt es, ohne Argument zeigt es den Status). macOS fragt danach für den
Hintergrundprozess erneut nach Mikrofon, Bedienungshilfen und Bildschirmaufnahme.

Mit `flay run` öffnet sich außerdem die **Web-UI** unter <http://127.0.0.1:8765>:
Chatverlauf (gesprochene und getippte Nachrichten, Antworten live mitgeschrieben), ein
Eingabefeld für Text sowie Schalter für **Zuhören** (dauerhaft), **Hey Flay**
(Aktivierungswort), **Mikrofon** (Push-to-Talk wird ignoriert) und **Sprachausgabe** (Flay
antwortet nur als Text). Während ein Ablauf aufgenommen wird, zeigt ein Balken das an; „Fertig“
speichert ihn wie „Ablauf fertig“. Die UI ist nur auf diesem Rechner
erreichbar und nimmt WebSocket-Verbindungen nur von der eigenen Seite an. Abschalten mit
`flay run --no-ui` oder `ui.enabled: false`.

Beim ersten Start werden das Whisper-Modell (~1,6 GB), die Stimme (Qwen3-TTS, ~2 GB; dazu
die Piper-Stimmen) und das Silero-VAD-Modell (2 MB) geladen.
Standardtaste: rechte Option-Taste (macOS) bzw. rechte Strg-Taste (Windows),
änderbar über `input.push_to_talk_key`. Wer spricht, während Flay antwortet, unterbricht die
Antwort.

**macOS:** Die Terminal-App (bzw. PhpStorm, wenn dort gestartet) braucht in
*Systemeinstellungen → Datenschutz & Sicherheit* die Berechtigung *Mikrofon* und für den Hotkey
*Bedienungshilfen* sowie *Eingabeüberwachung*. Ohne Mikrofon-Berechtigung liefert macOS still
Stille – Whisper erfindet daraus Wörter wie „you“. Flay verwirft deshalb stumme Aufnahmen
mit einer Warnung. Nach dem Erteilen die App neu starten.

Bei Erkennungsproblemen zuerst das Mikrofon prüfen:

```sh
uv run flay mic-test         # 4 s aufnehmen, Gerät, Pegel (dBFS) und erkannten Text anzeigen
uv run flay -v run           # zeigt zusätzlich Länge und Pegel jeder Aufnahme
```

Sprache liegt typischerweise bei −40 bis −15 dBFS; unter −60 dBFS gilt die Aufnahme als stumm.

Weitere Befehle:

```sh
uv run flay devices                          # Audio-Geräte (für audio.input_device/output_device)
uv run flay mic-test                         # Mikrofon testen
uv run flay bench                            # LLM-Latenz: erstes Token, Zeichen/s
uv run flay bench --model phi4               # anderes Modell vergleichen
uv run flay bench --base-url http://localhost:1234/v1 --model <name>   # anderer Server
```

Jede Runde wird mit Zeiten pro Stufe geloggt (`vad`, `stt`, `llm_first_token`,
`first_sentence`, `first_audio`, `llm_done`, `done`), gemessen ab dem Loslassen der Taste.
Die Antwort wird satzweise gesprochen, während das LLM noch schreibt.

## Werkzeuge

Flay kann auf deinem Rechner handeln – nur über die Werkzeuge in `tools.enabled`:

| Werkzeug | Beispiel | Was passiert |
|---|---|---|
| `open_project` | „Öffne das Projekt Stadt Wolkenthal“ | Sucht in PhpStorms zuletzt geöffneten Projekten und öffnet das passende per Toolbox-Starter |
| `close_project` | „Schließe das Projekt Moewe“ | Schließt das passende, gerade geöffnete Projekt (über PhpStorms MCP-Server) |
| `send_to_claude` | „Sag Claude, er soll Tests für den Login schreiben“ | Schreibt den Auftrag in Claudes Eingabezeile – **abschicken (Enter) tust du selbst** |
| `claude_status` | „Ist Claude fertig?“ | Liest Claudes Terminal und fasst zusammen: arbeitet noch, fertig, wartet auf Erlaubnis |
| `look_at_screen` | „Was siehst du auf meinem Bildschirm?“, „Schau dir das PhpStorm-Fenster an“, „Was ist auf dem zweiten Bildschirm?“ | Screenshot des Hauptbildschirms, des aktiven Fensters, einer App oder eines Monitors – verkleinert, an das lokale Modell |
| `click_on_screen` | „Klick auf das Git-Menü“, „Rechtsklick auf UTF-8“, „Drück auf Speichern“ | Sucht das Element und klickt (links, rechts, doppelt) – siehe unten |
| `scroll_screen` | „Scroll eine Seite runter“, „Scroll in Firefox nach oben“ | Scrollt das aktive oder genannte Fenster seitenweise |
| `open_url` | „Öffne python punkt org“, „Öffne den Link aus der Zwischenablage“ | Öffnet den Link im Standardbrowser |
| `read_webpage` | „Lies heise punkt de und sag mir, was es Neues gibt“ | Liest die Seite als Text (direkt oder im Browser im Hintergrund, sonst per Screenshot), folgt bei Bedarf Links und antwortet daraus |
| `fill_web_form` | „Trag auf dieser Seite bei Name Max Mustermann und bei E-Mail … ein“ | Füllt Felder eines Formulars im Browser aus – erst nach deinem „ja“, **abschicken tust du selbst** |
| `read_file` | „Lies die README von flay und sag mir, wie man es installiert“, „Was steht im PDF auf dem Schreibtisch?“ | Findet eine lokale Datei (Text, Code, Markdown, PDF, Word) und antwortet daraus oder gibt sie Claude |
| `workflow` | „Merk dir einen Ablauf: Release bauen“ … „Ablauf fertig“, später „Mach den Ablauf Release bauen“ | Schaut dir einmal zu und wiederholt die Klicks und Tastenkürzel auf Zuruf |
| `remember` | „Merk dir, dass Lea am 3. März Geburtstag hat“ | Speichert den Fakt im Gedächtnis |
| `forget` | „Vergiss, dass ich Tee mag“ | Löscht den passenden Fakt; passen mehrere, fragt Flay nach |
| `ssh_connect` | „SSH auf prod web“, „Verbinde dich mit dem Server staging“ | Öffnet ein neues Terminalfenster mit `ssh <host>` – nur für Server aus deiner ssh-Config |

**Damit Aktionen stimmen:** Nach einer Aktion spricht Flay den festen Satz des Werkzeugs („flay
ist geschlossen.“) statt einer Umformulierung durch das Modell – das ist schneller und kann
nichts Falsches behaupten. Was ausgeführt wurde, bleibt im Verlauf, sodass Flay bei
Nachfragen weiß, was wirklich passiert ist. Zu jeder Anfrage liest Flay, welche Projekte in
PhpStorm offen sind und für welche Claude läuft, und korrigiert Projektnamen und
Fachbegriffe in der Spracherkennung („Symphony“ → „Symfony“, „Stadt Wolkental“ → „Stadt Wolkenthal“).

Flay handelt nur mit einem Projekt, das du (auch verstümmelt) genannt hast – sonst fragt es
„Welches Projekt meinst du?“. Kommt der Name beim Schließen nur ungefähr vor („KEWO“), fragt es
„Meinst du project symfony kx ewo?“ und schließt erst nach „ja“. „…, never mind“ oder „vergiss
es“ am Ende nimmt eine Anfrage zurück; „Wie schließe ich …?“ ist eine Frage, keine Aktion.

**Server per SSH:** Welche Server es gibt, liest Flay bei jeder Anfrage aus der ssh-Config
(`~/.ssh/config` samt `Include`, unter Windows `%USERPROFILE%\.ssh\config`); Platzhalter wie
`Host *` zählen nicht. Der gesprochene Name wird wie ein Projektname abgeglichen, auch über den
`HostName` dahinter. An ssh geht nur der Host-Name aus der Config – Benutzer, Port, Schlüssel
und Jump-Hosts kommen von dort. Flay führt auf dem Server nichts aus, es öffnet nur die
Sitzung; einen Server, den du nicht genannt hast, wählt es nie selbst. Das Terminal stellt
`tools.ssh.terminal` ein (macOS: Terminal.app oder iTerm, Windows: Windows Terminal oder
Konsole). Auf dem Mac fragt macOS beim ersten Mal, ob Flay Terminal steuern darf.

**Klicken:** Flay sucht das Element zuerst über die Bedienungshilfen von macOS – Beschriftung
und exakter Rahmen jedes Buttons, Menüs oder Links, vorderste App zuerst (~0,3 s). Hat es keine
Beschriftung („das Glocken-Symbol rechts oben“) oder passt sie mehrfach, schaut das Bildmodell
in einen frischen Screenshot und nennt einen Punkt; kann es ihn nicht über die Bedienungshilfen
bestätigen, schaut es ein zweites Mal genauer auf einen Ausschnitt (`tools.pointer.refine`).
Zum Orten nimmt Flay ein kleineres Modell (Qwen3-VL 4B mit kurzem Kontext, `flay setup` legt
es an, wenn der Speicher reicht): ~4,5 s je neuem Bild statt ~8,5 s mit dem 8B, und es bleibt
neben dem Hauptmodell geladen. Beschriftungen sind also schnell und sicher, Symbole nach
Beschreibung langsamer und weniger zuverlässig. Flay klickt nie in sein
eigenes Fenster und fragt vor einem Linksklick auf etwas wie „Löschen“, „Senden“ oder „Push“
nach („Soll ich wirklich auf Löschen klicken?“); ein Rechtsklick öffnet nur ein Menü. Unter
Windows gibt es (noch) keine Bedienungshilfen-Suche, dort ortet immer das Bildmodell.

**Webseiten:** „Öffne diesen Link https://www.google.com/, fasse den Inhalt zusammen und gib es
an Claude“ – Flay sagt „Ich öffne die Seite und lese sie.“ und liest die Seite als Text, ohne
dass sich auf dem Bildschirm etwas ändert:

1. **Direkt** per HTTP (`tools.web.direct`, ~0,2 s, unsichtbar) – reicht für die meisten
   Seiten, aber ohne deine Logins und ohne Inhalte, die erst JavaScript aufbaut.
2. **Aus deinem Browser** (`tools.web.browser_text`): Die Seite öffnet sich im Hintergrund
   (der Browser kommt nicht nach vorn), Flay liest Text und Links über die Bedienungshilfen
   (macOS, ~1,5 s) bzw. UI Automation (Windows: Chrome, Edge, Firefox; noch ungetestet) – mit
   deinen Logins. Danach schließt Flay den Tab wieder (`tools.web.close_tabs`), sofern er
   noch die Seite zeigt, die Flay geöffnet hat.
3. **Per Screenshot**, wenn die Seite kaum Text hat (Bilder, Canvas): Dann kommt der Browser
   nach vorn, Flay nimmt bis zu `tools.web.max_pages` Bildschirme auf (je ~8 s).

Text ist schneller und genauer als Bilder: Befehle und Versionen kommen buchstabengenau an.
Von langen Seiten liest das Modell zuerst die Abschnitte, die zur Anfrage passen
(`tools.web.max_chars`, 8000 Zeichen), und sieht die Überschriften der übrigen; einen davon
kann es gezielt nachlesen („Inheritance“ für eine Frage nach Vererbung).
Daraus schreibt das Modell die Zusammenfassung, die Flay in Claudes Eingabezeile tippt (ohne
Enter). Ohne „an Claude“ beantwortet Flay die Frage zur Seite direkt. Den Link kannst du
tippen (Web-UI), sprechen („heise punkt de“, „docs punkt python punkt org slash 3“), kopieren
(„der Link aus der Zwischenablage“) oder aus Claudes letzter Antwort nehmen lassen („öffne den
Link von Claude“). Mit „diese Seite“ liest Flay die Seite, die gerade im Browser offen ist.

**Aufgaben auf Webseiten:** „Schick Claude die Installationsschritte von docs punkt astral punkt
sh slash uv als Aufgabe“ – steht das Gesuchte nicht auf der Seite, öffnet Flay den passenden
Link („Ich öffne Installation.“): direkt gelesene Seiten folgt es direkt, im Browser gelesene
drückt es dort im Hintergrund, ohne Maus. Ist der Text lang, liest es weiter. Höchstens
`tools.web.max_steps`-mal (Standard 4). Dann schreibt es Claude die Aufgabe mit den Details von
der Seite und der Quelle in die Eingabezeile (ohne Enter). Gemessen (M2 Pro, Modell warm):
Google zusammenfassen und an Claude ~14 s, uv-Installationsschritte über einen Link ~15–26 s.
Bei Zusammenfassungen öffnet Flay keine weiteren Links.

**Suchen und Vergleichen:** „Such auf docs punkt python punkt org nach TaskGroup und sag mir,
wofür es gut ist“ – Flay nutzt die Suche der Seite: ein Suchformular direkt per HTTP, sonst das
Suchfeld im Browser (im Hintergrund ausgefüllt und abgeschickt). Andere Formulare füllt es nie
aus. „Vergleiche docs punkt astral punkt sh slash uv und pip punkt pypa punkt io“ liest beide
Seiten und vergleicht sie (gemessen ~11 s). Nie geöffnet oder gedrückt werden
Suche, Anmelden/Abmelden, Kaufen, Senden u. ä.; bei Seiten aus Screenshots lehnt Flay
Cookie-Banner ab.

**Formulare:** „Trag auf dieser Seite bei Name Max Mustermann und bei E-Mail max at example
punkt org ein“ – Flay liest die Felder der Seite im Browser über die Bedienungshilfen (bzw. UI
Automation), ordnet sie über ihre Beschriftung zu und fragt erst: „Soll ich auf … bei Name
„Max Mustermann“ … eintragen? Abschicken musst du selbst.“ Nach „ja“ füllt es aus und holt den
Browser nach vorn. Passwortfelder füllt es nie, Formulare schickt es nie ab.

**Lokale Dateien:** „Lies die README von flay und sag mir, wie man es installiert“, „Fasse die
Roadmap im Projekt flay zusammen und gib es Claude“, „Was steht in der neuesten PDF in meinen
Downloads?“ – Flay sucht in PhpStorms Projekten, auf dem Schreibtisch, in Downloads und
Dokumenten (Name nach Wortanfängen, „neueste“ nach Datum) und liest Text, Code, Markdown, PDF
und Word-Dateien wie eine Webseite: lange Dateien in Abschnitten, passende zuerst.

**Abläufe:** „Merk dir einen Ablauf: Release bauen“ – Flay schaut zu, während du es einmal
machst: Klicks (mit der Beschriftung des Elements, sonst der Stelle im Fenster), Tastenkürzel,
Enter/Tab/Pfeile, Scrollen. „Ablauf fertig“ speichert ihn (`<data_dir>/workflows.json`),
„Mach den Ablauf Release bauen“ spielt ihn ab und sucht jedes Element neu – Fenster dürfen
sich verschieben. Getippter Text wird nie aufgezeichnet (könnte ein Passwort sein). Enthält
ein Ablauf Enter oder Beschriftungen wie „Löschen“, fragt Flay vor dem Abspielen. „Welche
Abläufe kennst du?“, „Lösch den Ablauf …“.

**Denken auf Wunsch:** Mit einem Modell mit Denk-Modus (`llm.thinking`, siehe
`flay.example.yaml`) überlegt Flay, wenn du es darum bittest („denk mal nach, wie …“,
„Schritt für Schritt“). Es sagt zuerst „Moment, ich überlege kurz.“; der Denktext selbst wird
nie gesprochen. Für normale Befehle bringt Denken nichts (gemessen), für Rechen-, Zeit- und
Logikfragen viel: mit `qwen3-vl:8b-thinking` 100 % statt 75 % richtig, dafür 7–30 s statt
1,5 s. Damit Ollama dafür nicht jedes Mal das Modell wechseln muss, in Ollama die
Kontextlänge auf 8k stellen – dann bleiben beide Modelle geladen (je 6,1 GB statt 10 GB).

### Claude Code anbinden

`flay` einmal global verfügbar machen (Änderungen am Code wirken sofort):

```sh
uv tool install --editable <Pfad zum Repo>
```

Dann im PhpStorm-Terminal des Projekts **`flay claude` statt `claude`** starten (weitere
Argumente werden an Claude durchgereicht, z.B. `flay claude --continue`). Claude läuft dort
wie gewohnt, aber in einer tmux-Sitzung pro Projekt – Flay kann so den Auftrag direkt in die
Eingabezeile schreiben, ohne Fensterwechsel, und Claudes Status lesen. Du siehst alles live
und drückst Enter.

**Welche Sitzung?** Nennst du ein Projekt („Sag Claude im Projekt Stadt Wolkenthal …“), die
dieses Projekts. Sonst die Sitzung, die du gerade in einem Terminal offen hast (zuletzt
benutzte zuerst). PhpStorms Projektliste wird nur selten gespeichert und taugt deshalb nicht,
um „das aktuelle Projekt“ zu erkennen.

Läuft keine sichtbare Sitzung, **öffnet Flay Claude selbst in PhpStorm**: Über PhpStorms
eingebauten MCP-Server (*Settings → Tools → MCP Server* aktivieren; Flay findet ihn
automatisch) legt es kurz `.flay-claude.sh` im Projekt an, lässt PhpStorm das Skript als
Run-Configuration in einem neuen Terminal-Tab ausführen (`flay claude`, hängt sich an eine
bestehende Sitzung an), löscht die Datei wieder, wartet, bis Claude bereit ist, und tippt den
Auftrag ein. Welches Projekt: das genannte, sonst das einzige offene; sind mehrere offen,
fragt Flay nach. Die offenen Projekte kommen live von PhpStorm.

Tipp: Trägst du in *Settings → Tools → Claude Code [Beta] → Claude command* `flay claude`
ein, startet auch der Claude-Button von PhpStorm Claude so, dass Flay hineinschreiben kann.

Ist ein genanntes Projekt in PhpStorm nicht geöffnet, startet Flay Claude im Hintergrund;
mit `flay claude` im Terminal dieses Projekts hängst du dich an. Flay sagt dir dann, für welches Projekt; mit `flay claude` im
PhpStorm-Terminal dieses Projekts hängst du dich an die Sitzung an und siehst den Auftrag.
Fragt Claude beim ersten Start in einem Ordner, ob du ihm vertraust, tippt Flay nichts (dort
ist „No, exit“ vorausgewählt) und bittet dich, die Frage selbst zu beantworten.
Abschalten mit `tools.claude.autostart: false`.

Ohne tmux (Windows) oder ohne passendes Projekt legt Flay den Auftrag in die Zwischenablage
und holt PhpStorm nach vorn; du fügst ihn mit `Cmd/Ctrl+V` in Claude ein. Erzwingen lässt sich
ein Weg mit `tools.claude.transport: tmux | clipboard`.

- PhpStorm fragt beim Öffnen eines Projekts standardmäßig, in welchem Fenster. Damit Projekte
  immer in einem neuen Fenster aufgehen: *Settings → Appearance & Behavior → System Settings →
  Project → Open project in: New window* (Flay weist beim Start darauf hin).
- Erwähnst du den Bildschirm („Bildschirm“, „screen“, „was siehst du“, „schau dir das an“),
  hängt Flay den Screenshot direkt an – unabhängig davon, ob das Modell das Werkzeug wählt.
- Screenshots bleiben lokal. Was du an Claude weitergibst, geht – wie jede Claude-Nutzung –
  an Anthropic.
- **macOS-Berechtigungen** für die App, aus der Flay läuft: *Bildschirmaufnahme* (ohne sie
  ist das Bild leer und Flay sagt das) und *Bedienungshilfen* (Hotkey, Klicken und Scrollen –
  ohne sie verwirft macOS Mausereignisse stillschweigend; Flay fragt beim ersten Klick danach).
- Werkzeug-Aufrufe erscheinen in der Web-UI im Chatverlauf. Der Button unten rechts öffnet
  das **Debug-Log**: alle Logs von Flay, Aktionen auf dem Rechner (Projekt geöffnet, an Claude
  geschrieben, Screenshot …) hervorgehoben und per „Nur Aktionen“ filterbar.
- Beim Start loggt Flay, welche `flay claude`-Sitzungen es findet.

## Persönlichkeit und Gedächtnis

- **Persona:** `persona.system_prompt` und `persona.mood` (frei benannte Stimmungen von 0–10,
  z.B. `cheerful: 6`) fließen in den System-Prompt ein, zusammen mit Datum und Uhrzeit.
- **Gesprächsverlauf:** Die letzten `memory.recent_messages` Nachrichten der Sitzung gehen
  wörtlich ans LLM, ältere werden zusammengefasst. Nach `memory.session_timeout_minutes`
  Pause beginnt eine neue Sitzung.
- **Langzeit-Gedächtnis (RAG):** Flay sammelt im Hintergrund Erinnerungen:
  - *Fakten* über dich, deine Projekte, Entscheidungen und Termine („Merk dir, dass ich am
    Freitag Urlaub habe“, „Im Projekt umx nutzen wir Redis“) – relative Daten werden zu
    absoluten, widerrufene Fakten gestrichen.
  - *Episoden:* Ist ein Gespräch vorbei, wird es zusammengefasst („Am 26.10. wollte der
    Nutzer …“), sodass Flay später darauf zurückgreifen kann („Was haben wir letztes Mal
    besprochen?“).
- **Datum und Wichtigkeit:** Jede Erinnerung trägt, wann sie geschrieben wurde, und eine
  Wichtigkeit: *wichtig* sind dauerhafte Regeln und Anweisungen („Gib Claude in Zukunft keinen
  Befehl mehr, etwas zu committen“) und was du als wichtig bezeichnest – sie gelten immer und
  stehen bei jeder Anfrage im Prompt (auch beim Formulieren von Aufträgen an Claude);
  *normal* ist dauerhaftes Wissen; *nebensächlich* sind einmalige Ergebnisse („die README
  wurde heute nicht angepasst“) und Gesprächszusammenfassungen – sie zählen beim Suchen weniger.
- **Beispiele aus dem Verlauf:** Führt Flay eine Aktion aus und widersprichst du nicht
  („nein“, „falsch“), merkt es sich den Satz, wie es ihn gehört hat, mit der Aktion
  („She K Command.“ → Projekt component-symfony-kx-command schließen). Ähnliche Anfragen
  bekommen diese Beispiele mit – so lernt Flay deine Sprechweise und die typischen Hörfehler.
  Hast du eine Aktion mit „ja“ bestätigt oder meinte derselbe Satz zweimal dasselbe, führt
  Flay ihn beim nächsten Mal direkt aus. Gemessen: 98,7 % statt 92,1 % Treffer, keine
  Rückfrage mehr statt der Aktion (`flay eval --examples eval/examples.yaml`).
- **Abruf:** Zu jeder Anfrage sucht Flay die passenden Erinnerungen – nach Bedeutung
  (Embedding-Modell `qwen3-embedding:0.6b`, Apache 2.0, ~640 MB, über Ollama) und nach
  Stichwörtern (SQLite FTS5) – und gibt die besten (höchstens `memory.recall.limit`) mit
  Datum an das Modell. Die Suche dauert ~20–30 ms und läuft parallel zur Weiterleitung;
  ohne Embedding-Modell sucht Flay nur nach Stichwörtern. Einmalig:
  `ollama pull qwen3-embedding:0.6b`.
- Die Hintergrundarbeit startet erst nach `memory.idle_seconds` Ruhe und wird abgebrochen,
  sobald du wieder sprichst – sie verlängert also keine Antwortzeit.

Alles liegt lokal in `<data_dir>/memory.sqlite3` (Pfad: `flay info`). Verwalten:

```sh
uv run flay memory               # Erinnerungen (Fakten und Episoden) anzeigen
uv run flay memory search Urlaub # suchen wie Flay es tut
uv run flay memory forget 3 7    # einzelne Erinnerungen löschen
uv run flay memory clear-chat    # nur Gesprächsverlauf + Zusammenfassungen löschen
uv run flay memory clear         # Gespräche, Zusammenfassungen und Fakten löschen
```

Mit `memory.enabled: false` merkt sich Flay nur die laufende Unterhaltung im Arbeitsspeicher.

**Antwortzeit:** Flay hält das Modell warm (`llm.keep_warm_minutes`), beginnt mit dem
Sprechen nach dem ersten Satzstück und erkennt beim dauerhaften Zuhören schon während der
Pause. Gemessen auf einem M2 Pro mit Parakeet: ~0,85 s vom Loslassen der Taste bzw. ~1,2 s vom
Sprechende (davon 0,6 s Warten auf das Satzende) bis zum ersten Ton. Mit Whisper
(`stt.backend: auto`, etwas genauer) sind es ~1,4 bzw. ~1,55 s.
Dauert die Antwort länger als 0,4 s nach dem Sprechende, sagt Flay vorab kurz „Mhm.“ bzw.
bei Aktionen „Moment.“ (vorab erzeugt, kostet keine Zeit; `tts.filler`, abschaltbar).

**Stimme:** Flay spricht Deutsch und Englisch mit derselben Stimme. Qwen3-TTS (0.6B, Apache
2.0) ahmt dafür eine kurze Aufnahme nach (`tts.voice.name`):

| Stimme | |
|---|---|
| `auto` (Standard) | die eigene Aufnahme, falls vorhanden, sonst `ramona` |
| `own` | deine Stimme: `flay voice record` nimmt einen deutschen und einen englischen Text auf (je ~10 s, bleibt lokal) |
| `ramona`, `hokuspokus` | Sprecherinnen von LibriVox (gemeinfrei) |
| `thorsten` | Sprecher (CC0) |
| Pfad zu einer WAV | eigene Aufnahme mit dem gesprochenen Text in einer `.txt` daneben |

**Eigene Piper-Stimme:** `flay voice dataset` zeigt die 965 Sätze des Stimmen-Trainings
nacheinander zum Vorlesen (prüft jede Aufnahme auf Stille, Übersteuerung und Tempo; jederzeit
unterbrechen und weitermachen). 30–60 Minuten ergeben eine eigene, schnelle Piper-Stimme für
beide Sprachen; trainiert wird auf dem Windows-Rechner mit
`tools/voice-training/dataset.ps1 -Recorded <Ordner>` und `train.ps1 -Name flay-<name>`
(siehe `tools/voice-training/README.md`).

`tts.voice.pitch` hebt oder senkt die Stimme in Halbtönen, `tts.voice.speed` ändert das Tempo
(1.2 = 20 % schneller). Beides wird einmal beim Start auf die Aufnahme angewendet und kostet
beim Sprechen keine Zeit. Probehören: `flay voice test` (oder `--name ramona`).
Auf einem M2 Pro (MLX, 4 Bit) kommt der erste Ton nach ~0,2 s; ein Satz braucht ~0,4 s je
Sekunde Sprache. Unter Windows läuft Qwen3-TTS mit PyTorch auf der NVIDIA-GPU (noch nicht
gemessen); ohne GPU nimmt Flay Piper (`tts.backend`), mit einer Stimme je Sprache: Kerstin (de)
und Kristin (en), beide weiblich, damit der Sprachwechsel ähnlich klingt.

**Ollama:** Ollama entlädt Modelle nach 5 Minuten Leerlauf; die nächste Antwort dauert dann
2–3 s länger. Abhilfe: `OLLAMA_KEEP_ALIVE=1h` (oder `-1`) für den Ollama-Server setzen.

## Konfiguration

Standardwerte stehen im Code (`src/flay/config.py`). Eigene Werte in `flay.yaml` eintragen
(Vorlage: `flay.example.yaml`); alternativ Pfad per `--config` oder `$FLAY_CONFIG`.
Unbekannte Schlüssel und ungültige Werte werden beim Start abgelehnt.

## Architektur

```
src/flay/
├── core/       Sprache (de/en), Nachrichten, Ereignisse, Satz-Splitter, Hardware-Erkennung
├── input/      UtteranceSource – Push-to-Talk (Hotkey/Terminal), später dauerhaftes Zuhören
├── audio/      AudioBuffer (PCM16 mono), Mikrofon, Lautsprecher (sounddevice), Silero VAD
├── stt/        SpeechToText – Whisper (mlx-whisper auf Apple Silicon, faster-whisper sonst)
├── llm/        ChatModel – austauschbar; Standard: OpenAI-kompatible API (Ollama, LM Studio, …)
├── tts/        TextToSpeech – Qwen3-TTS (eine Stimme, gestreamt) oder Piper, satzweise
├── memory/     Memory – Verlauf, Zusammenfassungen, Fakten (SQLite), Hintergrundpflege per LLM
├── assistant.py  Orchestrator: Äußerung → STT → LLM → TTS → Lautsprecher, mit Zeitmessung
├── prompt.py   System-Prompt aus Persona, Stimmung, Datum, Fakten und Zusammenfassung
├── ui/         Lokale Web-UI (aiohttp + WebSocket, eine HTML-Seite ohne Build-Schritt)
├── bench.py    LLM-Latenz-Benchmark
├── config.py   Settings (pydantic) + YAML-Loader
└── cli.py      Einstiegspunkt `flay`
```

Jede Stufe ist ein `Protocol`; Backends werden in `*/factory.py` anhand der Config gewählt und
sind austauschbar, ohne die Pipeline anzufassen.

**Multimodal vorbereitet:** Nachrichten bestehen aus Content-Parts (`TextPart`, `ImagePart`,
`AudioPart`). Version 1 erzeugt nur Text. Welche Eingaben ein Modell versteht, steht in
`llm.input_modalities`; `ModelCapabilities.ensure_supported` lehnt alles andere ab. Ein
multimodales Modell wird später per Config aktiviert, z.B. `input_modalities: [text, image]`.

## Entwicklung

```sh
uv run pytest
uv run ruff check . && uv run ruff format .
uv run mypy
```

CI prüft dasselbe auf macOS und Windows.

### Sprachmodelle vergleichen (`flay eval`)

`flay eval` schickt den Testsatz aus `eval/cases.yaml` (echte, teils falsch erkannte Befehle
und normale Gesprächssätze) durch Flay – mit Weiterleitung, Prompt und Werkzeug-Schleife wie im
Betrieb, aber mit simulierten Werkzeugen: auf dem Rechner passiert nichts. Gemessen werden die
Trefferquote (richtiges Werkzeug, richtiges Projekt; bei Gesprächssätzen keine Aktion) und die
Zeit bis zum Werkzeug-Aufruf bzw. zur ersten Antwort. Ergebnisse landen als JSON im
Datenordner unter `eval/`.

```sh
uv run flay eval                                   # Modell aus flay.yaml, jeder Fall 2×
uv run flay eval --base-url http://127.0.0.1:8080/v1 --model qwen3.5-9b \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
uv run flay eval --only 6,10,11 --repeat 5         # einzelne Fälle genauer
```

### Spracherkenner vergleichen (`flay eval-stt`)

`flay record` zeigt die Sätze aus `eval/speech.yaml` nacheinander an; du sprichst sie ein (Enter
startet und stoppt). Die Aufnahmen bleiben im Datenordner unter `eval/speech/<name>/`.
`flay eval-stt` erkennt sie mit jedem Spracherkenner und misst Wortfehlerrate, ob Verben,
Projektnamen und Fachbegriffe ankommen, die erkannte Sprache und die Zeit.

```sh
uv run flay record                                 # 25 Sätze einsprechen (~5 Minuten)
uv run flay eval-stt --set mine                    # Parakeet, Canary, Whisper, Whisper+Begriffe
uv run flay eval-stt --set synthetic               # mit Piper gesprochen (nur grob)
uv run flay eval-stt --set mine --backends canary --download   # fehlendes Modell laden
```

## Datenschutz: alles bleibt lokal

- **Kein Download im Betrieb.** Nur `flay setup` lädt Modelle (Spracherkennung und Stimmen
  von huggingface.co, Silero VAD von github.com). `flay run` arbeitet offline: fehlt ein Modell,
  bricht Flay mit Hinweis auf `flay setup` ab, statt nachzuladen. Hugging Face läuft im
  Offline-Modus, seine Telemetrie ist abgeschaltet.
- **Kein Upload.** Flay verbindet sich nur mit Programmen auf diesem Rechner: dem LLM-Server
  (`llm.base_url`, Standard Ollama auf localhost), PhpStorms MCP-Server und der eigenen
  Web-UI (beide nur 127.0.0.1). Zeigt `llm.base_url` woandershin, warnt Flay beim Start.
  **Bewusste Ausnahme:** Webseiten, die du nennst, lädt Flay zum Lesen direkt per HTTP
  (`read_webpage`, wie ein Browser, ohne Cookies und Logins). Abschalten mit
  `tools.web.direct: false` – dann lädt nur dein Browser Seiten, Flay liest sie über die
  Bedienungshilfen oder per Screenshot.
- **Geprüft:** Ein kompletter Durchlauf (Spracherkennung, LLM mit Gedächtnis-Werkzeugen,
  Screenshot mit Bildanalyse, Sprachausgabe, Web-UI) läuft in einer macOS-Sandbox, die jede
  Verbindung außer zu localhost blockiert.
- **Grenzen:** Was du in Claude abschickst, geht – wie jede Claude-Nutzung – an Anthropic;
  Flay tippt nur ein, abschicken tust du. Ollama selbst ist ein eigenes Programm (die
  Ollama-App sucht z.B. nach Updates).

## Lizenzen der Modelle und Stimmen

Alles, was Flay lädt, ist frei nutzbar:

| Komponente | Lizenz |
|---|---|
| Qwen3-VL 8B (LLM) | Apache-2.0 |
| Parakeet TDT 0.6B v3 (STT, Standard) | CC BY 4.0 – Namensnennung: „NVIDIA Parakeet TDT 0.6B v3“, ONNX-Fassung von istupakov |
| Whisper large-v3-turbo (STT, Alternative) | MIT |
| onnx-asr | MIT |
| Silero VAD | MIT |
| tmux | ISC |
| Qwen3-TTS 0.6B Base (TTS, Standard) | Apache-2.0 |
| mlx-audio (Qwen3-TTS auf Apple Silicon) | MIT |
| Stimmen Hokuspokus, Ramona Deininger-Schnabel (LibriVox) | gemeinfrei |
| Piper (TTS-Engine) | GPL-3.0 |
| Stimme Kerstin (de, Piper) | CC0 |
| Stimme Thorsten (de, Qwen3-Stimme und Piper-Alternative) | CC0 |
| Stimme Kristin (en) | gemeinfrei |
| Qwen3-VL 4B (Ortungsmodell), Qwen3-Embedding 0.6B (Gedächtnis-Suche) | Apache-2.0 |
| openWakeWord: Code, Melspektrogramm- und Embedding-Modell (Google speech_embedding) | Apache-2.0 |
| „Hey Flay“-Erkenner (`src/flay/data/hey_flay.npz`, selbst trainiert) | wie Flay |
| Piper-Mehrsprecher-Stimmen fürs Training des Aktivierungsworts: LibriTTS-R (en_US), MLS (de_DE), VCTK (en_GB) | CC BY 4.0 – Namensnennung: LibriTTS-R (Koizumi et al., Google), Multilingual LibriSpeech (Pratap et al., Meta), CSTR VCTK Corpus (University of Edinburgh); Piper-Modelle von rhasspy |
| Piper-Stimme Thorsten emotional (de, Training des Aktivierungsworts) | CC0 |
| pypdf (PDF lesen) | BSD-3-Clause |
| comtypes (UI Automation unter Windows) | MIT |

Nicht verwendet: Llama (eigene Lizenz, Bildmodelle in der EU ausgeschlossen), Gemma (eigene
Nutzungsbedingungen), Piper-Stimme `lessac` (nur Forschung), macOS-`say` (nur privat, nicht
kommerziell), die fertigen Aktivierungswort-Modelle und Trainingsdaten (ACAV100M) von
openWakeWord (CC BY-NC-SA, nicht kommerziell).

## Historie

Ältere Stände sind als Tags archiviert: `archive/v0-prototype` (Vosk/MMS-TTS, 2025).
Der Zwischenstand mit Gemma 4 via transformers + EmotionEncoder (2026) liegt nur lokal als
`archive/v1-transformers`, weil er private Gesprächsdaten enthält.
