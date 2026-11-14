# Cassi

**Cassi ist ein Sprachassistent, der komplett lokal auf dem eigenen Rechner läuft** – macOS
(Apple Silicon) und Windows, Deutsch und Englisch. Du sprichst (per Taste oder mit „Hey Cassi“),
Cassi versteht dich mit einer lokalen Spracherkennung, denkt mit einem lokalen Sprachmodell
nach, das auch Bilder versteht, und antwortet mit eigener Stimme. Gespräche, Aufnahmen und
Screenshots verlassen den Rechner nicht.

Cassi ist für die Arbeit am Rechner gedacht:
- Projekte in PhpStorm öffnen und schließen.
- Aufträge an Claude Code geben; abschicken tust du selbst.
- Den Bildschirm ansehen und dort klicken.
- Webseiten und lokale Dateien lesen und zusammenfassen.
- Formulare ausfüllen und Abläufe wiederholen.
- Sich Dinge merken.

Dazu gibt es eine Web-UI mit Chat und ein Symbol in der Menüleiste bzw. im Tray. Was als
Nächstes kommt, steht in [docs/roadmap.md](docs/roadmap.md).

## Sicherheitshinweise: Cassi bedient deinen Rechner

Cassi handelt mit deinen Rechten: Es klickt, tippt, drückt Tasten, öffnet und schließt
Programme, füllt Formulare aus und führt Befehle aus, die du in `cassi.yaml` einträgst. Die
Entscheidungen trifft ein lokales Sprachmodell, und das kann sich irren: Es kann dich falsch
verstehen, ein Element auf dem Bildschirm verwechseln oder an die falsche Stelle klicken oder
tippen. Die Nutzung erfolgt auf eigenes Risiko.

Eingebaute Sicherungen: Vor Klicks auf „Löschen“, „Senden“ u. ä. und vor Enter fragt Cassi
nach, ebenso vor weitreichenden Schritten in einem Plan. Formulare füllt es nur nach „ja“ aus
und schickt sie nie ab, Passwörter und Zahlungsdaten tippt es nie, in Claude Code schickt es
nie selbst ab, und es führt nur Befehle aus, die du vorher eingetragen hast. Diese Sicherungen
verringern das Risiko, schließen Fehler aber nicht aus.

Empfehlungen:

- **Beim Bedienen zusehen**, besonders bei „Bedien den Rechner …“-Aufgaben und Abläufen, und
  Rückfragen nur bestätigen, wenn du weißt, was passiert.
- **Wichtiges sichern:** aktuelle Backups, Arbeit gespeichert, Änderungen in Git committet.
- **Sensibles schließen**, bevor Cassi den Bildschirm bedient: Passwortmanager, Online-Banking,
  Admin-Oberflächen, produktive Server, vertrauliche Dokumente. Cassi sieht, was auf dem
  Bildschirm ist, und könnte dort etwas ändern.
- **Nur harmlose Befehle eintragen** (`tools.commands`): nichts, was Daten löscht,
  veröffentlicht oder deployt.
- **Dauerhaftes Zuhören bewusst einschalten:** Ohne Aktivierungswort reagiert Cassi auf alles,
  was es hört – auch auf Gespräche im Raum, Videos oder Anrufe.
- **Anhalten:** Sprechen oder die Taste unterbricht Cassi; ganz beenden über „Beenden“ im
  Menüleisten-/Tray-Symbol oder Strg+C im Terminal.

## Schnellstart

Getestet mit einem Mac mit M2 Pro (32 GB) und unter Windows mit einer RTX 4070 Ti (12 GB).
Das Installationsskript installiert, was fehlt ([uv](https://docs.astral.sh/uv/) samt
Python 3.12, [Git LFS](https://git-lfs.com) für die mitgelieferte Stimme,
[Ollama](https://ollama.com) ≥ 0.30 als lokalen Modell-Server), startet Ollama, lädt einmalig
alle Modelle – Spracherkennung, Stimmen und das Sprachmodell (zusammen ~15–20 GB) – und
richtet den Befehl `cassi` ein. Danach läuft alles offline. Es kann beliebig oft laufen
(= Reparieren); was schon da ist, wird übersprungen. Zum Schluss öffnet sich die
[Verwaltung](#verwaltung-reparieren-updaten-deinstallieren).

**macOS** (installiert per [Homebrew](https://brew.sh)): `install.command` im Finder
doppelklicken – oder ohne vorherigen Download im Terminal:

```sh
curl -fsSL https://raw.githubusercontent.com/minzink/local-assistant/main/install.command | bash
```

**Windows** (installiert per `winget`): `install.cmd` doppelklicken – oder ohne vorherigen
Download in PowerShell:

```powershell
irm https://raw.githubusercontent.com/minzink/local-assistant/main/install.ps1 | iex
```

Ohne Checkout lädt das Skript Cassi nach `~/cassi` (bzw. `$CASSI_HOME`). Danach startet
Cassi mit `cassi run` (in einem neuen Terminal, falls der Befehl gerade erst eingerichtet
wurde). Optionen: `-NoInstall` / `--no-install` prüft nur und installiert keine Programme,
`-SkipModels` / `--skip-models` lässt das Laden der Modelle aus. Jeder Lauf schreibt ein
ausführliches Log nach `logs/setup-<Zeitstempel>.log` (Systeminfos, Proxy-Einstellungen,
GPU/Treiber, Ausgabe aller Schritte) – bei Problemen dieses Log mitschicken.

### Verwaltung: Reparieren, Updaten, Deinstallieren

`cassi manage` (oder „Verwaltung öffnen …“ im Menüleisten-/Tray-Symbol) öffnet eine kleine
Seite im Browser. Oben rechts steht, ob Cassi läuft, daneben **„▶ Cassi starten“** (bzw.
„Chat öffnen“, wenn es läuft) und **„■ Beenden“**. Beenden trifft, was auf Cassis Chat-Port
läuft – auch eine alte oder kaputte Installation –, aber nur, wenn es Cassi (oder z.B. Flay)
ist; ein anderes Programm auf dem Port bleibt unangetastet. Darunter:

| Tab | Inhalt |
|---|---|
| Übersicht | Version, Programm- und Datenordner (mit Größe), Befehl, Konfiguration, Autostart, Git-Stand, Ollama und Sprachmodell |
| Wartung | **Diagnose** (`cassi doctor`), **Reparieren** (das Setup erneut), **Nach Updates suchen**, **Updaten** (neuester Stand, nur ohne eigene Änderungen, danach Reparieren), Autostart an/aus, Daten- und Log-Ordner öffnen |
| Daten | **Exportieren** (Zip in den Download-Ordner, auf Wunsch mit Aufnahmen), **Importieren** (Export-Datei auswählen; die bisherigen Daten werden vorher gesichert), Sicherungen öffnen |
| Deinstallieren | entfernt Autostart und Befehl; auf Wunsch auch den Datenordner (Gedächtnis, Aufnahmen) und Cassis Modelle in Ollama. Den Programmordner löschst du danach selbst. |

Es läuft immer nur eine Verwaltung je Programm: Ein weiterer Aufruf (Terminal, Tray-Menü,
Installationsskript) öffnet die schon laufende im Browser, statt eine neue zu starten (Adresse in
`<Datenordner>/manage-cassi.json`, nur für dich lesbar). Adapter wie
[Flay](https://github.com/minzink/flay) fügen eigene Tabs hinzu. Unter Windows
laufen Reparieren und Updaten in einem eigenen Fenster, weil Windows Dateien eines laufenden
Programms nicht ersetzen kann; die Verwaltung schließt sich dafür. Dasselbe ohne Browser:
`cassi update [--check]`, `cassi uninstall [--data] [--models]`.

**Neue Version übernimmt von selbst:** Läuft Cassi, während eine neuere Version installiert
wird (Updaten, `git pull`, neue Installation, die der Befehl `cassi` jetzt startet), merkt es
das binnen einer Minute. Im Chat steht dann „Neue Version: … Ich starte neu, sobald ich nichts
zu tun habe“; ist keine Antwort und keine Aufnahme mehr im Gang, startet Cassi die neue Version
an seiner Stelle (ohne neuen Browser-Tab, der Chat verbindet sich neu und zeigt „Neu gestartet
mit …“). Timer bleiben erhalten. Wurde die Installation gelöscht und es gibt nichts zu starten,
beendet sich Cassi und gibt den Port frei. Abschalten: `updates.switch: false` (z.B. beim
Entwickeln, sonst startet jeder Commit Cassi neu).

Die Seite läuft nur auf 127.0.0.1, und jede Aktion braucht einen Schlüssel, den nur die von
`cassi manage` geöffnete Adresse enthält – andere Webseiten im Browser können nichts auslösen.

<details>
<summary>Von Hand einrichten</summary>

```sh
git lfs install && git lfs pull
ollama serve                         # bzw. Ollama-App starten
uv sync
uv run cassi setup    # lädt Spracherkennung, Stimmen und die Ollama-Modelle
uv run cassi install-command   # Befehl cassi in ~/.local/bin
```

`cassi setup` lädt fehlende Ollama-Modelle selbst (`qwen3-vl:8b-instruct`,
`qwen3-embedding:0.6b`, mit genug Speicher auch `qwen3-vl:4b-instruct` fürs Klicken).
</details>

**Etwas klappt nicht oder ist langsam?** `uv run cassi doctor` prüft GPU, Ollama, Kontext,
Grafikspeicher, Stimme, Mikrofon, Berechtigungen, PhpStorm und tmux und nennt zu jedem
Problem die Abhilfe.

**„Connection error“ / „All connection attempts failed“ beim Start:** Auf dem Port des
LLM-Servers (Ollama: 11434) hat niemand geantwortet. Cassi wartet beim Start bis zu 45 s auf
einen Server, der noch hochfährt (z.B. Ollama direkt nach der Anmeldung), und nennt danach die
Ursache: Ollama läuft nicht, der Port ist von Windows reserviert (Hyper-V/WSL 2/Docker,
`netsh interface ipv4 show excludedportrange protocol=tcp`), `OLLAMA_HOST` zeigt auf einen
anderen Port oder `llm.base_url` in der `cassi.yaml` stimmt nicht. Das Setup-Skript prüft all
das ebenfalls und schreibt es ins Log.

Dann die **rechte Option-Taste** (macOS) bzw. **rechte Strg-Taste** (Windows) halten,
sprechen und loslassen, z.B. „Wie spät ist es?“. Die Web-UI öffnet sich unter
<http://127.0.0.1:8765>; dort kannst du auch tippen und das dauerhafte Zuhören einschalten.

Unter macOS braucht das Terminal einmalig die Berechtigungen *Mikrofon*, *Bedienungshilfen* und
*Eingabeüberwachung*, Details siehe unten.

Eigene Einstellungen kommen in eine `cassi.yaml`; die Vorlage ist `cassi.example.yaml`.
Statt Ollama geht jeder Server mit OpenAI-kompatibler API, z.B. LM Studio oder llama.cpp
(`llm.base_url`, `llm.model`). `uv run cassi info` zeigt die erkannte Hardware und die
verwendete Konfiguration.

Alles Weitere – eigene Stimme, Aktivierungswort, Claude Code, Gedächtnis – steht in den
folgenden Abschnitten.

## Starten

```sh
uv run cassi run              # Push-to-Talk-Taste halten, sprechen, loslassen
uv run cassi run --continuous # dauerhaft zuhören, einfach sprechen
uv run cassi run --terminal   # Enter startet, Enter stoppt (ohne Hotkey-Berechtigung)
```

**Dauerhaftes Zuhören** (`--continuous` bzw. `input.mode: continuous`, oder jederzeit über den
Schalter „Zuhören“ in der Web-UI bzw. „Dauerhaft zuhören“ im Menüleisten-Symbol; Push-to-Talk
funktioniert immer zusätzlich): Silero VAD erkennt,
wann du sprichst; eine Pause von 600 ms beendet den Satz. Schon bei der ersten kurzen Pause
beginnt die Spracherkennung, damit die Antwort früher kommt. Sprichst du, während Cassi redet,
verstummt es (Barge-in) – das braucht ein Headset; über Lautsprecher würde Cassi sich selbst
hören, dann `input.continuous.barge_in: false` setzen.

**Aktivierungswort „Hey Cassi“** (Schalter „Hey Cassi“ in der Web-UI bzw. „Nur auf „Hey Cassi“
reagieren“ im Menüleisten-Symbol, jederzeit; beim Start an mit
`input.continuous.wake_word: true`; braucht einmal `cassi setup`): Beim dauerhaften Zuhören
zählt dann nur, was mit „Hey Cassi“ beginnt („Hey Cassi, öffne flay“); „Hey Cassi“ allein
beantwortet Cassi mit „Ja?“ und nimmt den nächsten Satz ohne. Ein kleiner, selbst trainierter
Erkenner filtert (auf den Merkmalsmodellen von openWakeWord, Apache 2.0), der erkannte Text
entscheidet: ohne „Hey Cassi“ am Anfang wird nichts ausgeführt. Gespräche im Raum unterbrechen
Cassi dann nicht mehr; „Hey Cassi …“ schon. Mit eigenen Aufnahmen wird der Erkenner besser:
`cassi voice wake` (20× „Hey Cassi“), dann `cassi train-wake`. Ein anderes Aktivierungswort
(etwa von einem Adapter) besteht aus zwei Einstellungen: `input.continuous.wake_model` (der
Erkenner, eine `.npz` wie von `train-wake`) und `input.continuous.wake_pattern` (regulärer
Ausdruck dafür, was die Spracherkennung am Satzanfang daraus macht).

Unterbrichst du Cassi (per Taste oder Sprache), merkt es sich nur, was es bis dahin gesagt hat,
und weiß in der nächsten Antwort, dass du es unterbrochen hast – auch wenn es noch gar nichts
gesagt hatte. So bleibt eine abgebrochene Frage im Zusammenhang.

**Menüleiste/Tray:** Ein Punkt zeigt den Zustand (grau bereit, orange hört zu, blau denkt
nach, grün spricht, roter Strich = Mikrofon stumm). Das Menü schaltet Mikrofon und Sprache
stumm, schaltet dauerhaftes Zuhören und „Hey Cassi“, beendet eine laufende Ablauf-Aufnahme,
öffnet die Web-UI und beendet Cassi. Abschalten: `--no-tray` bzw. `ui.tray: false`.

**Autostart:** `uv run cassi autostart on` startet Cassi bei jeder Anmeldung im Hintergrund
(`off` entfernt es, ohne Argument zeigt es den Status). macOS fragt danach für den
Hintergrundprozess erneut nach Mikrofon, Bedienungshilfen und Bildschirmaufnahme.

Mit `cassi run` öffnet sich außerdem die **Web-UI** unter <http://127.0.0.1:8765>:
Chatverlauf (gesprochene und getippte Nachrichten, Antworten live mitgeschrieben), ein
Eingabefeld für Text sowie Schalter für **Zuhören** (dauerhaft), **Hey Cassi**
(Aktivierungswort), **Mikrofon** (Push-to-Talk wird ignoriert) und **Sprachausgabe** (Cassi
antwortet nur als Text). Während ein Ablauf aufgenommen wird, zeigt ein Balken das an; „Fertig“
speichert ihn wie „Ablauf fertig“. Die UI ist nur auf diesem Rechner
erreichbar und nimmt WebSocket-Verbindungen nur von der eigenen Seite an. Abschalten mit
`cassi run --no-ui` oder `ui.enabled: false`.

Beim ersten Start werden das Whisper-Modell (~1,6 GB), die Stimme (Qwen3-TTS, ~2 GB; dazu
die Piper-Stimmen) und das Silero-VAD-Modell (2 MB) geladen.
Standardtaste: rechte Option-Taste (macOS) bzw. rechte Strg-Taste (Windows),
änderbar über `input.push_to_talk_key`. Wer spricht, während Cassi antwortet, unterbricht die
Antwort.

**macOS:** Die Terminal-App (bzw. PhpStorm, wenn dort gestartet) braucht in
*Systemeinstellungen → Datenschutz & Sicherheit* die Berechtigung *Mikrofon* und für den Hotkey
*Bedienungshilfen* sowie *Eingabeüberwachung*. Ohne Mikrofon-Berechtigung liefert macOS still
Stille – Whisper erfindet daraus Wörter wie „you“. Cassi verwirft deshalb stumme Aufnahmen
mit einer Warnung. Nach dem Erteilen die App neu starten.

Bei Erkennungsproblemen zuerst das Mikrofon prüfen:

```sh
uv run cassi mic-test         # 4 s aufnehmen, Gerät, Pegel (dBFS) und erkannten Text anzeigen
uv run cassi -v run           # zeigt zusätzlich Länge und Pegel jeder Aufnahme
```

Sprache liegt typischerweise bei −40 bis −15 dBFS; unter −60 dBFS gilt die Aufnahme als stumm.

Weitere Befehle:

```sh
uv run cassi devices                          # Audio-Geräte (für audio.input_device/output_device)
uv run cassi mic-test                         # Mikrofon testen
uv run cassi bench                            # LLM-Latenz: erstes Token, Zeichen/s
uv run cassi bench --model phi4               # anderes Modell vergleichen
uv run cassi bench --base-url http://localhost:1234/v1 --model <name>   # anderer Server
```

Jede Runde wird mit Zeiten pro Stufe geloggt (`vad`, `stt`, `llm_first_token`,
`first_sentence`, `first_audio`, `llm_done`, `done`), gemessen ab dem Loslassen der Taste.
Die Antwort wird satzweise gesprochen, während das LLM noch schreibt.

## Werkzeuge

Cassi kann auf deinem Rechner handeln – nur über die Werkzeuge in `tools.enabled`:

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
| `forget` | „Vergiss, dass ich Tee mag“ | Löscht den passenden Fakt; passen mehrere, fragt Cassi nach |
| `ssh_connect` | „SSH auf prod web“, „Verbinde dich mit dem Server staging“ | Öffnet ein neues Terminalfenster mit `ssh <host>` – nur für Server aus deiner ssh-Config |
| `control_window` | „Öffne Firefox“, „Wechsel zu Slack“, „Schieb Firefox auf den linken Monitor“, „Leg PhpStorm und Firefox nebeneinander“, „Mach Slack zu“ | Startet, holt nach vorn, schließt, verschiebt, maximiert oder minimiert Apps (installierte und laufende) |
| `use_keyboard` | „Schreib: Viele Grüße, Timo“, „Speichern“, „Mach das rückgängig“, „Neuer Tab“ | Tippt diktierten Text ins aktive Feld (über die Zwischenablage, die danach wiederhergestellt wird) oder drückt ein Kürzel aus einer festen Liste – **nie Enter** |
| `phpstorm` | „Öffne die Datei UserController“, „Starte die Tests“, „Welche Fehler zeigt PhpStorm?“, „Was ist im Git geändert?“ | Über PhpStorms MCP-Server im vorderen (oder genannten) Projekt: Datei oder Klasse öffnen, Ausführungskonfiguration starten und das Ergebnis zusammenfassen, Probleme der offenen Datei, Git-Status |
| `run_command` | „Lass die Tests laufen“ (wenn `tests` eingerichtet ist) | Führt nur Befehle aus `tools.commands` aus; dauert es länger, sagt Cassi später, wie es ausging |
| `find_file` | „Öffne die letzte PDF aus Downloads“, „Wo liegt die Rechnung von Oktober?“ | Sucht nach Namen in Downloads, Schreibtisch und Dokumente (`tools.files.roots`), neueste zuerst, und öffnet mit der passenden App |
| `timer` | „Erinnere mich in 20 Minuten an den Tee“, „Stell einen Timer auf 5 Minuten“, „Welche Erinnerungen habe ich?“ | Stellt, listet und löscht Timer; sie überstehen einen Neustart |
| `system_control` | „Mach lauter“, „Lautstärke auf 30“, „Ton aus“, „Nächstes Lied“, „Sperr den Bildschirm“ | Lautstärke, Musik und Bildschirmsperre |
| `operate_computer` | „Stell in den Systemeinstellungen den Dunkelmodus an“, „Mach in Slack einen Kanal namens test auf“, „Sag mir Bescheid, wenn der Download fertig ist“ | Findet den Weg durch eine App selbst: schaut auf das vordere Fenster, klickt, tippt, drückt Tasten, scrollt, ordnet Fenster, wartet, beobachtet – siehe unten |
| `calendar` | „Was steht heute an?“, „Hab ich morgen Termine?“, „Wann ist mein nächster Termin?“ | Liest deine Termine (macOS: alle Kalender der Kalender-App; Windows: klassisches Outlook) – nur lesend |
| `adjust_settings` | „Sprich langsamer“, „Andere Stimme“, „Antworte kürzer“, „Schau nicht mehr auf meinen Bildschirm“ | Ändert, wie Cassi spricht und ob es auf den Bildschirm schauen darf; bleibt nach einem Neustart |
| `answer_claude` | „Ja“, „Ja, immer“, „Nein“ – nachdem Cassi gesagt hat „Claude in flay fragt: …“ | Beantwortet Claudes Freigabe-Frage im Terminal |

**Den Rechner frei bedienen:** Statt für jede Aufgabe ein eigenes Werkzeug zu haben, findet
Cassi den Weg selbst (`operate_computer`). In jeder Runde sieht das Modell das vordere Fenster
– seine beschrifteten Bedienelemente (Bedienungshilfen bzw. UI Automation) und den Text darauf
(Texterkennung) – und was es bisher getan hat, und wählt den nächsten Zug: klicken, tippen,
Tasten drücken, scrollen, Apps öffnen/wechseln/verschieben, warten, genauer hinschauen (Bild),
etwas beobachten oder fertig/nachfragen/aufgeben. Gemessen (M2 Pro): ~4 s je Zug, „Schieb
Firefox auf den zweiten Monitor“ in 2 Zügen. Höchstens `tools.computer.max_steps` Züge (12),
dann sagt Cassi, wie weit es kam. Vor Klicks auf „Löschen“, „Senden“ u. ä. und vor Enter fragt
es (Enter bei einer Suche nicht); Passwörter und Zahlungsdaten tippt es nie. Was funktioniert
hat, merkt es sich (`<data_dir>/solutions.json`) und zeigt es dem Modell bei ähnlichen Aufgaben
– beim zweiten Mal kennt es den Weg; ähnliche Ziele findet es über ihre Bedeutung
(Embedding-Modell des Gedächtnisses), nicht nur über gleiche Wörter. Nach jedem Zug wartet
Cassi, bis der Bildschirm ruhig ist (ein Menü fertig aufgeklappt), statt eine feste Zeit.
Schlägt das Modell zweimal denselben Zug vor, hört Cassi auf, statt zu kreisen. Mit `cassi
eval-computer` läuft ein Testsatz mit echten Klicks (`eval/computer.yaml`: TextEdit,
Systemeinstellungen – währenddessen Maus und Tastatur nicht anfassen); gemessen 5 von 6
richtig, ~7–11 s je einfacher Aufgabe. „Sag mir Bescheid, wenn … erscheint/fertig ist“ beobachtet
den Bildschirm im Hintergrund (alle 3 s Texterkennung, höchstens 30 min) und meldet sich.

**Eindeutige Bitten ohne Modell:** Ist neben dem Werkzeug auch alles Übrige eindeutig – ein
genau genanntes Projekt, App und Verb („Schieb Firefox auf den linken Monitor“), ein
Tastenkürzel, „Mach lauter“, „Erinnere mich in 20 Minuten an den Tee“, „Sprich langsamer“, „Hab
ich morgen Termine?“, Git-Status –, füllt Cassi die Argumente selbst aus (`cassi.direct`). Die
Aktion läuft dann nach Millisekunden statt nach ~1 s. Passt ein Name auf zwei Projekte, ist
etwas zu formulieren (ein Auftrag an Claude, ein Fakt) oder steckt mehr in der Bitte, fragt
Cassi wie bisher das Modell.

**Ungewöhnlich formuliert:** Die üblichen Formulierungen erkennen feste Muster ohne Zeitverlust.
Fragen, die nach dem Rechner klingen, aber keinem Muster entsprechen („Wirf Slack auf den
zweiten Bildschirm“, „Hab ich morgen Termine?“), ordnet das Modell zuerst ein (~0,4 s, nur für
solche Anfragen, und es läuft, während Cassi PhpStorm und Claude abfragt;
`llm.classify_requests`). Lange, ausschweifende Bitten (über 25 Wörter) verkürzt das Modell
zuerst auf ihren Kern und ordnet dann diesen ein. Verstümmelte Verben erkennt Cassi am Klang: „Slie K X
E V O“ klingt wie „Schließe kx-ewo“, „Mach mal Moewe auf“ öffnet das Projekt Moewe.

**Diktieren:** „Schreib: Hallo Komma wie geht's Fragezeichen“ tippt „Hallo, wie geht's?“
(Komma, Punkt, Fragezeichen, Ausrufezeichen, Doppelpunkt, neue Zeile, neuer Absatz);
„Streich das letzte Wort“ löscht es wieder.

**Zwischenablage:** „Übersetz, was ich kopiert habe“, „Fass die Zwischenablage zusammen“ –
der Text der Zwischenablage geht mit der Anfrage ans Modell (nur dann, nie gespeichert).

**Routinen:** „Merk dir als Feierabend: starte die Tests, dann zeig den Git-Status, dann schließ
Slack“ – später reicht „Feierabend“ (oder „Mach Feierabend“). Cassi arbeitet die Schritte als
Plan ab, mit derselben Rückfrage bei weitreichenden Schritten. „Vergiss die Routine
Feierabend“ löscht sie (`<data_dir>/routines.json`).

**Markierter Text:** „Übersetz das Markierte“, „Erklär mir den markierten Code“ – Cassi
kopiert die Markierung der vorderen App (die Zwischenablage bleibt, wie sie war) und gibt sie
dem Modell mit.

**Mehrere Schritte:** „Öffne Firefox, dann mach lauter und danach stell einen Timer auf 5
Minuten“ – Cassi nennt die Schritte und arbeitet sie nacheinander ab; jeder Schritt sieht, was
die vorigen ergeben haben, ein fehlgeschlagener Schritt stoppt den Rest. Ist ein Schritt
weitreichend (Claude, Befehle, Tippen, Formulare), fragt Cassi vorher „Soll ich loslegen?“.

**Von sich aus:** Cassi sagt Bescheid, wenn ein Timer abläuft, ein länger laufender Befehl
fertig ist und wenn Claude (mit `cassi claude` gestartet) nach mindestens 20 s Arbeit fertig
ist oder auf eine Freigabe wartet (`tools.notifications`). Durchsagen warten, bis Cassi nichts
anderes sagt; sprichst du dazwischen, verstummt Cassi wie sonst auch.

**Damit Aktionen stimmen:** Nach einer Aktion spricht Cassi den festen Satz des Werkzeugs („flay
ist geschlossen.“) statt einer Umformulierung durch das Modell – das ist schneller und kann
nichts Falsches behaupten. Was ausgeführt wurde, bleibt im Verlauf, sodass Cassi bei
Nachfragen weiß, was wirklich passiert ist. Zu jeder Anfrage liest Cassi, welche Projekte in
PhpStorm offen sind und für welche Claude läuft, und korrigiert Projektnamen und
Fachbegriffe in der Spracherkennung („Symphony“ → „Symfony“, „Stadt Wolkental“ → „Stadt Wolkenthal“).

Cassi handelt nur mit einem Projekt, das du (auch verstümmelt) genannt hast – sonst fragt es
„Welches Projekt meinst du?“. Kommt der Name beim Schließen nur ungefähr vor („KEWO“), fragt es
„Meinst du project symfony kx ewo?“ und schließt erst nach „ja“. „…, never mind“ oder „vergiss
es“ am Ende nimmt eine Anfrage zurück; „Wie schließe ich …?“ ist eine Frage, keine Aktion.

**Server per SSH:** Welche Server es gibt, liest Cassi bei jeder Anfrage aus der ssh-Config
(`~/.ssh/config` samt `Include`, unter Windows `%USERPROFILE%\.ssh\config`); Platzhalter wie
`Host *` zählen nicht. Der gesprochene Name wird wie ein Projektname abgeglichen, auch über den
`HostName` dahinter. An ssh geht nur der Host-Name aus der Config – Benutzer, Port, Schlüssel
und Jump-Hosts kommen von dort. Cassi führt auf dem Server nichts aus, es öffnet nur die
Sitzung; einen Server, den du nicht genannt hast, wählt es nie selbst. Das Terminal stellt
`tools.ssh.terminal` ein (macOS: Terminal.app oder iTerm, Windows: Windows Terminal oder
Konsole). Auf dem Mac fragt macOS beim ersten Mal, ob Cassi Terminal steuern darf.

**Klicken:** Cassi sucht das Element zuerst über die Bedienungshilfen von macOS – Beschriftung
und exakter Rahmen jedes Buttons, Menüs oder Links, vorderste App zuerst (~0,3 s). Hat es keine
Beschriftung („das Glocken-Symbol rechts oben“) oder passt sie mehrfach, schaut das Bildmodell
in einen frischen Screenshot und nennt einen Punkt; kann es ihn nicht über die Bedienungshilfen
bestätigen, schaut es ein zweites Mal genauer auf einen Ausschnitt (`tools.pointer.refine`).
Zum Orten nimmt Cassi ein kleineres Modell (Qwen3-VL 4B mit kurzem Kontext, `cassi setup` legt
es an, wenn der Speicher reicht): ~4,5 s je neuem Bild statt ~8,5 s mit dem 8B, und es bleibt
neben dem Hauptmodell geladen. Beschriftungen sind also schnell und sicher, Symbole nach
Beschreibung langsamer und weniger zuverlässig. Cassi klickt nie in sein
eigenes Fenster und fragt vor einem Linksklick auf etwas wie „Löschen“, „Senden“ oder „Push“
nach („Soll ich wirklich auf Löschen klicken?“); ein Rechtsklick öffnet nur ein Menü. Unter
Windows sucht Cassi beschriftete Elemente über UI Automation, das Gegenstück zu den
Bedienungshilfen (noch nicht auf einem echten Windows-Rechner ausprobiert; scheitert es, ortet
wie bisher das Bildmodell).

Findet sich eine Beschriftung nicht über die Bedienungshilfen (z.B. in PhpStorms
Symbolleisten), liest Cassi den Text auf dem Bildschirm mit der Texterkennung des Systems
(macOS: Vision, Windows: Windows.Media.Ocr; ~0,3 s, `tools.pointer.ocr`) und klickt auf das
gefundene Wort – erst danach fragt es das Bildmodell.

Für Symbole ohne Beschriftung nummeriert Cassi die anklickbaren Elemente im Screenshot (blaue
Rahmen mit Nummer) und lässt das Modell eine Nummer wählen statt einen Punkt zu schätzen;
findet es keine passende, nennt es wie bisher einen Punkt. Nach dem Klick vergleicht Cassi den
Bildschirm vorher und nachher (~0,6 s, `tools.pointer.verify`): Hat sich nichts geändert, sagt
es das, statt Erfolg zu melden.

**Webseiten:** „Öffne diesen Link https://www.google.com/, fasse den Inhalt zusammen und gib es
an Claude“ – Cassi sagt „Ich öffne die Seite und lese sie.“ und liest die Seite als Text, ohne
dass sich auf dem Bildschirm etwas ändert:

1. **Direkt** per HTTP (`tools.web.direct`, ~0,2 s, unsichtbar) – reicht für die meisten
   Seiten, aber ohne deine Logins und ohne Inhalte, die erst JavaScript aufbaut.
2. **Aus deinem Browser** (`tools.web.browser_text`): Die Seite öffnet sich im Hintergrund
   (der Browser kommt nicht nach vorn), Cassi liest Text und Links über die Bedienungshilfen
   (macOS, ~1,5 s) bzw. UI Automation (Windows: Chrome, Edge, Firefox; noch ungetestet) – mit
   deinen Logins. Danach schließt Cassi den Tab wieder (`tools.web.close_tabs`), sofern er
   noch die Seite zeigt, die Cassi geöffnet hat.
3. **Per Screenshot**, wenn die Seite kaum Text hat (Bilder, Canvas): Dann kommt der Browser
   nach vorn, Cassi nimmt bis zu `tools.web.max_pages` Bildschirme auf (je ~8 s).

Text ist schneller und genauer als Bilder: Befehle und Versionen kommen buchstabengenau an.
Von langen Seiten liest das Modell zuerst die Abschnitte, die zur Anfrage passen
(`tools.web.max_chars`, 8000 Zeichen), und sieht die Überschriften der übrigen; einen davon
kann es gezielt nachlesen („Inheritance“ für eine Frage nach Vererbung).
Daraus schreibt das Modell die Zusammenfassung, die Cassi in Claudes Eingabezeile tippt (ohne
Enter). Ohne „an Claude“ beantwortet Cassi die Frage zur Seite direkt. Den Link kannst du
tippen (Web-UI), sprechen („heise punkt de“, „docs punkt python punkt org slash 3“), kopieren
(„der Link aus der Zwischenablage“) oder aus Claudes letzter Antwort nehmen lassen („öffne den
Link von Claude“). Mit „diese Seite“ liest Cassi die Seite, die gerade im Browser offen ist.

**Aufgaben auf Webseiten:** „Schick Claude die Installationsschritte von docs punkt astral punkt
sh slash uv als Aufgabe“ – steht das Gesuchte nicht auf der Seite, öffnet Cassi den passenden
Link („Ich öffne Installation.“): direkt gelesene Seiten folgt es direkt, im Browser gelesene
drückt es dort im Hintergrund, ohne Maus. Ist der Text lang, liest es weiter. Höchstens
`tools.web.max_steps`-mal (Standard 4). Dann schreibt es Claude die Aufgabe mit den Details von
der Seite und der Quelle in die Eingabezeile (ohne Enter). Gemessen (M2 Pro, Modell warm):
Google zusammenfassen und an Claude ~14 s, uv-Installationsschritte über einen Link ~15–26 s.
Bei Zusammenfassungen öffnet Cassi keine weiteren Links.

**Suchen und Vergleichen:** „Such auf docs punkt python punkt org nach TaskGroup und sag mir,
wofür es gut ist“ – Cassi nutzt die Suche der Seite: ein Suchformular direkt per HTTP, sonst das
Suchfeld im Browser (im Hintergrund ausgefüllt und abgeschickt). Andere Formulare füllt es nie
aus. „Vergleiche docs punkt astral punkt sh slash uv und pip punkt pypa punkt io“ liest beide
Seiten und vergleicht sie (gemessen ~11 s). Nie geöffnet oder gedrückt werden
Suche, Anmelden/Abmelden, Kaufen, Senden u. ä.; bei Seiten aus Screenshots lehnt Cassi
Cookie-Banner ab.

**Formulare:** „Trag auf dieser Seite bei Name Max Mustermann und bei E-Mail max at example
punkt org ein“ – Cassi liest die Felder der Seite im Browser über die Bedienungshilfen (bzw. UI
Automation), ordnet sie über ihre Beschriftung zu und fragt erst: „Soll ich auf … bei Name
„Max Mustermann“ … eintragen? Abschicken musst du selbst.“ Nach „ja“ füllt es aus und holt den
Browser nach vorn. Passwortfelder füllt es nie, Formulare schickt es nie ab.

**Lokale Dateien:** „Lies die README von flay und sag mir, wie man es installiert“, „Fasse die
Roadmap im Projekt flay zusammen und gib es Claude“, „Was steht in der neuesten PDF in meinen
Downloads?“ – Cassi sucht in PhpStorms Projekten, auf dem Schreibtisch, in Downloads und
Dokumenten (Name nach Wortanfängen, „neueste“ nach Datum) und liest Text, Code, Markdown, PDF
und Word-Dateien wie eine Webseite: lange Dateien in Abschnitten, passende zuerst.

**Abläufe:** „Merk dir einen Ablauf: Release bauen“ – Cassi schaut zu, während du es einmal
machst: Klicks (mit der Beschriftung des Elements, sonst der Stelle im Fenster), Tastenkürzel,
Enter/Tab/Pfeile, Scrollen. „Ablauf fertig“ speichert ihn (`<data_dir>/workflows.json`),
„Mach den Ablauf Release bauen“ spielt ihn ab und sucht jedes Element neu – Fenster dürfen
sich verschieben. Getippter Text wird nie aufgezeichnet (könnte ein Passwort sein). Enthält
ein Ablauf Enter oder Beschriftungen wie „Löschen“, fragt Cassi vor dem Abspielen. „Welche
Abläufe kennst du?“, „Lösch den Ablauf …“.

**Denken auf Wunsch:** Mit einem Modell mit Denk-Modus (`llm.thinking`, siehe
`cassi.example.yaml`) überlegt Cassi, wenn du es darum bittest („denk mal nach, wie …“,
„Schritt für Schritt“). Es sagt zuerst „Moment, ich überlege kurz.“; der Denktext selbst wird
nie gesprochen. Für normale Befehle bringt Denken nichts (gemessen), für Rechen-, Zeit- und
Logikfragen viel: mit `qwen3-vl:8b-thinking` 100 % statt 75 % richtig, dafür 7–30 s statt
1,5 s. Damit Ollama dafür nicht jedes Mal das Modell wechseln muss, in Ollama die
Kontextlänge auf 8k stellen – dann bleiben beide Modelle geladen (je 6,1 GB statt 10 GB).

### Claude Code anbinden

`cassi` einmal global verfügbar machen (Änderungen am Code wirken sofort):

```sh
uv tool install --editable <Pfad zum Repo>
```

Dann im PhpStorm-Terminal des Projekts **`cassi claude` statt `claude`** starten (weitere
Argumente werden an Claude durchgereicht, z.B. `cassi claude --continue`). Claude läuft dort
wie gewohnt, aber in einer tmux-Sitzung pro Projekt – Cassi kann so den Auftrag direkt in die
Eingabezeile schreiben, ohne Fensterwechsel, und Claudes Status lesen. Du siehst alles live
und drückst Enter.

**Welche Sitzung?** Nennst du ein Projekt („Sag Claude im Projekt Stadt Wolkenthal …“), die
dieses Projekts. Sonst die Sitzung, die du gerade in einem Terminal offen hast (zuletzt
benutzte zuerst). PhpStorms Projektliste wird nur selten gespeichert und taugt deshalb nicht,
um „das aktuelle Projekt“ zu erkennen.

Läuft keine sichtbare Sitzung, **öffnet Cassi Claude selbst in PhpStorm**: Über PhpStorms
eingebauten MCP-Server (*Settings → Tools → MCP Server* aktivieren; Cassi findet ihn
automatisch) legt es kurz `.cassi-claude.sh` im Projekt an, lässt PhpStorm das Skript als
Run-Configuration in einem neuen Terminal-Tab ausführen (`cassi claude`, hängt sich an eine
bestehende Sitzung an), löscht die Datei wieder, wartet, bis Claude bereit ist, und tippt den
Auftrag ein. Welches Projekt: das genannte, sonst das einzige offene; sind mehrere offen,
fragt Cassi nach. Die offenen Projekte kommen live von PhpStorm.

Tipp: Trägst du in *Settings → Tools → Claude Code [Beta] → Claude command* `cassi claude`
ein, startet auch der Claude-Button von PhpStorm Claude so, dass Cassi hineinschreiben kann.

Ist ein genanntes Projekt in PhpStorm nicht geöffnet, startet Cassi Claude im Hintergrund;
mit `cassi claude` im Terminal dieses Projekts hängst du dich an. Cassi sagt dir dann, für welches Projekt; mit `cassi claude` im
PhpStorm-Terminal dieses Projekts hängst du dich an die Sitzung an und siehst den Auftrag.
Fragt Claude beim ersten Start in einem Ordner, ob du ihm vertraust, tippt Cassi nichts (dort
ist „No, exit“ vorausgewählt) und bittet dich, die Frage selbst zu beantworten.
Abschalten mit `tools.claude.autostart: false`.

Ohne tmux (Windows) oder ohne passendes Projekt legt Cassi den Auftrag in die Zwischenablage
und holt PhpStorm nach vorn; du fügst ihn mit `Cmd/Ctrl+V` in Claude ein. Erzwingen lässt sich
ein Weg mit `tools.claude.transport: tmux | clipboard`.

**Aufträge, mit denen Claude allein arbeiten kann:** Claude hört das Gespräch nicht. Cassi
schreibt den Auftrag deshalb so, dass er für sich steht: was zu tun ist und warum, wo im Code
(Datei, Klasse – soweit aus der Bitte, dem Gespräch davor oder PhpStorms Fenstertitel
bekannt), was schon bekannt ist (Fehlermeldungen, was du gesehen hast) und woran man erkennt,
dass es fertig ist. „Mach das wie eben besprochen“ kommt als echte Aufgabe an. Cassi erfindet
keine Anforderungen dazu; diktierst du den Wortlaut, bleibt er wörtlich. Entwickeln, Testen,
Committen – alles, was Code betrifft, macht Claude; Cassi formuliert nur den Auftrag.

**Freigaben per Sprache:** Wartet Claude (mit `cassi claude` gestartet) auf eine Freigabe, liest
Cassi die Frage vor: „Claude in flay fragt: Do you want to make this edit to app.py? Sag ja,
immer oder nein.“ „Ja“ / „Erlaub das“ wählt die erste Option, „Ja, immer“ die Option „nicht
mehr fragen“, „Nein“ lehnt ab. Cassi drückt nur, wenn die Frage noch auf Claudes Bildschirm
steht.

- PhpStorm fragt beim Öffnen eines Projekts standardmäßig, in welchem Fenster. Damit Projekte
  immer in einem neuen Fenster aufgehen: *Settings → Appearance & Behavior → System Settings →
  Project → Open project in: New window* (Cassi weist beim Start darauf hin).
- Erwähnst du den Bildschirm („Bildschirm“, „screen“, „was siehst du“, „schau dir das an“),
  hängt Cassi den Screenshot direkt an – unabhängig davon, ob das Modell das Werkzeug wählt.
  Dazu bekommt das Modell die offenen Fenster je Monitor als Text, so kennt es die Namen,
  ohne sie aus dem Bild zu lesen. Monitore lassen sich auch nach ihrer Lage nennen („der
  linke Monitor“, „mein anderer Bildschirm“).
- „Was ist das hier?“, „Erklär mir das hier“, „Was steht unter der Maus?“ zeigt dem Modell
  nur den Bereich um den Mauszeiger (640×400 Punkte) und sagt ihm, wo der Zeiger steht. Das
  ist nicht schneller als ein ganzer Bildschirm (Ollama bringt jedes Bild auf ~1100 Token),
  aber das Modell sieht die gemeinte Stelle schärfer und lässt sich vom Rest nicht ablenken.
- „Welche Programme sind offen?“, „Was habe ich alles offen?“ beantwortet Cassi nur aus der
  Fensterliste, ohne Screenshot: auf einem M2 Pro kam die Antwort nach ~3,5 s, ein Bild
  allein kostet dort ~7,5 s.
- Screenshots bleiben lokal. Was du an Claude weitergibst, geht – wie jede Claude-Nutzung –
  an Anthropic.
- **macOS-Berechtigungen** für die App, aus der Cassi läuft: *Bildschirmaufnahme* (ohne sie
  ist das Bild leer und Cassi sagt das) und *Bedienungshilfen* (Hotkey, Klicken und Scrollen –
  ohne sie verwirft macOS Mausereignisse stillschweigend; Cassi fragt beim ersten Klick danach).
- Werkzeug-Aufrufe erscheinen in der Web-UI im Chatverlauf. Der Button unten rechts öffnet
  das **Debug-Log**: alle Logs von Cassi, Aktionen auf dem Rechner (Projekt geöffnet, an Claude
  geschrieben, Screenshot …) hervorgehoben und per „Nur Aktionen“ filterbar.
- Beim Start loggt Cassi, welche `cassi claude`-Sitzungen es findet.

## Persönlichkeit und Gedächtnis

**Persönliche Daten liegen nur im Datenordner, nie im Repository** (macOS
`~/Library/Application Support/cassi`, Windows `%LOCALAPPDATA%\cassi`): Gedächtnis, Verlauf,
gelernte Abläufe und Lösungen, Einstellungen, Timer, selbst trainierte Modelle (z.B.
`models/hey_cassi.npz` aus `cassi train-wake`) und die eigene Stimme. Mitgelieferte Standards
(Aktivierungswort, Stimme Ramona) bleiben im Repository und gelten, solange es kein eigenes gibt.
Mitnehmen oder sichern: `cassi export [Datei] [--recordings]` und `cassi import Datei`, oder der
Tab „Daten“ der Verwaltung. Ein Export ist eine Zip-Datei; er enthält keine Downloads (die lädt
`cassi setup` neu), Aufnahmen nur mit `--recordings`. Ein Import nimmt nur Exporte desselben
Assistenten an (Flays Gedächtnis ist nicht Cassis), sichert vorher die bisherigen Daten nach
`backups/` und geht nur, wenn der Assistent nicht läuft. Heruntergeladene Modelle liegen in
`downloads_dir` (Standard: der Datenordner); ein Adapter mit eigenem Datenordner teilt sie.

- **Persona:** `persona.system_prompt` und `persona.mood` (frei benannte Stimmungen von 0–10,
  z.B. `cheerful: 6`) fließen in den System-Prompt ein, zusammen mit Datum und Uhrzeit.
- **Gesprächsverlauf:** Die letzten `memory.recent_messages` Nachrichten der Sitzung gehen
  wörtlich ans LLM, ältere werden zusammengefasst. Nach `memory.session_timeout_minutes`
  Pause beginnt eine neue Sitzung – ebenso mit dem Knopf **„Neues Gespräch“** in der Web-UI
  (bzw. „Neues Gespräch (Kontext leeren)“ im Tray-Menü): Der bisherige Verlauf verlässt den
  Kontext, bleibt aber gespeichert; Gelerntes (Fakten) bleibt, eine offene Ja/Nein-Rückfrage
  verfällt. Startet ein anderer Assistent oder ein anderes Modell als zuletzt (z.B. Cassi nach
  Flay – beide teilen den Datenordner), beginnt automatisch ein neues Gespräch, mit Hinweis im
  Chat.
- **Langzeit-Gedächtnis (RAG):** Cassi sammelt im Hintergrund Erinnerungen:
  - *Fakten* über dich, deine Projekte, Entscheidungen und Termine („Merk dir, dass ich am
    Freitag Urlaub habe“, „Im Projekt umx nutzen wir Redis“) – relative Daten werden zu
    absoluten, widerrufene Fakten gestrichen.
  - *Episoden:* Ist ein Gespräch vorbei, wird es zusammengefasst („Am 26.10. wollte der
    Nutzer …“), sodass Cassi später darauf zurückgreifen kann („Was haben wir letztes Mal
    besprochen?“).
- **Datum und Wichtigkeit:** Jede Erinnerung trägt, wann sie geschrieben wurde, und eine
  Wichtigkeit: *wichtig* sind dauerhafte Regeln und Anweisungen („Gib Claude in Zukunft keinen
  Befehl mehr, etwas zu committen“) und was du als wichtig bezeichnest – sie gelten immer und
  stehen bei jeder Anfrage im Prompt (auch beim Formulieren von Aufträgen an Claude);
  *normal* ist dauerhaftes Wissen; *nebensächlich* sind einmalige Ergebnisse („die README
  wurde heute nicht angepasst“) und Gesprächszusammenfassungen – sie zählen beim Suchen weniger.
- **Beispiele aus dem Verlauf:** Führt Cassi eine Aktion aus und widersprichst du nicht
  („nein“, „falsch“), merkt es sich den Satz, wie es ihn gehört hat, mit der Aktion
  („She K Command.“ → Projekt component-symfony-kx-command schließen). Ähnliche Anfragen
  bekommen diese Beispiele mit – so lernt Cassi deine Sprechweise und die typischen Hörfehler.
  Hast du eine Aktion mit „ja“ bestätigt oder meinte derselbe Satz zweimal dasselbe, führt
  Cassi ihn beim nächsten Mal direkt aus. Gemessen: 98,7 % statt 92,1 % Treffer, keine
  Rückfrage mehr statt der Aktion (`cassi eval --examples eval/examples.yaml`).
- **Abruf:** Zu jeder Anfrage sucht Cassi die passenden Erinnerungen – nach Bedeutung
  (Embedding-Modell `qwen3-embedding:0.6b`, Apache 2.0, ~640 MB, über Ollama) und nach
  Stichwörtern (SQLite FTS5) – und gibt die besten (höchstens `memory.recall.limit`) mit
  Datum an das Modell. Die Suche dauert ~20–30 ms und läuft parallel zur Weiterleitung;
  ohne Embedding-Modell sucht Cassi nur nach Stichwörtern. Einmalig:
  `ollama pull qwen3-embedding:0.6b`.
- Die Hintergrundarbeit startet erst nach `memory.idle_seconds` Ruhe und wird abgebrochen,
  sobald du wieder sprichst – sie verlängert also keine Antwortzeit.

Alles liegt lokal in `<data_dir>/memory.sqlite3` (Pfad: `cassi info`). Verwalten:

```sh
uv run cassi memory               # Erinnerungen (Fakten und Episoden) anzeigen
uv run cassi memory search Urlaub # suchen wie Cassi es tut
uv run cassi memory forget 3 7    # einzelne Erinnerungen löschen
uv run cassi memory clear-chat    # nur Gesprächsverlauf + Zusammenfassungen löschen
uv run cassi memory clear         # Gespräche, Zusammenfassungen und Fakten löschen
```

Mit `memory.enabled: false` merkt sich Cassi nur die laufende Unterhaltung im Arbeitsspeicher.

**Antwortzeit:** Cassi hält das Modell warm (`llm.keep_warm_minutes`), beginnt mit dem
Sprechen nach dem ersten Satzstück und erkennt beim dauerhaften Zuhören schon während der
Pause. Gemessen auf einem M2 Pro mit Parakeet: ~0,85 s vom Loslassen der Taste bzw. ~1,2 s vom
Sprechende (davon 0,6 s Warten auf das Satzende) bis zum ersten Ton. Mit Whisper
(`stt.backend: auto`, etwas genauer) sind es ~1,4 bzw. ~1,55 s.
Statt eines vorgefertigten „Hm“ spricht Cassi den ersten Satz der Antwort, sobald das Modell
ihn geschrieben hat (im Median ~0,9 s nach dem Sprechende), während der Rest noch entsteht. Nur wenn nach 1,2 s noch nichts gesprochen ist –
bei Aktionen, die das Modell erst ausfüllen muss –, sagt Cassi „Moment.“ (`tts.filler`,
abschaltbar; Füllwörter für Antworten lassen sich mit `tts.filler.phrases` wieder einschalten).

**Stimme:** Cassi spricht Deutsch und Englisch mit derselben Stimme. Qwen3-TTS (0.6B, Apache
2.0) ahmt dafür eine kurze Aufnahme nach (`tts.voice.name`):

| Stimme | |
|---|---|
| `auto` (Standard) | die eigene Aufnahme, falls vorhanden, sonst `ramona` |
| `own` | deine Stimme: `cassi voice record` nimmt einen deutschen und einen englischen Text auf (je ~10 s, bleibt lokal) |
| `ramona`, `hokuspokus` | Sprecherinnen von LibriVox (gemeinfrei) |
| `thorsten` | Sprecher (CC0) |
| Pfad zu einer WAV | eigene Aufnahme mit dem gesprochenen Text in einer `.txt` daneben |

**Eigene Piper-Stimme:** `cassi voice dataset` zeigt die 965 Sätze des Stimmen-Trainings
nacheinander zum Vorlesen (prüft jede Aufnahme auf Stille, Übersteuerung und Tempo; jederzeit
unterbrechen und weitermachen). 30–60 Minuten ergeben eine eigene, schnelle Piper-Stimme für
beide Sprachen; trainiert wird auf dem Windows-Rechner mit
`tools/voice-training/dataset.ps1 -Recorded <Ordner>` und `train.ps1 -Name cassi-<name>`
(siehe `tools/voice-training/README.md`).

`tts.voice.pitch` hebt oder senkt die Stimme in Halbtönen, `tts.voice.speed` ändert das Tempo
(1.2 = 20 % schneller). Beides wird einmal beim Start auf die Aufnahme angewendet und kostet
beim Sprechen keine Zeit. Probehören: `cassi voice test` (oder `--name ramona`).
Auf einem M2 Pro (MLX, 4 Bit) kommt der erste Ton nach ~0,2 s; ein Satz braucht ~0,4 s je
Sekunde Sprache. Unter Windows nimmt Cassi Piper (`tts.backend: auto`) mit Cassis eigener
Ramona-Stimme `cassi-ramona-medium` – dieselbe Stimme wie Qwen3-TTS, für beide Sprachen, erster
Ton nach ~0,05 s. Sie kommt mit dem Repository (Git LFS) und wird beim Start bzw. von
`cassi setup` in den Stimmen-Ordner übernommen. Fehlt sie (ohne `git lfs pull`), spricht Cassi
mit Kerstin (de) und Kristin (en). Qwen3-TTS mit PyTorch (`tts.backend: qwen3`) erzeugt ganze Sätze statt
zu streamen – auf einer RTX 4070 Ti ~4,5 s bis zum ersten Ton – und teilt sich den
Grafikspeicher mit dem LLM, was beide stark ausbremst.

**Ollama:** Unter 24 GB Grafikspeicher gibt Ollama Modellen nur 4096 Token Kontext – zu
wenig für Cassis Anweisungen samt Werkzeugen und Screenshot. Cassi legt dort beim Start eine
Variante mit 8k Kontext an (z.B. `qwen3-vl:8b-instruct-ctx8k`, kein Download) und nutzt sie.
Längere Anfragen kürzt Cassi vorher (`llm.context_tokens`, älteste Nachrichten zuerst). Nach
dem Start prüft Cassi, ob die Modelle ganz im Grafikspeicher liegen, und warnt im Log, wenn
Ollama einen Teil auf die CPU legt – dann werden Antworten um ein Vielfaches langsamer.
Ollama entlädt Modelle nach 5 Minuten Leerlauf; die nächste Antwort dauert dann
2–3 s länger. Abhilfe: `OLLAMA_KEEP_ALIVE=1h` (oder `-1`) für den Ollama-Server setzen.

## Konfiguration

Standardwerte stehen im Code (`src/cassi/config.py`). Eigene Werte in `cassi.yaml` eintragen
(Vorlage: `cassi.example.yaml`); alternativ Pfad per `--config` oder `$CASSI_CONFIG`.
Unbekannte Schlüssel und ungültige Werte werden beim Start abgelehnt.

## Als Paket einbinden: Adapter

Cassi läuft eigenständig. Andere Projekte können Cassi aber als Paket (`cassi`) einbinden und
als **Adapter** personalisieren – etwa [Flay](https://github.com/minzink/flay), das ein
eigenes Modell mitbringt. Ein Adapter ist eine Unterklasse von `cassi.adapters.Adapter` und
meldet sich über einen Entry Point an:

```toml
[project.entry-points."cassi.adapters"]
flay = "flay.adapter:FlayAdapter"
```

Cassi ruft zwei Hooks auf:

| Hook | Wann | Wofür |
|---|---|---|
| `configure(settings)` | nach dem Laden der Konfiguration, bei jedem Befehl | Einstellungen ändern (Modell, Persona, …); kein Netzwerk, keine Dateien |
| `prepare(settings, download)` | vor `run`, `eval`, `bench`, `eval-computer` (`download=False`) und in `cassi setup` (`download=True`) | Modelle bereitstellen; nur in `setup` darf heruntergeladen werden |
| `manage_tabs(settings)` | beim Öffnen der Verwaltung (`cassi manage`) | eigene Tabs: Infos (`Section`, `Row`) und Aktionen (`Action`, mit Ausgabe auf der Seite) aus `cassi.manage` – als Daten, ohne HTML |
| `uninstall(settings, models)` | beim Deinstallieren | entfernen, was der Adapter außerhalb seines Ordners angelegt hat (mit `models`: auch seine Modelle) |

In `configure` ändert ein Adapter alles, was auch `cassi.yaml` einstellen kann – etwa
`llm.model`, `persona` (der Name erscheint im Chat, im Tray-Menü und im Prompt) oder ein eigenes
Aktivierungswort (`input.continuous.wake_model`, `wake_pattern` und `wake_phrase` für die
Anzeige, wie Flays „Hey Flay“).

Welche installierten Adapter aktiv sind, steuert `adapters` in `cassi.yaml` (fehlt der Eintrag:
alle; `[]`: keiner). Ein Adapter, der sich nicht laden lässt, bricht den Start mit einer
Meldung ab, statt still zu fehlen.

## Architektur

```
src/cassi/
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
├── adapters.py Hooks für Pakete, die Cassi personalisieren (z.B. Flay)
├── app.py      welches Programm läuft (cassi oder z.B. flay) und wo es installiert ist
├── maintenance.py  Befehl einrichten, Reparieren, Updaten, Deinstallieren
├── manage.py   Verwaltung im Browser (cassi manage): Tabs, Aktionen, lokaler Server
├── updates.py  laufendes Cassi wechselt zu einer neueren installierten Version
└── cli.py      Einstiegspunkt `cassi`
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

Einmal je Klon die Git-Hooks einschalten: `git config core.hooksPath .githooks`. Sie prüfen
jeden Commit und jede Commit-Nachricht mit `scripts/check_private.py` auf private Daten (echte
Projekt- und Kundennamen, E-Mail-Adressen, Benutzerordner, Server). Die Begriffe sammelt das
Skript auf dem eigenen Rechner (Projektordner und weitere Begriffe aus
`~/.config/private-terms.txt`, Git-Identität, `~/.ssh/config`); `check_private.py tree` prüft
den ganzen Stand, `history` alle Commits.

CI prüft dasselbe auf macOS und Windows. Code-Aufteilung: `cassi.intents` enthält alle festen
Erkennungsmuster (Testtabelle `tests/test_patterns.py`), `cassi.planning` Pläne und Routinen,
`cassi.assistant` den Ablauf einer Runde. Für Windows gibt es eine Prüfliste:
`docs/windows-test.md`.

### Sprachmodelle vergleichen (`cassi eval`)

`cassi eval` schickt den Testsatz aus `eval/cases.yaml` (echte, teils falsch erkannte Befehle
und normale Gesprächssätze) durch Cassi – mit Weiterleitung, Prompt und Werkzeug-Schleife wie im
Betrieb, aber mit simulierten Werkzeugen: auf dem Rechner passiert nichts. Gemessen werden die
Trefferquote (richtiges Werkzeug, richtiges Projekt; bei Gesprächssätzen keine Aktion) und die
Zeit bis zum Werkzeug-Aufruf bzw. zur ersten Antwort. Ergebnisse landen als JSON im
Datenordner unter `eval/`.

```sh
uv run cassi eval                                   # Modell aus cassi.yaml, jeder Fall 2×
uv run cassi eval --base-url http://127.0.0.1:8080/v1 --model qwen3.5-9b \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
uv run cassi eval --only 6,10,11 --repeat 5         # einzelne Fälle genauer
```

### Wohin die Zeit geht (`cassi stats --perf`)

Zu jeder Anfrage schreibt Cassi ein Profil ins Log („Profil: …“) und in den Verlauf: die
Phasen (Spracherkennung, Rechnerzustand, Einordnung, Weiterleitung, Werkzeuge, Antwort, Warten
auf die Sprachausgabe), die Zeitpunkte ab Ende der Aufnahme (erster Satz, erster Ton, fertig)
und jeden Aufruf des Modells mit Zweck, Token, Zeit bis zum ersten Token und Lesetempo.
`cassi stats --perf` fasst die letzten Tage zusammen (Median und p90), `cassi eval` am Ende
seines Laufs. Ein Lesetempo von einigen Tausend Token/s heißt: der Anfang des Prompts kam aus
dem Cache des Servers; einige Hundert: er wurde neu gelesen.

Dafür sorgt Cassi selbst: Ollama hält für qwen3-vl nur einen Prompt-Cache. Die Werkzeugliste
bleibt deshalb in jeder Runde gleich, Werkzeuge, nach denen das Modell antwortet, werden mit
dem Gesprächsprompt weitergeleitet (`llm.shared_prefix_routing`), und nach einer Runde legt
Cassi den Gesprächsanfang im Leerlauf wieder in den Cache (`llm.rewarm`) – beim Start vorab.

### Spracherkenner vergleichen (`cassi eval-stt`)

`cassi record` zeigt die Sätze aus `eval/speech.yaml` nacheinander an; du sprichst sie ein (Enter
startet und stoppt). Die Aufnahmen bleiben im Datenordner unter `eval/speech/<name>/`.
`cassi eval-stt` erkennt sie mit jedem Spracherkenner und misst Wortfehlerrate, ob Verben,
Projektnamen und Fachbegriffe ankommen, die erkannte Sprache und die Zeit.

```sh
uv run cassi record                                 # 25 Sätze einsprechen (~5 Minuten)
uv run cassi eval-stt --set mine                    # Parakeet, Canary, Whisper, Whisper+Begriffe
uv run cassi eval-stt --set synthetic               # mit Piper gesprochen (nur grob)
uv run cassi eval-stt --set mine --backends canary --download   # fehlendes Modell laden
```

## Datenschutz: alles bleibt lokal

- **Kein Download im Betrieb.** Nur `cassi setup` lädt Modelle (Spracherkennung und Stimmen
  von huggingface.co, Silero VAD von github.com). `cassi run` arbeitet offline: fehlt ein Modell,
  bricht Cassi mit Hinweis auf `cassi setup` ab, statt nachzuladen. Hugging Face läuft im
  Offline-Modus, seine Telemetrie ist abgeschaltet.
- **Kein Upload.** Cassi verbindet sich nur mit Programmen auf diesem Rechner: dem LLM-Server
  (`llm.base_url`, Standard Ollama auf localhost), PhpStorms MCP-Server und der eigenen
  Web-UI (beide nur 127.0.0.1). Zeigt `llm.base_url` woandershin, warnt Cassi beim Start.
  **Bewusste Ausnahme:** Webseiten, die du nennst, lädt Cassi zum Lesen direkt per HTTP
  (`read_webpage`, wie ein Browser, ohne Cookies und Logins). Abschalten mit
  `tools.web.direct: false` – dann lädt nur dein Browser Seiten, Cassi liest sie über die
  Bedienungshilfen oder per Screenshot.
- **Geprüft:** Ein kompletter Durchlauf (Spracherkennung, LLM mit Gedächtnis-Werkzeugen,
  Screenshot mit Bildanalyse, Sprachausgabe, Web-UI) läuft in einer macOS-Sandbox, die jede
  Verbindung außer zu localhost blockiert.
- **Private Fenster:** Solange ein Passwortmanager (1Password, Bitwarden, KeePass,
  Schlüsselbund …) oder ein Fenster mit Titeln wie „Online-Banking“, „PayPal“ oder „Privates
  Fenster“ im aufgenommenen Bereich sichtbar ist, macht Cassi keine Bildschirmaufnahme und keine
  Texterkennung und sagt das (`privacy.private_apps`, `privacy.private_titles`). „Schau nicht
  mehr auf meinen Bildschirm“ (oder der Schalter im Menüleisten-Symbol) pausiert alle
  Aufnahmen.
- **Verlauf:** Cassi schreibt jede Anfrage mit Werkzeugen, Dauer und Fehlern nach
  `<data_dir>/history.jsonl` (nur lokal, `privacy.history_days`, Standard 30 Tage, 0 = aus);
  diktierter Text und Formularwerte stehen dort nur als Länge.
  `cassi history` zeigt die letzten, `cassi stats` wertet aus (Werkzeuge, Fehler, Dauer),
  `cassi history --eval` macht aus fehlgeschlagenen Anfragen Testfälle für `eval/cases.yaml`.
  Scheitert dieselbe Bitte zweimal hintereinander, schlägt Cassi vor, es anders zu sagen (das
  merkt es sich dann); beim Start nennt das Log die Zahl der Fehlschläge der letzten Woche.
- **Grenzen:** Was du in Claude abschickst, geht – wie jede Claude-Nutzung – an Anthropic;
  Cassi tippt nur ein, abschicken tust du. Ollama selbst ist ein eigenes Programm (die
  Ollama-App sucht z.B. nach Updates).

## Lizenzen der Modelle und Stimmen

Alles, was Cassi lädt, ist frei nutzbar:

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
| „Hey Cassi“-Erkenner (`src/cassi/data/hey_cassi.npz`, selbst trainiert) | wie Cassi |
| Piper-Mehrsprecher-Stimmen fürs Training des Aktivierungsworts: LibriTTS-R (en_US), MLS (de_DE), VCTK (en_GB) | CC BY 4.0 – Namensnennung: LibriTTS-R (Koizumi et al., Google), Multilingual LibriSpeech (Pratap et al., Meta), CSTR VCTK Corpus (University of Edinburgh); Piper-Modelle von rhasspy |
| Piper-Stimme Thorsten emotional (de, Training des Aktivierungsworts) | CC0 |
| pypdf (PDF lesen) | BSD-3-Clause |
| comtypes (UI Automation unter Windows) | MIT |

Nicht verwendet: Llama (eigene Lizenz, Bildmodelle in der EU ausgeschlossen), Gemma (eigene
Nutzungsbedingungen), Piper-Stimme `lessac` (nur Forschung), macOS-`say` (nur privat, nicht
kommerziell), die fertigen Aktivierungswort-Modelle und Trainingsdaten (ACAV100M) von
openWakeWord (CC BY-NC-SA, nicht kommerziell).

## Historie

Bis November 2026 hieß Cassi „Flay“ (Paket, Befehl, `flay.yaml`, `$FLAY_CONFIG`). Der Name
Flay gehört jetzt dem Adapter, der auf Cassi aufsetzt. Der alte Datenordner (`…/flay` mit
Gedächtnis, Aufnahmen und Stimmen) ist seit der Trennung Flays eigener Datenordner; Cassis
Daten liegen in `…/cassi` (damals einmal dorthin verschoben). Eine alte `flay.yaml` muss von
Hand in `cassi.yaml` umbenannt werden.

