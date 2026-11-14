# Windows-Test: Checkliste

Alles hier ist für Windows gebaut, aber noch nicht auf einem echten Windows-Rechner gelaufen.
Bei jedem Fehler steht der Grund im Log (`cassi run -v`, Web-UI → Debug). Cassi fällt bei
Fehlern auf den bisherigen Weg zurück (Bildmodell, Zwischenablage), es sollte also nichts
abstürzen – aber langsamer oder ungenauer sein.

## 0. Vorbereitung

```
git pull
uv sync
uv run cassi doctor
```

- [ ] `cassi doctor` zeigt keine ✗. Besonders ansehen: Grafikspeicher (alles auf der GPU?),
      Kontext (8192?), Texterkennung (verfügbar?), Kalender.
- [ ] `uv run cassi eval` – Ergebnis notieren (Mac: 99 %, Median 1,66 s) und die JSON-Datei
      aus `%LOCALAPPDATA%\cassi\eval` aufheben.

## 1. Fenster und Apps (`control_window`)

- [ ] „Öffne Firefox“ (nicht offen) → startet über das Startmenü
- [ ] „Wechsel zu PhpStorm“ (offen, hinten) → kommt nach vorn (AttachThreadInput)
- [ ] „Schieb Firefox auf den linken Monitor“ / „…auf den anderen Bildschirm“
- [ ] „Leg PhpStorm und Firefox nebeneinander“
- [ ] „Minimier Slack“, „Maximier Firefox“, „Mach Slack zu“ (fragt nicht, schließt normal)

## 2. Klicken

- [ ] „Klick auf Datei“ in einer normalen App → im Log „UI Automation“, ohne Bildmodell
- [ ] „Klick auf Refactor“ in PhpStorm → im Log „Texterkennung“ (Windows.Media.Ocr)
      – falls „Texterkennung nicht verfügbar“: Einstellungen → Zeit und Sprache → Sprache →
      Deutsch → Optionen → „Optische Zeichenerkennung“ installieren
- [ ] „Klick auf das Glocken-Symbol“ (ohne Beschriftung) → nummerierte Kandidaten
- [ ] Nach einem Klick ins Leere sagt Cassi „…aber es hat sich nichts sichtbar getan“

## 3. Tastatur, System

- [ ] „Schreib: Hallo Komma wie geht's Fragezeichen“ in Notepad → „Hallo, wie geht's?“
- [ ] „Streich das letzte Wort“ (Strg+Rücktaste)
- [ ] „Speichern“, „Rückgängig“
- [ ] „Mach lauter“, „Ton aus“, „Nächstes Lied“ (Medientasten), „Sperr den Bildschirm“

## 4. Rechner frei bedienen (`operate_computer`)

- [ ] „Öffne im Editor ein neues Dokument“ – Züge im Log ansehen (`Rechner bedienen 1: …`)
- [ ] „Stell in den Einstellungen den dunklen Modus ein“ – wie viele Züge, wie lange?
- [ ] „Sag mir Bescheid, wenn ‚Fertig‘ auf dem Bildschirm steht“ – dann irgendwo „Fertig“
      anzeigen lassen; Cassi meldet sich innerhalb von ~3 s
- [ ] `uv run cassi eval-computer` (siehe README) – Erfolgsquote und Züge notieren

## 5. Kalender (klassisches Outlook)

- [ ] „Was steht heute an?“ – Outlook muss installiert sein (das neue Outlook hat keine
      COM-Schnittstelle: dann sagt Cassi, es kann den Kalender nicht lesen)
- [ ] „Hab ich morgen Termine?“, „Wann ist mein nächster Termin?“

## 6. Datenschutz

- [ ] KeePass oder 1Password öffnen, „Was siehst du auf meinem Bildschirm?“ → Cassi lehnt ab
- [ ] InPrivate-Fenster in Edge öffnen → ebenso
- [ ] „Schau nicht mehr auf meinen Bildschirm“ → Screenshot wird abgelehnt; „Du darfst wieder
      auf den Bildschirm schauen“
- [ ] Tray-Menü: „Verstanden: …“, „Geantwortet: …“, Schalter „Bildschirm nicht ansehen“
- [ ] `uv run cassi history`, `uv run cassi stats`

## 7. Sprache und Stimme

- [ ] „Sprich langsamer“ (Piper: sofort), „Andere Stimme“ (wechselt zwischen installierten
      Piper-Stimmen), „Antworte kürzer“ – nach `cassi run` neu starten: bleibt so

## Rückmeldung

Für jeden ✗: was gesagt wurde, was passiert ist, der Abschnitt aus dem Log. `cassi history
--failed` listet die fehlgeschlagenen Anfragen.
