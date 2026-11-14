#!/usr/bin/env bash
# Cassi unter macOS einrichten: Programme (uv, Git LFS, Ollama), Python-Pakete, Modelle, Befehl.
# Kann beliebig oft ausgeführt werden (= Reparieren); was schon da ist, wird übersprungen.
#   scripts/setup.sh [--no-install] [--skip-models] [--project DIR --command NAME]
# Jeder Lauf schreibt ein ausführliches Log nach logs/setup-<Zeitstempel>.log.
# --project/--command richten ein Paket ein, das Cassi einbindet (z.B. Flay: ../flay, flay).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup-$(date +%Y%m%d-%H%M%S).log"
OLLAMA_URL="http://127.0.0.1:11434"
NO_INSTALL=0
SKIP_MODELS=0
PROJECT="$ROOT"  # the uv project to install (Cassi itself, or a package built on it)
COMMAND=cassi    # its command
while [ $# -gt 0 ]; do
    case "$1" in
        --no-install) NO_INSTALL=1 ;;   # fehlende Programme nicht per Homebrew installieren
        --skip-models) SKIP_MODELS=1 ;; # 'cassi setup' (Modelle laden) auslassen
        --project) PROJECT="$(cd "${2:?--project braucht einen Ordner}" && pwd)"; shift ;;
        --command) COMMAND="${2:?--command braucht einen Namen}"; shift ;;
        -h | --help) sed -n '2,6p' "$0"; exit 0 ;;
        *) echo "Unbekannte Option: $1" >&2; exit 2 ;;
    esac
    shift
done
export NO_COLOR=1 UV_NO_PROGRESS=1 PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# Everything (also the output of uv, brew, cassi) goes to the screen and the log file.
exec > >(tee -a "$LOG_FILE") 2>&1

ts() { date +%H:%M:%S; }
step() { printf '\n%s ==> %s\n' "$(ts)" "$*"; }
ok() { printf '%s  OK  %s\n' "$(ts)" "$*"; }
info() { printf '%s      %s\n' "$(ts)" "$*"; }
warn() { printf '%s  !!  %s\n' "$(ts)" "$*"; }
fail() {
    printf '%s  XX  %s\n' "$(ts)" "$*"
    printf '\nSetup fehlgeschlagen. Vollständiges Log: %s\n' "$LOG_FILE"
    exit 1
}
run() {
    info "> $*"
    "$@"
    local code=$?
    [ "$code" -eq 0 ] || warn "Exit-Code $code: $*"
    return "$code"
}
# HTTP GET without any proxy, like Cassi for local servers.
get_direct() { curl --silent --fail --noproxy '*' --max-time "${2:-3}" "$1"; }

diagnostics() {
    step "Systeminformationen (für Fehlersuche)"
    info "macOS $(sw_vers -productVersion 2>/dev/null) ($(uname -m)), $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    info "RAM: $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )) GB"
    info "Cassi: $ROOT ($(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo 'kein Git'))"
    [ "$PROJECT" = "$ROOT" ] || info "Projekt: $PROJECT ($COMMAND, $(git -C "$PROJECT" rev-parse --short HEAD 2>/dev/null || echo 'kein Git'))"
    local free_gb
    free_gb=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
    info "Freier Speicher: ${free_gb} GB"
    [ "${free_gb:-0}" -ge 25 ] || warn "Weniger als 25 GB frei – Modelle brauchen ~15–20 GB"
    [ "$(uname -m)" = "arm64" ] || warn "Kein Apple Silicon – Cassi ist für M-Macs gedacht (MLX)"
    for name in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy no_proxy OLLAMA_HOST; do
        [ -n "${!name:-}" ] && info "Umgebung: $name=${!name}"
    done
    if scutil --proxy 2>/dev/null | grep -qE '(HTTP|HTTPS)Enable : 1'; then
        warn "System-Proxy aktiv – Cassi umgeht ihn für lokale Server:"
        scutil --proxy | grep -E 'Proxy|Port|Exceptions' | sed 's/^/        /'
    fi
}

ensure() { # ensure <command> <brew formula> <name>
    local command=$1 package=$2 name=$3
    if command -v "$command" >/dev/null 2>&1; then
        ok "$name gefunden: $(command -v "$command")"
        return 0
    fi
    [ "$NO_INSTALL" -eq 0 ] || fail "$name fehlt (brew install $package)"
    command -v brew >/dev/null 2>&1 || fail "$name fehlt und Homebrew ist nicht installiert – siehe https://brew.sh"
    info "$name wird installiert (brew install $package) ..."
    run brew install "$package" || fail "$name konnte nicht installiert werden"
    command -v "$command" >/dev/null 2>&1 || fail "$name installiert, aber '$command' nicht im PATH"
    ok "$name installiert"
}

programs() {
    step "Programme prüfen"
    ensure git git Git
    ensure git-lfs git-lfs "Git LFS"
    ensure uv uv uv
    ensure ollama ollama Ollama
    local version
    version=$(ollama --version 2>&1 | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | tail -1)
    info "Ollama: ${version:-unbekannt}"
    if [ -n "$version" ] && [ "$(printf '%s\n0.30\n' "$version" | sort -V | head -1)" != "0.30" ]; then
        warn "Ollama $version ist zu alt (mind. 0.30): brew upgrade ollama bzw. Ollama-App aktualisieren"
    fi
}

lfs() {
    step "Mitgelieferte Stimmen (Git LFS)"
    if [ ! -d "$ROOT/.git" ]; then
        warn "Kein Git-Checkout – Git LFS übersprungen"
        return
    fi
    run git -C "$ROOT" lfs install
    run git -C "$ROOT" lfs pull || warn "git lfs pull fehlgeschlagen"
    local voice="$ROOT/src/cassi/tts/voices/ramona.wav"
    if [ -f "$voice" ] && [ "$(wc -c <"$voice")" -lt 2048 ]; then
        warn "$voice ist nur ein LFS-Zeiger – Stimme fehlt"
    else
        ok "Stimmen vorhanden"
    fi
}

ollama_server() {
    step "Ollama-Server"
    local body
    if body=$(get_direct "$OLLAMA_URL/api/version"); then
        ok "Ollama läuft auf $OLLAMA_URL: $body"
        return
    fi
    if lsof -nP -iTCP:11434 -sTCP:LISTEN >/dev/null 2>&1; then
        warn "Port 11434 ist belegt, antwortet aber nicht:"
        lsof -nP -iTCP:11434 -sTCP:LISTEN | sed 's/^/        /'
    fi
    info "Ollama antwortet nicht – wird gestartet ..."
    if [ -d /Applications/Ollama.app ] || [ -d "$HOME/Applications/Ollama.app" ]; then
        open -a Ollama
    elif brew list ollama >/dev/null 2>&1; then
        run brew services start ollama
    else
        nohup ollama serve >"$LOG_DIR/ollama-serve.log" 2>&1 &
        info "ollama serve läuft im Hintergrund, Log: $LOG_DIR/ollama-serve.log"
    fi
    for _ in $(seq 1 60); do
        sleep 1
        if body=$(get_direct "$OLLAMA_URL/api/version" 1); then
            ok "Ollama läuft: $body"
            return
        fi
    done
    fail "Ollama startet nicht ($OLLAMA_URL). Logs: ~/.ollama/logs/server.log"
}

python_packages() {
    step "Python-Pakete (uv sync)"
    cd "$PROJECT" || fail "$PROJECT nicht gefunden"
    run uv sync || fail "uv sync fehlgeschlagen"
    run uv run python --version
    run uv run python -c "import urllib.request; print('Python-Proxies:', urllib.request.getproxies() or 'keine')"
    # The same connection Cassi makes at startup (address from cassi.yaml), with diagnosis.
    run uv run python -c "from cassi.config import load_settings; from cassi import net; u = load_settings(None).llm.base_url; p = net.unreachable(u.rstrip('/') + '/models'); print('LLM-Server aus Python erreichbar:', u) if p is None else print(net.llm_unreachable_message(u, p)); raise SystemExit(p is not None)" \
        || fail "Cassi erreicht den LLM-Server nicht (Diagnose oben)"
    ok "Python-Pakete installiert"
}

models() {
    step "Modelle laden ($COMMAND setup: Spracherkennung, Stimmen, LLM über Ollama)"
    cd "$PROJECT" || fail "$PROJECT nicht gefunden"
    run uv run "$COMMAND" -v setup || fail "$COMMAND setup fehlgeschlagen (Details oben)"
    run uv run "$COMMAND" info >>"$LOG_FILE" 2>&1 # only into the log
    ok "Modelle vollständig"
}

launcher() {
    step "Befehl $COMMAND einrichten"
    cd "$PROJECT" || fail "$PROJECT nicht gefunden"
    run uv run "$COMMAND" install-command || warn "Befehl nicht eingerichtet – starten mit: uv run $COMMAND run"
}

step "Cassi-Setup gestartet, Log: $LOG_FILE"
diagnostics
programs
lfs
ollama_server
python_packages
if [ "$SKIP_MODELS" -eq 1 ]; then warn "Modelle übersprungen (--skip-models)"; else models; fi
launcher

printf '\nFertig. Starten mit:  %s run   (Verwaltung: %s manage)\n' "$COMMAND" "$COMMAND"
printf 'Einmalig in Systemeinstellungen → Datenschutz & Sicherheit für das Terminal erlauben:\n'
printf '  Mikrofon, Bedienungshilfen, Eingabeüberwachung (und Bildschirmaufnahme für Screenshots).\n'
printf 'Log: %s\n' "$LOG_FILE"
