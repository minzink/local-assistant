#!/usr/bin/env bash
# Flay unter macOS einrichten: Programme (uv, Git LFS, Ollama), Python-Pakete, Modelle.
# Kann beliebig oft ausgeführt werden; was schon da ist, wird übersprungen.
#   scripts/setup.sh [--no-install] [--skip-models]
# Jeder Lauf schreibt ein ausführliches Log nach logs/setup-<Zeitstempel>.log.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup-$(date +%Y%m%d-%H%M%S).log"
OLLAMA_URL="http://127.0.0.1:11434"
NO_INSTALL=0
SKIP_MODELS=0
for arg in "$@"; do
    case "$arg" in
        --no-install) NO_INSTALL=1 ;;   # fehlende Programme nicht per Homebrew installieren
        --skip-models) SKIP_MODELS=1 ;; # 'flay setup' (Modelle laden) auslassen
        -h | --help) sed -n '2,5p' "$0"; exit 0 ;;
        *) echo "Unbekannte Option: $arg" >&2; exit 2 ;;
    esac
done
export NO_COLOR=1 UV_NO_PROGRESS=1 PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# Everything (also the output of uv, brew, flay) goes to the screen and the log file.
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
# HTTP GET without any proxy, like Flay for local servers.
get_direct() { curl --silent --fail --noproxy '*' --max-time "${2:-3}" "$1"; }

diagnostics() {
    step "Systeminformationen (für Fehlersuche)"
    info "macOS $(sw_vers -productVersion 2>/dev/null) ($(uname -m)), $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
    info "RAM: $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )) GB"
    info "Projekt: $ROOT ($(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo 'kein Git'))"
    local free_gb
    free_gb=$(df -g "$ROOT" | awk 'NR==2 {print $4}')
    info "Freier Speicher: ${free_gb} GB"
    [ "${free_gb:-0}" -ge 25 ] || warn "Weniger als 25 GB frei – Modelle brauchen ~15–20 GB"
    [ "$(uname -m)" = "arm64" ] || warn "Kein Apple Silicon – Flay ist für M-Macs gedacht (MLX)"
    for name in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy no_proxy OLLAMA_HOST; do
        [ -n "${!name:-}" ] && info "Umgebung: $name=${!name}"
    done
    if scutil --proxy 2>/dev/null | grep -qE '(HTTP|HTTPS)Enable : 1'; then
        warn "System-Proxy aktiv – Flay umgeht ihn für lokale Server:"
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
    local voice="$ROOT/src/flay/tts/voices/ramona.wav"
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
    cd "$ROOT" || fail "$ROOT nicht gefunden"
    run uv sync || fail "uv sync fehlgeschlagen"
    run uv run python --version
    run uv run python -c "import urllib.request; print('Python-Proxies:', urllib.request.getproxies() or 'keine')"
    # The same connection Flay makes at startup (address from flay.yaml), with diagnosis.
    run uv run python -c "from flay.config import load_settings; from flay import net; u = load_settings(None).llm.base_url; p = net.unreachable(u.rstrip('/') + '/models'); print('LLM-Server aus Python erreichbar:', u) if p is None else print(net.llm_unreachable_message(u, p)); raise SystemExit(p is not None)" \
        || fail "Flay erreicht den LLM-Server nicht (Diagnose oben)"
    ok "Python-Pakete installiert"
}

models() {
    step "Modelle laden (flay setup: Spracherkennung, Stimmen, LLM über Ollama)"
    cd "$ROOT" || fail "$ROOT nicht gefunden"
    run uv run flay -v setup || fail "flay setup fehlgeschlagen (Details oben)"
    run uv run flay info >>"$LOG_FILE" 2>&1 # only into the log
    ok "Modelle vollständig"
}

step "Flay-Setup gestartet, Log: $LOG_FILE"
diagnostics
programs
lfs
ollama_server
python_packages
if [ "$SKIP_MODELS" -eq 1 ]; then warn "Modelle übersprungen (--skip-models)"; else models; fi

printf '\nFertig. Starten mit:  uv run flay run\n'
printf 'Einmalig in Systemeinstellungen → Datenschutz & Sicherheit für das Terminal erlauben:\n'
printf '  Mikrofon, Bedienungshilfen, Eingabeüberwachung (und Bildschirmaufnahme für Screenshots).\n'
printf 'Log: %s\n' "$LOG_FILE"
