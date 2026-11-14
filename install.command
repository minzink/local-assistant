#!/usr/bin/env bash
# Cassi installieren oder reparieren (macOS). Im Finder doppelklicken oder im Terminal starten:
#   ./install.command [--no-install] [--skip-models]
# Ohne Checkout (lädt Cassi nach ~/cassi bzw. $CASSI_HOME):
#   curl -fsSL https://raw.githubusercontent.com/minzink/local-assistant/main/install.command | bash
# Danach öffnet sich die Verwaltung (Reparieren, Updaten, Deinstallieren): cassi manage
set -euo pipefail

REPO="https://github.com/minzink/local-assistant.git"
here="${BASH_SOURCE[0]:-}"
if [ -n "$here" ] && [ -f "$(dirname "$here")/scripts/setup.sh" ]; then
    ROOT="$(cd "$(dirname "$here")" && pwd)"
else
    ROOT="${CASSI_HOME:-$HOME/cassi}"
    if [ ! -d "$ROOT/.git" ]; then
        # macOS offers to install the developer tools (with git) on first use.
        command -v git >/dev/null 2>&1 || { echo "Git fehlt: xcode-select --install"; exit 1; }
        echo "Lade Cassi nach $ROOT ..."
        git clone "$REPO" "$ROOT"
    fi
fi

if ! bash "$ROOT/scripts/setup.sh" "$@"; then
    echo
    echo "Die Installation ist nicht vollständig – das Log steht oben. Später erneut starten."
    exit 1
fi
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
echo
echo "Öffne die Verwaltung ..."
cd "$ROOT" && exec uv run --quiet cassi manage
