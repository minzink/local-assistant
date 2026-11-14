"""Stop private data before it reaches a commit (git hooks in .githooks/, also used by Flay).

The repos must never contain names of real projects, customers or companies, e-mail
addresses, server names, user paths or internal IP addresses (see .ai-instruct). This
script checks what is about to be committed against terms that it collects on this computer
at run time – nothing private is stored in the repo:

- the names of the folders listed in the terms file (``dir: ~/phpstormprojects``),
- git's user name and e-mail address, the login name,
- host names and addresses from ``~/.ssh/config``,
- every other line of the terms file: ``~/.config/private-terms.txt`` or ``$PRIVATE_TERMS``.

Plus patterns that are private anywhere: e-mail addresses (except GitHub's noreply and
example.org/com) and home folders (/Users/<name>, C:\\Users\\<name>). Internal addresses
come from ~/.ssh/config or the terms file (a network prefix like ``10.20`` matches all of it);
example addresses in tests (10.0.0.1) are fine. Lock files are skipped: they only hold
package URLs and versions, which look like addresses.

Usage:
    check_private.py staged          # added lines and file names of the staged change
    check_private.py message FILE    # a commit message
    check_private.py tree            # every tracked file as it is now
    check_private.py history         # every commit: added lines, file names, messages
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

TERMS_FILE = Path(os.environ.get("PRIVATE_TERMS", Path.home() / ".config" / "private-terms.txt"))

# Public on purpose: the user's first name and city, the GitHub account, the invented project
# names from .ai-instruct and the names of these repos.
ALLOWED = {
    "timo", "bielefeld", "github.com", "localhost", "minzink",
    "ai", "cassi", "flay", "local-assistant", "phoenix", "moewe", "umx", "stadt-wolkenthal",
    "project-symfony-kx-ewo", "component-symfony-kx-command", "project-symfony-kx-forms",
    "lib-react-kx-forms-app",
}  # fmt: skip
# Folder names that are ordinary words, and parts of folder names too common to flag alone.
COMMON = {
    "cli", "test", "tests", "lando", "github", "scripts", "docker", "agent", "sandbox",
    "typo3", "group", "shop", "portal", "digital", "stadt", "netz", "netze", "cms", "data",
    "dental", "union", "baeder", "biogas", "dekor", "1password",
}  # fmt: skip
SKIPPED = {"uv.lock"}
PATTERNS = {
    "E-Mail-Adresse": re.compile(
        r"[\w.%+-]+@(?!users\.noreply\.github\.com|anthropic\.com|example\.(?:com|org)\b)"
        r"[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}",
        re.IGNORECASE,
    ),
    "Benutzerordner": re.compile(
        r"(?:/Users/|/home/|[A-Z]:\\+Users\\+)(?!(?:x|name|du|you|me|user|runner|runneradmin)\b)"
        r"[A-Za-z][\w.-]*",
    ),
}


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    ).stdout


def _folder_terms(folder: Path) -> Iterator[str]:
    for path in folder.iterdir() if folder.is_dir() else []:
        if not path.is_dir() or path.name.startswith("."):
            continue
        name = path.name.lower()
        yield name
        yield from (part for part in re.split(r"[-_. ]", name) if len(part) >= 5)


def collect_terms() -> set[str]:
    terms: set[str] = set()
    if TERMS_FILE.is_file():
        for line in TERMS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("dir:"):
                terms.update(_folder_terms(Path(line[4:].strip()).expanduser()))
            else:
                terms.add(line.lower())
    for key in ("user.name", "user.email"):
        for value in git("config", "--get-all", key).splitlines():
            if "noreply" in value:
                continue
            terms.update(part.lower() for part in re.split(r"[\s@]", value) if part)
            if "@" in value:  # the company domain without its ending, too
                terms.add(value.split("@")[1].rsplit(".", 1)[0].lower())
    for name in (os.environ.get("USER"), os.environ.get("USERNAME"), Path.home().name):
        if name:
            terms.add(name.lower())
    ssh = Path.home() / ".ssh" / "config"
    if ssh.is_file():
        for line in ssh.read_text(encoding="utf-8", errors="replace").splitlines():
            words = line.split()
            if len(words) >= 2 and words[0].lower() in ("host", "hostname"):
                terms.update(w.lower() for w in words[1:] if not any(c in w for c in "*?!"))
    return {t for t in terms if len(t) >= 3 and t not in ALLOWED and t not in COMMON}


def finder(terms: Iterable[str]) -> re.Pattern[str] | None:
    ordered = sorted(terms, key=len, reverse=True)
    if not ordered:
        return None
    alternatives = "|".join(re.escape(t) for t in ordered)
    return re.compile(rf"(?<![\w-])(?:{alternatives})(?![\w])", re.IGNORECASE)


def problems(text: str, terms: re.Pattern[str] | None) -> list[str]:
    found = []
    if terms is not None:
        found += [f"„{m.group(0)}“" for m in terms.finditer(text)]
    for label, pattern in PATTERNS.items():
        found += [f"{label} „{m.group(0)}“" for m in pattern.finditer(text)]
    return found


def added_lines(diff: str) -> Iterator[tuple[str, str]]:
    """(where, text) for every added line, file name and commit message line of a diff or of
    ``git log -p`` output."""
    current, commit, in_message = "", "", False
    for line in diff.splitlines():
        if line.startswith("commit ") and re.fullmatch(r"commit [0-9a-f]{40}", line):
            commit, in_message = line[7:14], False
        elif line.startswith(("Author:", "Date:", "Merge:")) and commit:
            in_message = True
        elif line.startswith("diff --git"):
            in_message = False
        elif in_message and line.startswith("    "):
            yield f"Nachricht {commit}", line
        elif line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else line[4:]
            yield f"{commit} {current}".strip(), current
        elif line.startswith("+"):
            yield f"{commit} {current}".strip(), line[1:]


def report(hits: list[tuple[str, str]]) -> int:
    if not hits:
        return 0
    print("Private Daten gefunden – so nicht committen:", file=sys.stderr)
    for where, what in hits[:50]:
        print(f"  {where}: {what}", file=sys.stderr)
    if len(hits) > 50:
        print(f"  … und {len(hits) - 50} weitere", file=sys.stderr)
    print(
        "Ersetzen (erfundene Namen, example.org), nicht mit --no-verify umgehen. "
        f"Begriffe: git, ~/.ssh/config und {TERMS_FILE}.",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("staged", "message", "tree", "history"):
        print(__doc__)
        return 2
    terms = finder(collect_terms())
    hits: list[tuple[str, str]] = []
    mode = argv[0]
    if mode == "message":
        text = Path(argv[1]).read_text(encoding="utf-8", errors="replace")
        lines = [line for line in text.splitlines() if not line.startswith("#")]
        hits = [("Commit-Nachricht", p) for line in lines for p in problems(line, terms)]
    elif mode == "tree":
        for name in git("ls-files", "-z").split("\0"):
            path = Path(name)
            if not name or path.name in SKIPPED:
                continue
            hits += [(name, p) for p in problems(name, terms)]
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue  # binary: models, audio
            for number, line in enumerate(text.splitlines(), 1):
                hits += [(f"{name}:{number}", p) for p in problems(line, terms)]
    else:
        if mode == "staged":
            diff = git("diff", "--cached", "-U0", "--no-color", "--no-ext-diff", "--text")
        else:
            diff = git("log", "--all", "-p", "-U0", "--no-color", "--no-ext-diff", "--text")
        seen: set[tuple[str, str]] = set()
        for where, line in added_lines(diff):
            if Path(where.split(" ")[-1]).name in SKIPPED:
                continue
            for problem in problems(line, terms):
                if (where, problem) not in seen:
                    seen.add((where, problem))
                    hits.append((where, problem))
    return report(hits)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
