"""Verbs as the user's speech recognition hears them: "She K Command" for "Schließe KX Command".

When a short request like that was carried out (and not contradicted), its verb is learned
for that action. From then on "She <any project>" is a close request for every project –
Cassi does not need an example per project.

German verbs are often split: "mach KX Forms zu", "führ KX Forms aus". The first word and
a particle at the very end belong together and are learned as one verb ("mach … zu");
"mach … zu" and "mach … auf" are then different verbs. Without a particle the first word
alone is the verb ("she").

To stay safe, a learned verb only counts in a short request whose words between verb and
particle name a project; function words, known verbs and parts of the name are no verbs.
"""

from __future__ import annotations

import re

from cassi.tools.phpstorm import match_names, mention

# Actions with a project name whose verb can be learned.
VERB_TOOLS = ("close_project", "open_project")
# What a learned verb stands for, written out for the model.
CANONICAL = {"close_project": "Schließe", "open_project": "Öffne"}
# Longer requests are sentences, not "verb + name".
MAX_WORDS = 7
# Particles that complete a split verb at the end of the request.
PARTICLES = frozenset("zu auf aus an ab ein weg up down off".split())
# Verbs that mean nothing without their particle ("mach … zu" / "mach … auf").
LIGHT_VERBS = frozenset("mach mache macht machen make tu tue tun".split())
# Words that start requests without being their verb ("Und …", "Ich hab …").
FUNCTION_WORDS = frozenset(
    "und and ich du er sie es wir ihr i you he it we they "
    "das der die den dem the a an ein eine einen bitte please mal jetzt nun noch auch doch "
    "kannst kann könntest could can would würdest sag sagt tell schau look was what wie how "
    "ist is hast hab habe have nein no ja yes okay ok danke thanks projekt project".split()
)
# Known verbs: nothing to learn about them.
_KNOWN = re.compile(
    r"^(?:schließ\w*|schliess\w*|öffne\w*|oeffne\w*|close\w*|open\w*|start\w*|beend\w*)$"
)


def _words(text: str) -> list[str]:
    return re.findall(r"[\wäöüß]+", text.lower())


def candidates(request: str) -> list[tuple[str, str]]:
    """Possible verbs of a short request with the words they leave for the project name.

    "Mach K NVS zu" → [("mach … zu", "k nvs"), ("mach", "k nvs zu")]: the split verb
    first; the first word alone as fallback. Longer requests have none.
    """
    words = _words(request)
    if not 2 <= len(words) <= MAX_WORDS:
        return []
    head, rest = words[0], words[1:]
    found = []
    if len(rest) >= 2 and rest[-1] in PARTICLES:
        found.append((f"{head} … {rest[-1]}", " ".join(rest[:-1])))
    found.append((head, " ".join(rest)))
    return found


def learned_verb(request: str, tool: str, name: str) -> str | None:
    """The verb this request used for ``tool`` – if it is one worth learning."""
    if tool not in VERB_TOOLS or not name:
        return None
    for verb, rest in candidates(request):
        head = verb.split()[0]
        if len(head) < 2 or head in FUNCTION_WORDS or _KNOWN.match(head):
            return None
        if head in LIGHT_VERBS and " … " not in verb:
            continue  # "mach" alone could mean open as well as close
        if mention(name, head) is not None or mention(name, rest) is None:
            continue  # part of the name, or the rest does not name the project
        return verb
    return None


# Recognizing a garbled verb by its sound -------------------------------------------------
#
# Before anything was learned, "Slie K X E V O" (Schließe kx-ewo) and "She K Command" have
# no verb Cassi knows. Their first word still *sounds* like the beginning of "schließe": the
# Kölner Phonetik code of "slie" (85) and "she" (8) starts the code of "schließe" (858).
# This only counts in a short request whose remaining words are just a project name.

_CODES = (
    ("aeijouyäöü", "0"), ("b", "1"), ("fvw", "3"), ("gkq", "4"), ("l", "5"), ("mn", "6"),
    ("r", "7"), ("szß", "8"),
)  # fmt: skip
# Real verbs that start like "schließe" or "öffne" but mean something else.
_OTHER_VERBS = frozenset(
    "sag sage sags such suche schau schaue sieh siehe sende send schick schicke stell stelle "
    "setz setze start starte schreib schreibe spiel spiele speicher speichere sperr sperre "
    "show see say set stop stopp zeig zeige les lies lass öffnet offen oft ob of off".split()
)
# A garbled verb this short is still informative, but not the bare first letter of a word
# like "so" or "se".
_MIN_LETTERS = 3


def phonetic_code(word: str) -> str:
    """Kölner Phonetik of a word (enough of it for verbs; no special cases for names)."""
    word = re.sub(r"[^a-zäöüß]", "", word.lower())
    digits: list[str] = []
    for index, letter in enumerate(word):
        before = word[index - 1] if index else ""
        after = word[index + 1] if index + 1 < len(word) else ""
        if letter == "h":
            code = "-"
        elif letter == "p":
            code = "3" if after == "h" else "1"
        elif letter in "dt":
            code = "8" if after in "csz" else "2"
        elif letter == "c":
            if index == 0:
                code = "4" if after in "ahkloqrux" else "8"
            else:
                code = "8" if before in "sz" or after not in "ahkoqux" else "4"
        elif letter == "x":
            code = "8" if before in "ckq" else "48"
        else:
            code = next((c for letters, c in _CODES if letter in letters), "")
        digits.append(code)
    collapsed: list[str] = []
    for code in digits:
        if code != "-" and (not collapsed or collapsed[-1] != code):
            collapsed.append(code)
    first, rest = (collapsed[0], collapsed[1:]) if collapsed else ("", [])
    return first + "".join(c for c in rest if c != "0")


_CLOSE_SOUNDS = tuple(phonetic_code(v) for v in ("schließe", "close"))
_OPEN_SOUNDS = tuple(phonetic_code(v) for v in ("öffne", "open"))
_FILLERS = frozenset("mal bitte auch noch das den die projekt project doch jetzt".split())


def heard_verb(request: str, projects: list[str]) -> tuple[str, str] | None:
    """open_project / close_project for a short request whose verb is garbled beyond
    recognition but sounds like "schließe"/"öffne" ("Slie K X E V O"), or is the split verb
    "mach … auf/zu" ("Mach mal Moewe auf") – only if the other words name a project.

    Returns the tool and the request written out for the model ("Schließe k x e v o")."""
    words = _words(request)
    if not 2 <= len(words) <= MAX_WORDS:
        return None
    head, rest = words[0], words[1:]
    tool = None
    if head in LIGHT_VERBS and rest[-1] in ("auf", "zu"):
        tool = "open_project" if rest[-1] == "auf" else "close_project"
        rest = rest[:-1]
    elif head not in _OTHER_VERBS and head not in FUNCTION_WORDS and not _KNOWN.match(head):
        code = phonetic_code(head)
        if len(head) >= _MIN_LETTERS and code:
            if any(sound.startswith(code) for sound in _CLOSE_SOUNDS):
                tool = "close_project"
            elif any(sound.startswith(code) for sound in _OPEN_SOUNDS):
                tool = "open_project"
    name = " ".join(w for w in rest if w not in _FILLERS)
    if tool is None or not name:
        return None
    # The rest must be just the name, matched as a project name is matched elsewhere.
    if not match_names(name, projects):
        return None
    return tool, f"{CANONICAL[tool]} {name}"
