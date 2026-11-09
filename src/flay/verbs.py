"""Verbs as the user's speech recognition hears them: "She K Command" for "Schließe KX Command".

When a short request like that was carried out (and not contradicted), its verb is learned
for that action. From then on "She <any project>" is a close request for every project –
Flay does not need an example per project.

German verbs are often split: "mach KX Forms zu", "führ KX Forms aus". The first word and
a particle at the very end belong together and are learned as one verb ("mach … zu");
"mach … zu" and "mach … auf" are then different verbs. Without a particle the first word
alone is the verb ("she").

To stay safe, a learned verb only counts in a short request whose words between verb and
particle name a project; function words, known verbs and parts of the name are no verbs.
"""

from __future__ import annotations

import re

from flay.tools.phpstorm import mention

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
