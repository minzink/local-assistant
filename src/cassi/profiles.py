"""Profiles: separate AIs on one Cassi – each with its own definition, what it learned and
what it remembers.

A profile is a folder that serves as Cassi's data folder for that AI – memory
(memory.sqlite3), history, learned notes, reflexes, key settings, own voice and wake word
models – plus ``profile.yaml``, its definition::

    title: Schach-Flay
    description: Flay, nur zum Schachspielen
    adapters: [flay]          # what it builds on; [] = plain Cassi
    character:                # who it is – made into persona.system_prompt (Cassi-based)
      identity: A calm chess coach.
      speaking_style: Short and patient.
    settings:                 # like cassi.yaml, only what differs for this AI
      persona:
        name: Kasimir
        system_prompt: You are Kasimir, …

All of it can be set in the management UI (tab "Profile"): ``define`` writes it – for an AI
built on an adapter (Flay) into that adapter's own files (``Adapter.define_profile``).

Shared by all profiles: cassi.yaml (devices, server, …), downloaded models and the export
folder – they belong to the computer and the user, not to one AI.

Built in: "cassi" (Cassi's data folder itself) and one for each adapter that is an AI of its
own (``Adapter.own_profile``, e.g. "flay", in the folder it always had). Own profiles live in
``<data folder>/profiles/<name>``.

Which one is used: ``--profil`` (or CASSI_PROFILE), else the one chosen last (``cassi profil
wechseln``, the management UI), else ``profile`` in cassi.yaml, else the adapter's own when
started by its command (``flay``), else the first adapter's own, else "cassi".
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from cassi.config import Settings, default_data_dir

log = logging.getLogger(__name__)

BASE = "cassi"
FOLDER = "profiles"
DEFINITION = "profile.yaml"
ENVIRONMENT = "CASSI_PROFILE"
# Never copied into a new profile: other profiles, this computer's records, old exports.
NOT_COPIED = (FOLDER, "run", "backups", "exports", "voices")

Copy = Literal["nothing", "definition", "everything"]


class ProfileError(ValueError):
    """A profile cannot be used, made or removed; the message says why (German, for the user)."""


@dataclass(frozen=True)
class Profile:
    name: str  # short, for commands and folders: "cassi", "flay", "schach-flay"
    folder: Path
    title: str = ""
    description: str = ""
    # Adapters it builds on (only those that are AIs of their own; others apply to all).
    adapters: tuple[str, ...] = ()
    # Settings that differ for this AI, like cassi.yaml.
    settings: Mapping[str, Any] = field(default_factory=dict)
    # Built in ("cassi", an adapter's own): cannot be removed.
    builtin: bool = False
    # Who it is (identity, speaking_style) – kept to make the system prompt anew on a change.
    character: Mapping[str, str] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.title or self.name


@dataclass(frozen=True)
class Definition:
    """What defines an AI, as set in the management UI; empty / None: stays as it is."""

    name: str = ""  # what the AI is called ("Kasimir")
    identity: str = ""  # who it is, its personality (free text, English works best)
    speaking_style: str = ""  # how it talks
    voice: str = ""  # Cassi's voice: "auto", "ramona", "thorsten", "own", …
    pitch: float | None = None  # semitones
    speed: float | None = None  # 1.1 = 10 % faster
    model: str = ""  # the language model (Ollama name)

    def __bool__(self) -> bool:
        return any((self.name, self.identity, self.speaking_style, self.voice, self.model)) or (
            self.pitch,
            self.speed,
        ) != (None, None)


# An adapter's ``define_profile(folder, definition) -> settings``: writes into its files in
# the profile's folder what it takes from ``definition``; returns settings for profile.yaml.
DefineHook = Callable[[Path, Definition], Mapping[str, Any]]


@dataclass(frozen=True)
class AdapterInfo:
    """What profiles need to know of an installed adapter (its class, not loaded yet)."""

    name: str
    own_profile: bool = False
    title: str = ""
    # Files in its data folder that define the AI (copied for a new profile from it), as
    # opposed to what it learned or remembers.
    definition_files: tuple[str, ...] = ()
    define: DefineHook | None = None


def adapter_infos() -> dict[str, AdapterInfo]:
    from cassi.adapters import installed

    infos: dict[str, AdapterInfo] = {}
    for name, load in installed().items():
        try:
            factory = load()
        except Exception as exc:  # a broken adapter is reported when it is used
            log.debug("Adapter %s nicht geladen: %s", name, exc)
            infos[name] = AdapterInfo(name)
            continue
        infos[name] = AdapterInfo(
            name,
            bool(getattr(factory, "own_profile", False)),
            str(getattr(factory, "profile_title", "") or name.capitalize()),
            tuple(getattr(factory, "definition_files", ())),
            getattr(factory, "define_profile", None),
        )
    return infos


def slug(title: str) -> str:
    """A profile's short name from its title: "Schach-Flay (Test)" → "schach-flay-test"."""
    text = title.strip().lower()
    for umlaut, plain in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(umlaut, plain)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")[:40]


def available(base: Path, infos: Mapping[str, AdapterInfo] | None = None) -> list[Profile]:
    """All profiles: the built-in ones first, then the own ones by name."""
    infos = adapter_infos() if infos is None else infos
    found = [
        Profile(BASE, base, "Cassi", "Cassi ohne Adapter, im eigenen Datenordner", builtin=True)
    ]
    for info in sorted(infos.values(), key=lambda i: i.name):
        if info.own_profile:
            found.append(
                Profile(
                    info.name,
                    default_data_dir(info.name),
                    info.title,
                    f"{info.title} – im eigenen Datenordner",
                    (info.name,),
                    builtin=True,
                )
            )
    taken = {p.name for p in found}
    folder = base / FOLDER
    for path in sorted(folder.iterdir()) if folder.is_dir() else []:
        if path.name in taken or not (path / DEFINITION).is_file():
            continue
        try:
            found.append(_read(path))
        except ProfileError as exc:
            log.warning("%s", exc)
    return found


def _read(folder: Path) -> Profile:
    try:
        data = yaml.safe_load((folder / DEFINITION).read_text("utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileError(f"Profil {folder.name}: {DEFINITION} unlesbar ({exc})") from exc
    if not isinstance(data, dict):
        raise ProfileError(f"Profil {folder.name}: {DEFINITION} ist kein Mapping")
    adapters = data.get("adapters") or []
    settings = data.get("settings") or {}
    character = data.get("character") or {}
    shaped = isinstance(adapters, list) and isinstance(settings, dict)
    if not (shaped and isinstance(character, dict)):
        raise ProfileError(
            f"Profil {folder.name}: 'adapters', 'settings' oder 'character' falsch geformt"
        )
    return Profile(
        folder.name,
        folder,
        str(data.get("title") or folder.name),
        str(data.get("description") or ""),
        tuple(str(a) for a in adapters),
        settings,
        character={str(k): str(v) for k, v in character.items() if v},
    )


def find(base: Path, name: str, infos: Mapping[str, AdapterInfo] | None = None) -> Profile:
    profiles = available(base, infos)
    wanted = name.strip().lower()
    for profile in profiles:
        if profile.name in (wanted, slug(name)) or profile.title.lower() == wanted:
            return profile
    names = ", ".join(p.name for p in profiles)
    raise ProfileError(f"Profil „{name}“ gibt es nicht (vorhanden: {names})")


def _choice_file(base: Path) -> Path:
    return base / "run" / "profile.json"


def chosen(base: Path) -> str | None:
    """The profile chosen last (management UI, ``cassi profil wechseln``)."""
    try:
        data = json.loads(_choice_file(base).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    name = data.get("profile") if isinstance(data, dict) else None
    return str(name) if name else None


def choose(base: Path, name: str) -> None:
    path = _choice_file(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"profile": name}), "utf-8")


def active_name(
    settings: Settings,
    explicit: str | None = None,
    program: str = BASE,
    infos: Mapping[str, AdapterInfo] | None = None,
) -> str:
    """Which profile to use (see the module)."""
    infos = adapter_infos() if infos is None else infos
    for name in (explicit, os.environ.get(ENVIRONMENT), chosen(settings.shared_dir)):
        if name:
            return name
    if settings.profile:
        return settings.profile
    own = [i.name for i in sorted(infos.values(), key=lambda i: i.name) if i.own_profile]
    allowed = own if settings.adapters is None else [n for n in own if n in settings.adapters]
    if program in allowed:
        return program
    return allowed[0] if allowed else BASE


def apply(
    settings: Settings, profile: Profile, infos: Mapping[str, AdapterInfo] | None = None
) -> Settings:
    """``settings`` (from cassi.yaml) for ``profile``: its data folder, its adapters and its
    own settings. Downloads stay in the shared folder."""
    infos = adapter_infos() if infos is None else infos
    base = settings.shared_dir
    data = settings.model_dump()
    _merge(data, dict(profile.settings))
    # Adapters that are AIs of their own only where the profile builds on them; others
    # (tools for every AI) as cassi.yaml says.
    general = [
        name
        for name in (sorted(infos) if settings.adapters is None else settings.adapters)
        if not (infos.get(name) and infos[name].own_profile)
    ]
    data.update(
        data_dir=profile.folder,
        downloads_dir=settings.downloads_dir or base,
        profile=profile.name,
        base_dir=base,
        adapters=general + [a for a in profile.adapters if a not in general],
    )
    return Settings.model_validate(data)


def resolve(
    settings: Settings,
    explicit: str | None = None,
    program: str = BASE,
    infos: Mapping[str, AdapterInfo] | None = None,
) -> Settings:
    """``settings`` from cassi.yaml for the profile to use (raises ``ProfileError``)."""
    infos = adapter_infos() if infos is None else infos
    name = active_name(settings, explicit, program, infos)
    profile = find(settings.shared_dir, name, infos)
    profile.folder.mkdir(parents=True, exist_ok=True)
    return apply(settings, profile, infos)


def _merge(into: dict[str, Any], changes: Mapping[str, Any]) -> None:
    for key, value in changes.items():
        if isinstance(value, Mapping) and isinstance(into.get(key), dict):
            _merge(into[key], value)
        else:
            into[key] = value


def create(
    base: Path,
    title: str,
    description: str = "",
    template: Profile | None = None,
    copy: Copy = "definition",
    settings: Mapping[str, Any] | None = None,
    adapters: Iterable[str] | None = None,
    infos: Mapping[str, AdapterInfo] | None = None,
) -> Profile:
    """A new profile. From ``template``: nothing but what it builds on, its definition (its
    settings and the files that define it, not what it learned or remembers), or everything
    (a copy, memory included)."""
    infos = adapter_infos() if infos is None else infos
    name = slug(title)
    if not name:
        raise ProfileError("Der Name braucht Buchstaben oder Ziffern")
    if any(p.name == name for p in available(base, infos)):
        raise ProfileError(f"Ein Profil „{name}“ gibt es schon")
    folder = base / FOLDER / name
    built_on = tuple(adapters) if adapters is not None else template.adapters if template else ()
    own: dict[str, Any] = {}
    if template is not None and copy != "nothing":
        own = json.loads(json.dumps(dict(template.settings)))  # a deep copy
        if copy == "everything":
            shutil.copytree(
                template.folder, folder, ignore=_not_copied(template.folder), dirs_exist_ok=True
            )
        else:
            for adapter in template.adapters:
                for relative in infos[adapter].definition_files if adapter in infos else ():
                    source = template.folder / relative
                    if source.is_file():
                        (folder / relative).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, folder / relative)
    _merge(own, dict(settings or {}))
    folder.mkdir(parents=True, exist_ok=True)
    character = dict(template.character) if template is not None and copy != "nothing" else {}
    _write(folder, title.strip(), description.strip(), built_on, own, character)
    return _read(folder)


def _write(
    folder: Path,
    title: str,
    description: str,
    adapters: Iterable[str],
    settings: Mapping[str, Any],
    character: Mapping[str, str],
) -> None:
    definition: dict[str, Any] = {
        "title": title,
        "description": description,
        "adapters": list(adapters),
    }
    if character:
        definition["character"] = dict(character)
    definition["settings"] = json.loads(json.dumps(dict(settings)))  # plain dicts for YAML
    (folder / DEFINITION).write_text(
        "# Definition dieses Profils – Einstellungen wie in cassi.yaml, nur was abweicht.\n"
        + yaml.safe_dump(definition, allow_unicode=True, sort_keys=False),
        "utf-8",
    )


def define(
    base: Path,
    name: str,
    definition: Definition,
    title: str = "",
    description: str = "",
    infos: Mapping[str, AdapterInfo] | None = None,
) -> Profile:
    """Change who an AI is – name, personality, voice, model – and its title or description;
    what is empty stays. An AI built on an adapter that defines its AIs itself (Flay: its
    character and Modelfile) gets it in that adapter's files, else in profile.yaml."""
    infos = adapter_infos() if infos is None else infos
    profile = find(base, name, infos)
    hooks = [
        hook
        for adapter in profile.adapters
        if adapter in infos and (hook := infos[adapter].define) is not None
    ]
    if profile.builtin and not hooks and (definition or title or description):
        raise ProfileError(
            f"„{profile.label}“ ist eingebaut – seine Einstellungen stehen in cassi.yaml "
            "(Tab „Einstellungen“). Für eine eigene KI: „Neues Profil anlegen“."
        )
    if profile.builtin and (title or description):
        raise ProfileError(f"„{profile.label}“ ist eingebaut – Name und Zweck bleiben")
    profile.folder.mkdir(parents=True, exist_ok=True)
    settings: dict[str, Any] = json.loads(json.dumps(dict(profile.settings)))
    character = dict(profile.character)
    if definition:
        if hooks:
            for hook in hooks:
                _merge(settings, dict(hook(profile.folder, definition)))
        else:
            _merge(settings, _cassi_settings(definition, settings, character))
    if profile.builtin:
        if settings != dict(profile.settings):
            raise ProfileError(f"„{profile.label}“ ist eingebaut – nicht alles ließ sich speichern")
        return profile
    _write(
        profile.folder,
        title.strip() or profile.title,
        description.strip() or profile.description,
        profile.adapters,
        settings,
        character,
    )
    return _read(profile.folder)


def _cassi_settings(
    definition: Definition, settings: Mapping[str, Any], character: dict[str, str]
) -> dict[str, Any]:
    """``definition`` as Cassi's settings; updates ``character`` (kept for the next change)."""
    from cassi.config import PersonaSettings

    changes: dict[str, Any] = {}
    persona = dict(settings.get("persona") or {})
    if definition.identity:
        character["identity"] = definition.identity.strip()
    if definition.speaking_style:
        character["speaking_style"] = definition.speaking_style.strip()
    if definition.name or definition.identity or definition.speaking_style:
        name = definition.name.strip() or str(persona.get("name") or PersonaSettings().name)
        changes["persona"] = {"name": name, "system_prompt": system_prompt(name, character)}
    voice: dict[str, Any] = {}
    if definition.voice:
        voice["name"] = definition.voice
    if definition.pitch is not None:
        voice["pitch"] = definition.pitch
    if definition.speed is not None:
        voice["speed"] = definition.speed
    if voice:
        changes["tts"] = {"voice": voice}
    if definition.model:
        changes["llm"] = {"model": definition.model}
    return changes


SPOKEN = (
    "Your replies are spoken aloud: say what the request needs in natural sentences – often "
    "one is enough – without markdown, lists or emojis. No sign-off, no offer of further help, "
    "no remark that you are there."
)


def system_prompt(name: str, character: Mapping[str, str]) -> str:
    """A system prompt for a Cassi-based AI from its name and character (English: small
    models follow it best; the reply language is added per turn)."""
    parts = [f"You are {name}, a voice assistant on the user's computer."]
    if identity := character.get("identity"):
        parts.append(f"Who you are: {identity}")
    if style := character.get("speaking_style"):
        parts.append(f"How you talk: {style}")
    parts.append(
        f"Stay honest: you are an AI. Never claim to be human or to have experienced things "
        f"you have not. {SPOKEN}"
    )
    return "\n".join(parts)


def _not_copied(root: Path) -> Callable[[str, list[str]], list[str]]:
    def ignore(directory: str, names: list[str]) -> list[str]:
        top = Path(directory) == root
        return [
            n
            for n in names
            if (top and n in NOT_COPIED)
            or n.endswith((".log", "-wal", "-shm", "-journal"))
            or (top and n.startswith("manage-") and n.endswith(".json"))
            or (top and n == DEFINITION)
        ]

    return ignore


def remove(
    base: Path, name: str, active: str, infos: Mapping[str, AdapterInfo] | None = None
) -> Path:
    """Remove an own profile (not a built-in one, not the one in use). Kept as a zip in the
    shared backups folder first; returns where."""
    profile = find(base, name, infos)
    if profile.builtin:
        raise ProfileError(f"„{profile.label}“ ist eingebaut und lässt sich nicht löschen")
    if profile.name == active:
        raise ProfileError(f"„{profile.label}“ ist gerade in Gebrauch – erst wechseln")
    backups = base / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stem = backups / f"profil-{profile.name}-{time.strftime('%Y-%m-%d-%H%M%S')}"
    archive = Path(shutil.make_archive(str(stem), "zip", profile.folder))
    shutil.rmtree(profile.folder)
    return archive


def size(folder: Path) -> int:
    """Bytes in a profile's folder (without other profiles in Cassi's)."""
    total = 0
    for path in folder.rglob("*") if folder.is_dir() else []:
        relative = path.relative_to(folder).parts
        if relative and relative[0] in NOT_COPIED:
            continue
        if path.is_file():
            total += path.stat().st_size
    return total
