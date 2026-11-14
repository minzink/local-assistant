"""Where the time of a request goes: phases and model calls, per turn.

A turn starts a profile (``begin``); code marks phases with ``span("state")`` and the LLM
client reports each call (``llm_call``) with its purpose – the innermost open span. All of
it is collected through a context variable, so tasks started during the turn (recall,
classification) report into the same profile without passing it around.

At the end the assistant logs one line ("Profil: …") and publishes the profile; the history
keeps it, and `flay stats --perf` / `flay eval` sum it up.

Prefill speed tells whether the server reused the start of the prompt: thousands of tokens
per second mean the prefix was cached, a few hundred mean it was processed again.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMCall:
    purpose: str
    model: str
    seconds: float
    first_token: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    images: int = 0

    @property
    def prefill_rate(self) -> float | None:
        """Prompt tokens per second until the first token (≈ prompt processing speed)."""
        if not self.prompt_tokens or not self.first_token:
            return None
        return self.prompt_tokens / self.first_token

    @property
    def decode_rate(self) -> float | None:
        if not self.output_tokens or self.first_token is None:
            return None
        generating = self.seconds - self.first_token
        return self.output_tokens / generating if generating > 0.05 else None


@dataclass
class Profile:
    start: float = field(default_factory=time.perf_counter)
    # Phase name → seconds spent in it (summed if it happens more than once).
    phases: dict[str, float] = field(default_factory=dict)
    # Moments since the start: first_sentence, first_audio, done …
    marks: dict[str, float] = field(default_factory=dict)
    calls: list[LLMCall] = field(default_factory=list)

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, time.perf_counter() - self.start)

    def add(self, phase: str, seconds: float) -> None:
        self.phases[phase] = self.phases.get(phase, 0.0) + seconds

    @property
    def llm_seconds(self) -> float:
        return sum(c.seconds for c in self.calls)

    def summary(self) -> str:
        """One log line: moments, phases, model calls."""
        parts = [f"{name} {s:.2f}s" for name, s in self.marks.items()]
        parts += [f"[{name} {s:.2f}s]" for name, s in self.phases.items()]
        for call in self.calls:
            tokens = (
                f" {call.prompt_tokens}→{call.output_tokens} tok"
                if call.prompt_tokens is not None
                else ""
            )
            rate = f" {call.prefill_rate:.0f} tok/s ein" if call.prefill_rate else ""
            first = f" erstes {call.first_token:.2f}s" if call.first_token is not None else ""
            image = f" {call.images} Bild" if call.images else ""
            parts.append(f"LLM {call.purpose} {call.seconds:.2f}s{first}{tokens}{rate}{image}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phases": {k: round(v, 3) for k, v in self.phases.items()},
            "marks": {k: round(v, 3) for k, v in self.marks.items()},
            "calls": [
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(c).items()}
                for c in self.calls
            ],
        }


_profile: ContextVar[Profile | None] = ContextVar("flay_profile", default=None)
# Requests outside any span (memory upkeep, summaries) count as "other".
_purpose: ContextVar[str] = ContextVar("flay_purpose", default="other")

# Purposes whose prompt starts like the conversation's: after them the server's cached
# prompt start fits the next answer. Anything else pushed it out (see Assistant._rewarm).
CONVERSATION_PURPOSES = frozenset({"answer", "warm", "route_shared"})


@dataclass
class _ServerState:
    last_purpose: str = "warm"
    active: int = 0


SERVER = _ServerState()


@contextlib.contextmanager
def request(purpose: str) -> Iterator[None]:
    """Around every request to the model server: what it was for, and whether one runs."""
    SERVER.active += 1
    try:
        yield
    finally:
        SERVER.active -= 1
        SERVER.last_purpose = purpose


def cache_fits_conversation() -> bool:
    """The server's cached prompt start is (probably) the conversation's."""
    return SERVER.last_purpose in CONVERSATION_PURPOSES


def begin() -> Profile:
    """A new profile for the turn that runs in this context (and the tasks it starts)."""
    profile = Profile()
    _profile.set(profile)
    _purpose.set("other")
    return profile


def current() -> Profile | None:
    return _profile.get()


@contextlib.contextmanager
def span(phase: str) -> Iterator[None]:
    """Time a phase; model calls inside it count as made for ``phase``."""
    start = time.perf_counter()
    token = _purpose.set(phase)
    try:
        yield
    finally:
        _purpose.reset(token)
        if (profile := _profile.get()) is not None:
            profile.add(phase, time.perf_counter() - start)


def purpose() -> str:
    return _purpose.get()


def llm_call(call: LLMCall) -> None:
    if (profile := _profile.get()) is not None:
        profile.calls.append(call)


def mark(name: str) -> None:
    if (profile := _profile.get()) is not None:
        profile.mark(name)
