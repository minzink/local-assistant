"""LLM latency benchmark: time to first token and output speed per prompt."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from flay.core.messages import Message, Role
from flay.llm.base import ChatModel, TextDelta

PROMPTS = [
    "Wie spät ist es ungefähr, wenn die Sonne untergeht?",
    "Erzähl mir in zwei Sätzen etwas über Sonnenberg.",
    "Was ist der Unterschied zwischen Wetter und Klima?",
    "Give me one tip for better sleep.",
]


@dataclass(frozen=True)
class BenchResult:
    prompt: str
    first_token: float
    total: float
    chars: int

    @property
    def chars_per_second(self) -> float:
        generation = self.total - self.first_token
        return self.chars / generation if generation > 0 else 0.0


async def run_prompt(model: ChatModel, system_prompt: str, prompt: str) -> BenchResult:
    messages = [Message.from_text(Role.SYSTEM, system_prompt), Message.from_text(Role.USER, prompt)]
    start = time.perf_counter()
    first_token: float | None = None
    chars = 0
    async for event in model.stream(messages):
        if isinstance(event, TextDelta):
            first_token = first_token or time.perf_counter() - start
            chars += len(event.text)
    total = time.perf_counter() - start
    return BenchResult(prompt, first_token or total, total, chars)


async def benchmark(model: ChatModel, system_prompt: str, rounds: int) -> list[BenchResult]:
    # The first request loads the model on most servers; it is not measured.
    await run_prompt(model, system_prompt, PROMPTS[0])
    return [
        await run_prompt(model, system_prompt, prompt) for _ in range(rounds) for prompt in PROMPTS
    ]


def format_report(results: list[BenchResult]) -> str:
    first = [r.first_token for r in results]
    speed = [r.chars_per_second for r in results]
    return (
        f"Anfragen:           {len(results)}\n"
        f"Erstes Token:       median {statistics.median(first):.2f}s, max {max(first):.2f}s\n"
        f"Ausgabe:            median {statistics.median(speed):.0f} Zeichen/s"
    )
