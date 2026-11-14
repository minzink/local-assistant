"""Plans: several steps in one request ("…, dann …"), routines (named plans) and how a step
is routed. Part of the assistant (cassi.assistant.Assistant owns one Planner); it uses the
assistant's tools, model and way of answering.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from cassi import perf
from cassi.core.language import Language
from cassi.core.messages import (
    Message,
    Role,
    TextPart,
    ToolCall,
)
from cassi.filler import TurnFiller
from cassi.intents import (
    _LOOK_STEP,
    _NO,
    _YES,
    CANCELLED,
    CONFIRM_SECONDS,
    REPORT_CHARS,
    SEEN_CHARS,
    file_intent,
    fixed_reply,
    tool_intent,
    web_intent,
)
from cassi.llm.base import TextDelta
from cassi.plans import FAR_REACHING, plan_intents, split_steps
from cassi.plans import describe as describe_plan
from cassi.tools.base import ToolResult, speech

if TYPE_CHECKING:
    from cassi.assistant import Assistant, TurnTimings

log = logging.getLogger(__name__)


class Planner:
    def __init__(self, host: Assistant) -> None:
        self._host = host
        # A plan waiting for "ja" (it has a far-reaching step), and when it was asked.
        self.pending: tuple[list[str], float] | None = None

    def step_intent(self, step: str) -> str | None:
        """The tool one step of a plan clearly asks for."""
        if _LOOK_STEP.search(step) and "look_at_screen" in self._host._tools:
            # "Mach einen Screenshot, um mir zu sagen, was auf google.com steht": the user
            # asked for the picture, even though a page is named.
            return "look_at_screen"
        for found in (tool_intent(step), self._host._computer_intent(step), web_intent(step)):
            if found is not None and found in self._host._tools:
                return found
        if file_intent(step) and "read_file" in self._host._tools:
            return "read_file"
        return None

    def plan_steps(self, text: str) -> list[str] | None:
        """The steps if the request is a plan of several actions."""
        steps = split_steps(text)
        intents = plan_intents(steps, self.step_intent) if len(steps) >= 2 else None
        if intents is None:
            if file_intent(text) or web_intent(text) is not None:
                # "Fass die README zusammen und gib es Claude": one task, which reading the
                # file or page does as a whole (and hands on).
                return None
            # "Schau dir den Bildschirm an und kopier den Text an Claude": two different
            # actions joined by "und".
            steps = split_steps(text, at_and=True)
            intents = plan_intents(steps, self.step_intent, distinct=True)
        if intents is None:
            return None
        # "Öffne heise.de, dann klick auf Impressum" is one task on a page: read_webpage
        # does it inside the page, better than two separate steps.
        page_steps = {"open_url", "read_webpage", "click_on_screen", "scroll_screen", None}
        if web_intent(text) is not None and set(intents) <= page_steps:
            return None
        return steps

    def confirmed_plan(self, text: str) -> list[str] | None:
        """The waiting plan, if ``text`` says yes to it."""
        pending, self.pending = self.pending, None
        if pending is None or time.monotonic() - pending[1] > CONFIRM_SECONDS:
            return None
        if len(text.split()) <= 6 and _YES.search(text):
            return pending[0]
        if not _NO.search(text):
            return None
        self.pending = None
        return []  # declined: nothing to run, said below

    async def plan(
        self,
        user_message: Message,
        steps: list[str],
        language: Language,
        timings: TurnTimings,
        filler: TurnFiller | None,
        state: str | None,
    ) -> None:
        """Say the steps; ask first if one reaches far, else do them right away."""
        intents = [self.step_intent(step) for step in steps]
        log.info(
            "Plan: %s", " → ".join(f"{s} [{i or '?'}]" for s, i in zip(steps, intents, strict=True))
        )
        described = describe_plan(steps, language.value)
        if FAR_REACHING & set(filter(None, intents)):
            self.pending = (steps, time.monotonic())
            question = speech("Soll ich loslegen?", "Shall I start?")[language]
            await self._host._respond(
                user_message, language, timings, filler=filler, fixed=f"{described} {question}"
            )
            return
        await self.run_plan(user_message, steps, language, timings, filler, state, described)

    async def run_plan(
        self,
        user_message: Message,
        steps: list[str],
        language: Language,
        timings: TurnTimings,
        filler: TurnFiller | None,
        state: str | None,
        described: str | None = None,
    ) -> None:
        if not steps:
            await self._host._respond(
                user_message, language, timings, filler=filler, fixed=CANCELLED[language]
            )
            return

        async def work(say: Callable[[str], None]) -> tuple[list[Message], str | None]:
            preludes: list[Message] = []
            reports: list[str] = []
            silent = False
            for number, step in enumerate(steps, start=1):
                # Each step sees what the earlier ones reported ("schick Claude die Fehler").
                context = step + (
                    f"\n(Earlier steps reported: {' | '.join(reports)})" if reports else ""
                )
                routed = Message.from_text(Role.USER, context, language)
                intent = self.step_intent(step)
                if intent is not None:
                    prelude, results = await self._host._route_tool(intent, routed, language, state)
                else:
                    prelude, results = await self.route_any(routed, language, state)
                timings.mark("tool")
                preludes += prelude
                if not results:
                    stop = speech(
                        f"Bei Schritt {number}, {step}, wusste ich nicht, was zu tun ist. Ich habe "
                        "dort aufgehört.",
                        f"At step {number}, {step}, I didn't know what to do. I stopped there.",
                    )
                    return [], stop[language]
                spoken = fixed_reply(results, language)
                if not all(result.ok for result in results):
                    stopped = speech(
                        " Die übrigen Schritte habe ich gelassen.", " I left the remaining steps."
                    )[language]
                    return [], (
                        spoken or speech("Das hat nicht geklappt.", "That failed.")[language]
                    ) + stopped
                if spoken:
                    say(spoken)
                else:
                    silent = True  # the model sums up what this step found
                if any(result.images for result in results) and number < len(steps):
                    # The next step is routed without the picture: it gets what the picture
                    # shows for this request, text quoted word for word.
                    seen = await self.read_images(results, user_message.text, language)
                    reports.append(f"{step}: {seen}")
                    continue
                reports += [f"{step}: {result.text[:REPORT_CHARS]}" for result in results]
            if silent:
                return preludes, None
            return [], speech("Fertig.", "Done.")[language]

        await self._host._respond(
            user_message,
            language,
            timings,
            filler=filler,
            state=state,
            announce=described,
            work=work,
        )

    async def read_images(self, results: list[ToolResult], request: str, language: Language) -> str:
        """What the screenshots of a plan step show, for the steps after it."""
        images = [image for result in results for image in result.images]
        question = Message(
            Role.USER,
            (
                TextPart(
                    f"The user asked: {request}\nDescribe what this screenshot shows that "
                    "matters for the request. Quote any text the request is about word for "
                    "word and completely."
                ),
                *images,
            ),
            language,
        )
        with perf.span("read_image"):
            parts = [
                event.text
                async for event in self._host._llm.stream([question])
                if isinstance(event, TextDelta)
            ]
        return "".join(parts).strip()[:SEEN_CHARS]

    async def route_any(
        self, user_message: Message, language: Language, state: str | None
    ) -> tuple[list[Message], list[ToolResult]]:
        """One step without a clear tool: the model picks the tool and fills it in."""
        router = Message.from_text(
            Role.SYSTEM,
            "Turn the user's request into exactly one tool call. Only name a project if the "
            "user names one." + (f"\n{state}" if state else "") + self._host._rules(),
        )
        request = Message.from_text(Role.USER, user_message.text, language)
        calls = [
            event
            async for event in self._host._llm.stream([router, request], self._host._tools.specs)
            if isinstance(event, ToolCall)
        ]
        if not calls:
            return [], []
        return await self._host._act(calls[0], language)

    async def routine(
        self,
        user_message: Message,
        text: str,
        language: Language,
        timings: TurnTimings,
        filler: TurnFiller | None,
        state: str | None,
    ) -> bool:
        """Set up, forget or run a routine; False if the request is about none."""
        assert self._host._routines is not None
        if (defined := self._host._routines.define(text)) is not None:
            name, wording = defined
            listed = describe_plan(split_steps(wording) or [wording], language.value)
            said = speech(
                f"Ich habe mir {name} gemerkt. {listed}",
                f"I've saved {name}. {listed}",
            )[language]
            await self._host._respond(user_message, language, timings, filler=filler, fixed=said)
            return True
        if (forgotten := self._host._routines.forget(text)) is not None:
            said = speech(f"{forgotten} habe ich vergessen.", f"I forgot {forgotten}.")[language]
            await self._host._respond(user_message, language, timings, filler=filler, fixed=said)
            return True
        if (routine := self._host._routines.requested(text)) is not None:
            log.info("Routine „%s“: %s", routine[0], routine[1])
            planned = split_steps(routine[1]) or [routine[1]]
            await self.plan(user_message, planned, language, timings, filler, state)
            return True
        return False
