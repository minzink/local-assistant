"""`cassi eval-computer`: operate_computer with real clicks on harmless goals (eval/computer.yaml).

Unlike `cassi eval`, nothing is simulated: Cassi really opens TextEdit, types, clicks through
System Settings. Each case is judged by what the screen shows afterwards (front app, window
title, text found by OCR). Reports success, moves and seconds per case.

Do not touch mouse and keyboard while it runs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from cassi.config import Settings

log = logging.getLogger(__name__)


@dataclass
class ComputerResult:
    goal: str
    ok: bool
    problem: str
    moves: int
    seconds: float
    said: str


def judge(check: dict[str, Any]) -> str:
    """What is wrong with the screen now ('' if it is as expected)."""
    from cassi.tools import ocr
    from cassi.tools.screen import is_own, list_windows

    windows = [w for w in list_windows() if not is_own(w)]
    front = windows[0] if windows else None
    app = check.get("front_app")
    names = [app] if isinstance(app, str) else list(app or [])
    if names and (front is None or not any(n.casefold() in front.app.casefold() for n in names)):
        return f"vorn ist {front.app if front else 'nichts'} statt {app}"
    if titles := check.get("title"):
        title = front.title if front else ""
        if not any(str(t).casefold() in title.casefold() for t in titles):
            return f"Fenstertitel „{title}“ enthält nichts von {titles}"
    gone = check.get("no_window")
    gone_names = [gone] if isinstance(gone, str) else list(gone or [])
    if any(n.casefold() in w.app.casefold() for n in gone_names for w in windows):
        return f"{gone} hat noch ein Fenster"
    if texts := check.get("text"):
        from cassi.tools.screen import PrivateScreenError

        try:
            boxes = ocr.read_screen(front.bounds) if front else []
        except PrivateScreenError as exc:
            return f"Text nicht lesbar: {exc}"
        if not any(ocr.find_text(str(t), boxes) for t in texts) and not any(
            str(t).casefold() in box.text.casefold() for t in texts for box in boxes
        ):
            return f"nichts von {texts} auf dem Bildschirm"
    return ""


def _shell(commands: list[str]) -> None:
    for command in commands:
        subprocess.run(
            command if sys.platform == "win32" else shlex.split(command),
            shell=sys.platform == "win32",
            capture_output=True,
            timeout=30,
            check=False,
        )
    time.sleep(1.0)


async def run(settings: Settings, path: Path, print_line: Any = print) -> int:
    from cassi.core.messages import ToolCall
    from cassi.tools.factory import create_tools

    data = yaml.safe_load(path.read_text("utf-8")).get(sys.platform)
    if not data:
        print_line(f"Keine Fälle für {sys.platform} in {path}.")
        return 1
    tools = create_tools(settings)
    computer = tools.get("operate_computer")
    if computer is None:
        print_line("operate_computer ist nicht eingeschaltet (tools.enabled).")
        return 1
    _shell(data.get("setup", []))
    results: list[ComputerResult] = []
    try:
        for case in data["cases"]:
            start = time.monotonic()
            from cassi.core.messages import ToolCall

            result = await tools.execute(
                ToolCall("eval", "operate_computer", {"goal": case["goal"]})
            )
            seconds = time.monotonic() - start
            await asyncio.sleep(0.8)
            problem = await asyncio.to_thread(judge, case.get("check", {}))
            if result.confirm is not None:
                problem = problem or "hat nachgefragt"
            moves = result.text.count(" → ") + 1 if "moves:" in result.text else 0
            said = next(iter(result.say.values())) if result.say else result.text
            results.append(ComputerResult(case["goal"], not problem, problem, moves, seconds, said))
            mark = "✓" if not problem else "✗"
            print_line(f"{mark} {seconds:5.1f} s  {case['goal']}")
            if problem:
                print_line(f"      → {problem} (Cassi: {said})")
    finally:
        _shell(data.get("cleanup", []))
    ok = sum(r.ok for r in results)
    total = sum(r.seconds for r in results)
    print_line(f"\nRichtig: {ok}/{len(results)}, zusammen {total:.0f} s")
    out = settings.data_dir / "eval" / f"{datetime.now():%Y%m%d-%H%M%S}-computer.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=1))
    print_line(f"Ergebnisse: {out}")
    return 0 if ok == len(results) else 1
