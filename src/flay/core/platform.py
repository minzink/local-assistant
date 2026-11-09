"""Hardware detection used to pick backends when the config says ``auto``.

Deliberately avoids importing torch or other heavy libraries.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from enum import StrEnum


class Accelerator(StrEnum):
    APPLE_SILICON = "apple_silicon"
    CUDA = "cuda"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    system: str
    machine: str
    accelerator: Accelerator


def detect_hardware() -> HardwareProfile:
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin" and machine == "arm64":
        accelerator = Accelerator.APPLE_SILICON
    elif shutil.which("nvidia-smi"):
        accelerator = Accelerator.CUDA
    else:
        accelerator = Accelerator.CPU

    return HardwareProfile(system=system, machine=machine, accelerator=accelerator)
