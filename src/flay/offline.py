"""Flay runs offline: models are only downloaded by `flay setup`, never while running."""

from __future__ import annotations

import ipaddress
import os
from urllib.parse import urlparse


class MissingModelError(RuntimeError):
    """A model file is not on disk and Flay is not allowed to download it now."""

    def __init__(self, what: str) -> None:
        super().__init__(f"{what} fehlt – einmalig 'flay setup' ausführen, das lädt alle Modelle.")


def enforce_offline(allow_downloads: bool) -> None:
    """Configure libraries before they are imported (Hugging Face reads this at import)."""
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if not allow_downloads:
        os.environ["HF_HUB_OFFLINE"] = "1"


def is_local_url(url: str) -> bool:
    """True if ``url`` points to this computer."""
    host = urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
