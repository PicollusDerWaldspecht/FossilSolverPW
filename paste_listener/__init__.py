"""Streamlit custom component that listens for global Ctrl+V paste events.

Whenever the user pastes an image anywhere on the Streamlit page, the
component returns a dict ``{"url": <data URL>, "ts": <timestamp>}``. The
helper :func:`global_paste_image` decodes that into a PIL image.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import streamlit.components.v1 as components
from PIL import Image

_COMPONENT_DIR = Path(__file__).parent

_paste_component = components.declare_component(
    "global_paste_listener",
    path=str(_COMPONENT_DIR),
)


def global_paste_image(key: str = "global_paste") -> tuple[Image.Image | None, int | None]:
    """Render the invisible paste listener and return any pasted image.

    Returns a tuple ``(image, ts)``. ``image`` is ``None`` when nothing has
    been pasted yet. ``ts`` is the JS millisecond timestamp of the paste so
    callers can detect whether a new paste has happened since the last run.
    """

    payload = _paste_component(key=key, default=None)
    if not payload or not isinstance(payload, dict):
        return None, None

    data_url = payload.get("url")
    ts = payload.get("ts")
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        return None, ts if isinstance(ts, int) else None

    try:
        _, encoded = data_url.split(",", 1)
        binary = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(binary))
        image.load()
    except Exception:
        return None, ts if isinstance(ts, int) else None

    return image, ts if isinstance(ts, int) else None
