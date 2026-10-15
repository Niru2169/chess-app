from __future__ import annotations

from django.conf import settings
from typing import Any

def autochess_settings(request) -> dict[str, Any]:
    """Template context: AutoChess runtime config.

    Args:
        request: Django request (unused).

    Returns:
        Context fields for templates.
    """
    return {
        "AUTOCHESS_MAX_PLIES": getattr(settings, "AUTOCHESS_MAX_PLIES", 600),
    }
