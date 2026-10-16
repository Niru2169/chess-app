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
        "GNN_ENABLED": getattr(settings, "AUTOCHESS_GNN_ENABLED", True),
        "GNN_REFRESH_MS": getattr(settings, "AUTOCHESS_GNN_REFRESH_MS", 1500),
        "GNN_TOP_P": getattr(settings, "AUTOCHESS_GNN_TOP_P", 0.9),
    }
