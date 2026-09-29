"""What the web pages need to know about the exported GTM model in gtm_model/ (FR xxviii)."""

from __future__ import annotations

from functools import lru_cache

from django.conf import settings

from sonic.gtm.model_info import model_info


@lru_cache(maxsize=1)
def gtm_info() -> dict:
    """Labels and a version string from the export's metadata.json; empty if the model is missing."""
    return model_info(settings.BASE_DIR / "gtm_model")
