"""Facts about the exported GTM model in gtm_model/ (labels, version), shared by the web app and the comparison report."""

from __future__ import annotations

import json
from pathlib import Path

from ..dataset.config import REPO_ROOT

GTM_MODEL_DIR = REPO_ROOT / "gtm_model"


def model_info(folder: Path = GTM_MODEL_DIR) -> dict:
    """Labels and a version string from the export's metadata.json; empty if the model is missing."""
    path = folder / "metadata.json"
    if not path.exists():
        return {"available": False, "version": "", "labels": []}
    meta = json.loads(path.read_text(encoding="utf-8"))
    stamp = meta.get("timeStamp", "")[:19].replace(":", "").replace("-", "").replace("T", "-")
    return {
        "available": True,
        "version": f"gtm-{meta.get('modelName', 'TM')}-{stamp}",
        "labels": meta.get("wordLabels", []),
        "speech_commands": meta.get("tfjsSpeechCommandsVersion", ""),
    }
