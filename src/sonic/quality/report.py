"""Dataset quality report (SRS dataset deliverable "Quality report").

``python -m sonic.quality report`` grades every clip in the built dataset and
writes ``data/metadata/quality_report.csv`` (one row per clip) and
``data/metadata/quality_summary.json`` (grades per class, most common issues).
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..dataset.config import DEFAULT_METADATA_DIR, DEFAULT_OUT_DIR
from .analyze import GRADES, assess_file

BACKGROUND_CLASS = "Background Noise"
METRICS = ["duration_s", "sample_rate", "peak_dbfs", "active_level_dbfs", "noise_floor_dbfs",
           "dynamic_range_db", "spectral_flatness", "clipped_fraction", "silent_fraction", "dropouts"]


def _assess(job: tuple[dict, Path]) -> dict:
    row, path = job
    result = assess_file(path, expect_event=row["class_name"] != BACKGROUND_CLASS)
    return {
        "audio_id": row["audio_id"],
        "class_name": row["class_name"],
        "source_dataset": row["source_dataset"],
        "grade": result.grade,
        "issues": "; ".join(result.issues),
        **{m: result.metrics.get(m, "") for m in METRICS},
    }


def build_report(dataset_dir: Path = DEFAULT_OUT_DIR, metadata_dir: Path = DEFAULT_METADATA_DIR, workers: int = 4) -> dict:
    with open(metadata_dir / "dataset_metadata.csv", newline="", encoding="utf-8") as f:
        clips = list(csv.DictReader(f))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_assess, [(c, dataset_dir / c["filename"]) for c in clips], chunksize=32))

    with open(metadata_dir / "quality_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_class: dict[str, Counter] = defaultdict(Counter)
    issues: Counter = Counter()
    for r in rows:
        per_class[r["class_name"]][r["grade"]] += 1
        for issue in filter(None, r["issues"].split("; ")):
            issues[issue.split(":")[0]] += 1
    summary = {
        "clips": len(rows),
        "grades": {g: sum(c[g] for c in per_class.values()) for g in GRADES},
        "per_class": {k: {g: v[g] for g in GRADES} for k, v in sorted(per_class.items())},
        "issues": dict(issues.most_common()),
    }
    (metadata_dir / "quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_summary(summary: dict) -> str:
    lines = [f"{'Class':<24}" + "".join(f"{g:>12}" for g in GRADES)]
    for name, counts in summary["per_class"].items():
        lines.append(f"{name:<24}" + "".join(f"{counts[g]:>12}" for g in GRADES))
    lines.append(f"{'TOTAL':<24}" + "".join(f"{summary['grades'][g]:>12}" for g in GRADES))
    lines.append("\nIssues found (a clip can have several):")
    lines.extend(f"  {name}: {n}" for name, n in summary["issues"].items())
    return "\n".join(lines)
