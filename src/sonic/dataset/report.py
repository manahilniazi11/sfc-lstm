"""Dataset statistics and balance checks (SRS FR xviii: dataset balance checking)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .splitting import SPLITS

# Smallest class must have at least this fraction of the largest class.
BALANCE_RATIO = 0.8


def compute_statistics(rows: list[dict], rejected: list[dict], classes, opts) -> dict:
    per_class: dict[str, dict] = {}
    for sound_class in classes:
        class_rows = [r for r in rows if r["class_name"] == sound_class.name]
        per_split = Counter(r["split"] for r in class_rows)
        per_class[sound_class.name] = {
            **{s: per_split.get(s, 0) for s in SPLITS},
            "total": len(class_rows),
            "unique_recordings": len({r["origin_id"] for r in class_rows}),
            "duration_s": round(sum(float(r["duration_s"]) for r in class_rows), 1),
            "sources": dict(Counter(r["source_dataset"] for r in class_rows)),
            "mandatory": sound_class.mandatory,
        }

    rejected_by_reason: dict[str, int] = defaultdict(int)
    for r in rejected:
        rejected_by_reason[r["reason"].split(" (")[0].split(":")[0]] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": opts.seed,
        "per_class_target": opts.per_class,
        "split_ratios": dict(zip(SPLITS, opts.ratios)),
        "totals": {
            **{s: sum(c[s] for c in per_class.values()) for s in SPLITS},
            "clips": len(rows),
            "unique_recordings": len({r["origin_id"] for r in rows}),
            "duration_s": round(sum(float(r["duration_s"]) for r in rows), 1),  # float(): rows read back from the CSV are text
            "rejected": len(rejected),
        },
        "classes": per_class,
        "rejected_by_reason": dict(sorted(rejected_by_reason.items())),
        "warnings": balance_warnings(per_class, opts.per_class),
    }


def balance_warnings(per_class: dict[str, dict], target: int) -> list[str]:
    warnings = []
    for name, c in per_class.items():
        if c["total"] == 0:
            if c["mandatory"]:
                warnings.append(f"{name}: no clips at all (mandatory class)")
            continue
        if target and c["total"] < target:
            warnings.append(f"{name}: {c['total']} clips, short of the {target} target by {target - c['total']}")
        if target and c["unique_recordings"] < target:
            warnings.append(
                f"{name}: only {c['unique_recordings']} unique original recordings "
                f"(the SRS counts unique originals toward the {target} per class)"
            )
        empty = [s for s in SPLITS if c[s] == 0]
        if empty:
            warnings.append(f"{name}: no clips in {', '.join(empty)} split")

    totals = [c["total"] for c in per_class.values() if c["mandatory"]]
    if totals and max(totals) and min(totals) / max(totals) < BALANCE_RATIO:
        warnings.append(
            f"class imbalance: smallest mandatory class has {min(totals)} clips, "
            f"largest has {max(totals)} (ratio below {BALANCE_RATIO})"
        )
    return warnings


def write_statistics(stats: dict, path: Path) -> None:
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def format_summary(stats: dict) -> str:
    header = f"{'Class':<24}{'train':>7}{'val':>6}{'test':>6}{'total':>7}{'unique':>8}{'minutes':>9}"
    lines = [header, "-" * len(header)]
    for name, c in stats["classes"].items():
        lines.append(
            f"{name:<24}{c['train']:>7}{c['validation']:>6}{c['test']:>6}"
            f"{c['total']:>7}{c['unique_recordings']:>8}{c['duration_s'] / 60:>9.1f}"
        )
    t = stats["totals"]
    lines.append("-" * len(header))
    lines.append(
        f"{'TOTAL':<24}{t['train']:>7}{t['validation']:>6}{t['test']:>6}"
        f"{t['clips']:>7}{t['unique_recordings']:>8}{t['duration_s'] / 60:>9.1f}"
    )
    if stats["rejected_by_reason"]:
        lines.append("")
        lines.append(f"Rejected {t['rejected']} files:")
        lines.extend(f"  {reason}: {n}" for reason, n in stats["rejected_by_reason"].items())
    if stats["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in stats["warnings"])
    return "\n".join(lines)
