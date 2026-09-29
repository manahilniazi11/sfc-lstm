"""Stratified, group-aware train/validation/test splitting."""

from __future__ import annotations

import random

SPLITS = ("train", "validation", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


def assign_splits(
    group_sizes: dict[str, int],
    rng: random.Random,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
) -> dict[str, str]:
    """Assign whole groups of ONE class to splits, returning ``{group_id: split}``.

    Running this per class makes the split stratified. Groups are never cut,
    so every clip of an original recording stays in one split. Largest groups
    are placed first (they are hardest to fit), each into the split that is
    furthest below its target share.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    total = sum(group_sizes.values())
    targets = {s: r * total for s, r in zip(SPLITS, ratios)}
    filled = dict.fromkeys(SPLITS, 0)

    order = sorted(group_sizes)
    rng.shuffle(order)
    order.sort(key=lambda g: -group_sizes[g])  # stable: ties keep the shuffled order

    def deficit(split: str) -> float:
        target = targets[split]
        return (target - filled[split]) / target if target else float("-inf")

    assignment = {}
    for group in order:
        split = max(SPLITS, key=deficit)
        assignment[group] = split
        filled[split] += group_sizes[group]
    return assignment
