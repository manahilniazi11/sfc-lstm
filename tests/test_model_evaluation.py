"""Tests for the shared evaluation code: calibration, clip aggregation and metrics."""

from __future__ import annotations

import numpy as np
import pytest

from sonic.models.evaluate import (
    aggregate,
    calibrate,
    compute_metrics,
    expected_calibration_error,
    fit_temperature,
)

CLASSES = ["A", "B", "C"]


def test_temperature_softens_an_overconfident_model_without_changing_predictions():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    correct = rng.random(2000) < 0.7  # right 70 % of the time...
    pred = np.where(correct, y, (y + 1) % 3)
    probs = np.full((2000, 3), 0.005)
    probs[np.arange(2000), pred] = 0.99  # ...but always 99 % confident
    t = fit_temperature(probs, y)
    fixed = calibrate(probs, t)
    assert t > 1
    assert (fixed.argmax(1) == probs.argmax(1)).all()
    assert expected_calibration_error(fixed, y) < expected_calibration_error(probs, y)


def test_aggregation_methods():
    # One clip: two background-ish windows and one clear "C" event.
    windows = np.array([[0.6, 0.3, 0.1], [0.6, 0.3, 0.1], [0.05, 0.05, 0.9]])
    offsets = np.array([0, 3])
    assert aggregate(windows, offsets, "mean").argmax() == 0  # the event is diluted
    assert aggregate(windows, offsets, "max").argmax() == 2   # the event wins
    for method in ("mean", "max", "top3"):
        assert aggregate(windows, offsets, method).sum() == pytest.approx(1.0)


def test_clip_without_windows_gets_uniform_scores():
    out = aggregate(np.zeros((0, 3)), np.array([0, 0]), "mean")
    assert np.allclose(out, 1 / 3)


def test_metrics_and_critical_recall():
    y = np.array([0, 0, 1, 1, 2, 2])
    probs = np.eye(3)[[0, 0, 1, 0, 2, 2]]  # one B predicted as A
    m = compute_metrics(y, probs, CLASSES, critical=["B"])
    assert m["accuracy"] == pytest.approx(5 / 6, abs=1e-3)
    assert m["critical_recall"] == {"B": 0.5}
    assert m["confusion_matrix"][1] == [1, 1, 0]
    assert m["per_class"]["A"]["precision"] == pytest.approx(2 / 3, abs=1e-3)


def _meta(f1, crit, ms=5.0):
    return {"validation": {"clip": {"macro_f1": f1, "critical_recall": crit}}, "inference_ms_per_window": ms}


def test_selection_prefers_models_that_meet_every_critical_target():
    from sonic.models.compare import select

    metas = {"a": _meta(0.83, {"G": 0.90, "S": 0.80}), "b": _meta(0.79, {"G": 0.90, "S": 0.86})}
    assert select(metas)[0] == "b"  # lower F1, but the only one with every critical class >= 85 %


def test_selection_breaks_near_ties_on_critical_recall():
    from sonic.models.compare import select

    metas = {"a": _meta(0.830, {"G": 0.90, "S": 0.80}), "c": _meta(0.825, {"G": 0.95, "S": 0.84}),
             "d": _meta(0.700, {"G": 0.99, "S": 0.84})}
    assert select(metas)[0] == "c"  # within 0.01 of the best F1 and better critical recall; d is too far behind
