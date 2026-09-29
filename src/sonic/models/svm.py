"""Model 1: Support Vector Machine (RBF kernel) on the 123 summary features.

Features are standardized (mean 0, variance 1, fitted on training data only)
because the SVM measures distances, and raw features range from ~0.01
(flatness) to thousands (centroid in Hz). C (how hard the SVM tries to fit
every training point) and gamma (how local the RBF kernel is) are tuned on
the validation set.
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

NAME = "svm"
GRID = [{"C": c, "gamma": g} for c in (1, 3, 10, 30, 100) for g in ("scale", 0.003, 0.01, 0.03)]


class SVMModel:
    """Standardizer + SVM whose scores are turned into probabilities (Platt sigmoid, 5-fold)."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def predict_proba(self, batch: dict) -> np.ndarray:
        return self.pipeline.predict_proba(batch["X"])

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, model_dir / "model.joblib")


def load(model_dir: Path) -> SVMModel:
    return SVMModel(joblib.load(model_dir / "model.joblib"))


def _pipeline(C: float, gamma, probability: bool) -> Pipeline:
    svc = SVC(C=C, gamma=gamma, random_state=0, cache_size=1000)
    if probability:
        # One SVM on all training data; a sigmoid fitted by 5-fold CV maps its
        # decision scores to probabilities (scikit-learn's replacement for
        # the deprecated SVC(probability=True)).
        svc = CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False)
    return make_pipeline(StandardScaler(), svc)


def _score(params: dict, X_tr, y_tr, X_va, y_va) -> dict:
    start = time.perf_counter()
    model = _pipeline(params["C"], params["gamma"], probability=False).fit(X_tr, y_tr)
    macro_f1 = f1_score(y_va, model.predict(X_va), average="macro")
    return {**params, "val_segment_macro_f1": round(float(macro_f1), 4), "seconds": round(time.perf_counter() - start, 1)}


def train(data, train_rows: np.ndarray, val_rows: np.ndarray, workers: int = 4):
    """Grid search on validation macro-F1, then refit the best with probability outputs."""
    X_tr, y_tr, X_va, y_va = data.X[train_rows], data.y[train_rows], data.X[val_rows], data.y[val_rows]
    print(f"  grid search: {len(GRID)} settings on {len(train_rows)} training segments ({workers} in parallel)")
    results = Parallel(n_jobs=workers, verbose=0)(delayed(_score)(p, X_tr, y_tr, X_va, y_va) for p in GRID)
    results.sort(key=lambda r: -r["val_segment_macro_f1"])
    for r in results[:5]:
        print(f"    C={r['C']:<5} gamma={r['gamma']!s:<6} val macro F1 {r['val_segment_macro_f1']:.3f}  ({r['seconds']} s)")
    best = {"C": results[0]["C"], "gamma": results[0]["gamma"]}
    print(f"  refitting best {best} with probability outputs")
    model = SVMModel(_pipeline(best["C"], best["gamma"], probability=True).fit(X_tr, y_tr))
    return model, best, results
