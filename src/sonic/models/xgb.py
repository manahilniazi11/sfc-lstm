"""Model 2: XGBoost (gradient-boosted decision trees) on the 123 summary features.

Trees split on one feature at a time, so no scaling is needed. Each boosting
round adds one small tree per class that corrects the previous rounds'
mistakes. Early stopping on validation loss picks the number of rounds; a
random search over tree depth, learning rate, subsampling and regularization
picks the rest. Gain-based feature importance is saved for the report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from ..features import feature_names, load_feature_settings

NAME = "xgboost"
# Narrowed after a first search: on our 4-core CPU a trial with learning rate
# 0.05 and depth 9 took ~4 minutes for no better score than lr 0.12 / depth 7.
TRIALS = 12
MAX_ROUNDS = 1000
EARLY_STOP = 30


class XGBModel:
    def __init__(self, model: XGBClassifier):
        self.model = model

    def predict_proba(self, batch: dict) -> np.ndarray:
        return self.model.predict_proba(batch["X"])

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_dir / "model.json")
        names = feature_names(load_feature_settings())
        gain = self.model.get_booster().get_score(importance_type="gain")
        ranked = sorted(((names[int(k[1:])], round(v, 3)) for k, v in gain.items()), key=lambda kv: -kv[1])
        (model_dir / "feature_importance.json").write_text(json.dumps(dict(ranked), indent=1), encoding="utf-8")


def load(model_dir: Path) -> XGBModel:
    model = XGBClassifier()
    model.load_model(model_dir / "model.json")
    return XGBModel(model)


def sample_params(rng: np.random.Generator) -> dict:
    return {
        "max_depth": int(rng.integers(4, 9)),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.08), np.log(0.3)))),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "min_child_weight": float(rng.uniform(1, 10)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(0.5), np.log(10)))),
        "reg_alpha": float(rng.uniform(0, 1)),
    }


def _fit(params: dict, X_tr, y_tr, X_va, y_va) -> XGBClassifier:
    model = XGBClassifier(
        **params,
        n_estimators=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOP,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
        random_state=0,
    )
    return model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)


def train(data, train_rows: np.ndarray, val_rows: np.ndarray, trials: int = TRIALS, seed: int = 0):
    X_tr, y_tr, X_va, y_va = data.X[train_rows], data.y[train_rows], data.X[val_rows], data.y[val_rows]
    rng = np.random.default_rng(seed)
    results, best, best_f1 = [], None, -1.0
    print(f"  random search: {trials} settings, early stopping after {EARLY_STOP} rounds without improvement")
    for i in range(trials):
        params = sample_params(rng)
        start = time.perf_counter()
        model = _fit(params, X_tr, y_tr, X_va, y_va)
        macro_f1 = float(f1_score(y_va, model.predict(X_va), average="macro"))
        results.append({**{k: round(v, 4) for k, v in params.items()}, "rounds": int(model.best_iteration) + 1,
                        "val_segment_macro_f1": round(macro_f1, 4), "seconds": round(time.perf_counter() - start, 1)})
        if macro_f1 > best_f1:
            best, best_f1 = (model, params), macro_f1
        print(f"    {i + 1:>2}/{trials}  depth {params['max_depth']:>2}  lr {params['learning_rate']:.3f}  "
              f"rounds {results[-1]['rounds']:>4}  val macro F1 {macro_f1:.3f}  (best {best_f1:.3f})")
    model, params = best
    results.sort(key=lambda r: -r["val_segment_macro_f1"])
    return XGBModel(model), {**{k: round(v, 4) for k, v in params.items()}, "rounds": int(model.best_iteration) + 1}, results
