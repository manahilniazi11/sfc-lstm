"""Model 4: Convolutional neural network trained from scratch on log-mel spectrograms.

Input: 64 mel bands x 63 frames (1 s), standardized per mel band with
training-set statistics.

    4 blocks of [Conv 3x3 -> BatchNorm -> ReLU] x 2 -> MaxPool 2x2 -> Dropout
        (32, 64, 128, 256 filters; 64x63 -> 32x31 -> 16x15 -> 8x7 -> 4x3)
    global average pool + global max pool (512)
    Dense 128 -> ReLU -> Dropout 0.4 -> Dense (classes, softmax)

Global max pooling keeps short events (a gunshot in 3 frames) from being
averaged away; average pooling suits steady sounds (sirens, machines).

Training tricks, all standard for audio:
* SpecAugment: random time and frequency masks, so the model cannot rely on
  one band or one moment.
* Mixup: training on blends of two clips with blended labels; smooths the
  decision boundaries and resembles overlapping sounds.
* Label smoothing, AdamW with cosine learning-rate decay, early stopping on
  validation macro F1 (best epoch kept).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

NAME = "cnn"
EPOCHS = 40
BATCH = 64
# An epoch takes ~2 minutes on our 4-core CPU (no GPU), so only the main
# variant is trained by default; the others are for a longer (overnight) run.
VARIANTS = [
    {"width": 32, "dropout": 0.1, "mixup": 0.2},
]
EXTRA_VARIANTS = [
    {"width": 32, "dropout": 0.2, "mixup": 0.0},
    {"width": 24, "dropout": 0.1, "mixup": 0.2},
]


class CNNModel:
    def __init__(self, model, mean: np.ndarray, std: np.ndarray):
        self.model = model
        self.mean, self.std = mean, std

    def _inputs(self, logmel: np.ndarray) -> np.ndarray:
        x = (np.asarray(logmel, dtype=np.float32) - self.mean) / self.std
        return x[..., np.newaxis]

    def predict_proba(self, batch: dict) -> np.ndarray:
        x = self._inputs(batch["logmel"])
        if len(x) <= 256:  # the web app's case: a direct call skips predict()'s set-up cost
            return np.asarray(self.model(x, training=False))
        return self.model.predict(x, batch_size=256, verbose=0)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(model_dir / "model.keras")
        np.savez(model_dir / "normalization.npz", mean=self.mean, std=self.std)


def load(model_dir: Path) -> CNNModel:
    import keras

    norm = np.load(model_dir / "normalization.npz")
    return CNNModel(keras.models.load_model(model_dir / "model.keras"), norm["mean"], norm["std"])


def build(n_classes: int, width: int = 32, dropout: float = 0.1, input_shape=(64, 63, 1)):
    import keras
    from keras import layers

    inputs = keras.Input(input_shape)
    x = inputs
    for i in range(4):
        filters = width * 2**i
        for _ in range(2):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling2D()(x), layers.GlobalMaxPooling2D()(x)])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="sonic_cnn")


def spec_augment(x: np.ndarray, rng: np.random.Generator, masks: int = 2, max_width: int = 8) -> np.ndarray:
    """Zero (= the band mean after standardization) random time and frequency stripes."""
    x = x.copy()
    n, bands, frames = x.shape[:3]
    for i in range(n):
        for _ in range(masks):
            w = int(rng.integers(0, max_width + 1))
            f0 = int(rng.integers(0, bands - w + 1))
            x[i, f0 : f0 + w] = 0
            w = int(rng.integers(0, max_width + 1))
            t0 = int(rng.integers(0, frames - w + 1))
            x[i, :, t0 : t0 + w] = 0
    return x


def _batches(X: np.ndarray, y: np.ndarray, n_classes: int, mixup: float, rng: np.random.Generator):
    """One epoch of augmented (SpecAugment + optional mixup) batches with soft labels."""
    order = rng.permutation(len(X))
    for start in range(0, len(X), BATCH):
        idx = order[start : start + BATCH]
        xb = spec_augment(X[idx], rng)
        yb = np.eye(n_classes, dtype=np.float32)[y[idx]]
        if mixup > 0:
            lam = rng.beta(mixup, mixup, size=(len(idx), 1)).astype(np.float32)
            perm = rng.permutation(len(idx))
            xb = lam[..., None, None] * xb + (1 - lam[..., None, None]) * xb[perm]
            yb = lam * yb + (1 - lam) * yb[perm]
        yield xb, yb


def _fit(variant: dict, X_tr, y_tr, X_va, y_va, n_classes: int, seed: int = 0):
    import keras
    from sklearn.metrics import f1_score

    keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    model = build(n_classes, variant["width"], variant["dropout"], X_tr.shape[1:])
    steps = int(np.ceil(len(X_tr) / BATCH)) * EPOCHS
    schedule = keras.optimizers.schedules.CosineDecay(1e-3, decay_steps=steps, alpha=0.05)
    model.compile(optimizer=keras.optimizers.AdamW(schedule, weight_decay=1e-4),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1))
    best_f1, best_weights, history, patience = -1.0, None, [], 0
    for epoch in range(EPOCHS):
        start = time.perf_counter()
        losses = [float(model.train_on_batch(xb, yb)) for xb, yb in _batches(X_tr, y_tr, n_classes, variant["mixup"], rng)]
        f1 = float(f1_score(y_va, model.predict(X_va, batch_size=256, verbose=0).argmax(1), average="macro"))
        history.append({"epoch": epoch + 1, "loss": round(float(np.mean(losses)), 4), "val_macro_f1": round(f1, 4),
                        "seconds": round(time.perf_counter() - start, 1)})
        print(f"      epoch {epoch + 1:>2}/{EPOCHS}  loss {history[-1]['loss']:.3f}  val macro F1 {f1:.3f}"
              f"  ({history[-1]['seconds']:.0f} s)", flush=True)
        if f1 > best_f1:
            best_f1, best_weights, patience = f1, model.get_weights(), 0
        else:
            patience += 1
            if patience >= 8:
                break
    model.set_weights(best_weights)
    return model, best_f1, history


def train(data, train_rows: np.ndarray, val_rows: np.ndarray, variants: list[dict] = VARIANTS):
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(4)
    X_tr = np.asarray(data.logmel[train_rows], dtype=np.float32)
    X_va = np.asarray(data.logmel[val_rows], dtype=np.float32)
    mean = X_tr.mean(axis=(0, 2), keepdims=True)[0]  # per mel band: (64, 1)
    std = X_tr.std(axis=(0, 2), keepdims=True)[0] + 1e-6
    X_tr = ((X_tr - mean) / std)[..., np.newaxis]
    X_va = ((X_va - mean) / std)[..., np.newaxis]
    y_tr, y_va = data.y[train_rows], data.y[val_rows]

    results, best = [], None
    for i, variant in enumerate(variants, start=1):
        print(f"  variant {i}/{len(variants)}: {variant}", flush=True)
        model, f1, history = _fit(variant, X_tr, y_tr, X_va, y_va, len(data.classes))
        results.append({**variant, "val_segment_macro_f1": round(f1, 4), "history": history})
        print(f"    best val macro F1 {f1:.3f}", flush=True)
        if best is None or f1 > best[1]:
            best = (model, f1, variant)
    model, _, variant = best
    params = {**variant, "epochs_max": EPOCHS, "batch": BATCH, "parameters": int(model.count_params())}
    return CNNModel(model, mean, std), params, results
