"""Model 3: transfer learning from Google's YAMNet (pretrained on AudioSet).

YAMNet is a MobileNet-v1 network trained on 2 million AudioSet clips. Its
input is exactly our preprocessing output (16 kHz mono waveform, ~1 s), and
for each 0.96 s patch it produces a 1024-number *embedding* that describes
the sound. We keep YAMNet frozen and train only a small classifier on the
embeddings:

    waveform -> YAMNet (frozen) -> 1024-d embedding -> Dense 256 -> Dropout -> Dense (classes)

YAMNet's own AudioSet class scores are discarded: every prediction comes
from the classifier we trained on our data (SRS 1.8: no external decision).
Model: https://tfhub.dev/google/yamnet/1 (Apache 2.0), downloaded once.
"""

from __future__ import annotations

import csv
import json
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from ..augmentation.build import DEFAULT_AUGMENTED_DIR
from ..dataset.config import AUDIO_DATA_DIR, load_classes
from ..features.build import DEFAULT_FEATURES_DIR
from ..preprocessing.build import DEFAULT_SEGMENTS_DIR
from .data import cache_key

NAME = "yamnet"
YAMNET_URL = "https://tfhub.dev/google/yamnet/1?tf-hub-format=compressed"
YAMNET_DIR = AUDIO_DATA_DIR / "model_cache" / "yamnet-1"
EMBEDDINGS = DEFAULT_FEATURES_DIR / "yamnet_embeddings.npz"
GRID = [{"hidden": h, "dropout": d, "lr": lr} for h in (256, 512) for d in (0.3, 0.5) for lr in (1e-3, 3e-4)]

_yamnet = None
SMALL_BATCH = 256  # up to this many windows are predicted in one direct call (a 2-minute upload)


def yamnet():
    """The frozen YAMNet SavedModel, downloaded and unpacked on first use."""
    global _yamnet
    if _yamnet is None:
        import tensorflow as tf

        if not (YAMNET_DIR / "saved_model.pb").exists():
            YAMNET_DIR.mkdir(parents=True, exist_ok=True)
            archive = YAMNET_DIR.parent / "yamnet-1.tar.gz"
            print(f"  downloading YAMNet from {YAMNET_URL}")
            urllib.request.urlretrieve(YAMNET_URL, archive)
            with tarfile.open(archive) as tf_archive:
                tf_archive.extractall(YAMNET_DIR, filter="data")
            archive.unlink()
        _yamnet = tf.saved_model.load(str(YAMNET_DIR))
    return _yamnet


_batch_embedder = None


def _embedder():
    """YAMNet takes one waveform per call; tf.map_fn runs the calls for a whole batch inside one
    TensorFlow graph (several in parallel), which gives identical embeddings about twice as fast."""
    global _batch_embedder
    if _batch_embedder is None:
        import tensorflow as tf

        model = yamnet()

        @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
        def batch_embed(waves):
            return tf.map_fn(lambda w: tf.reduce_mean(model(w)[1], axis=0), waves,
                             fn_output_signature=tf.TensorSpec([1024], tf.float32), parallel_iterations=8)

        _batch_embedder = batch_embed
    return _batch_embedder


def embed(waves: np.ndarray) -> np.ndarray:
    """(n, samples) 16 kHz waveforms -> (n, 1024) embeddings (mean over YAMNet's 0.96 s patches)."""
    if len(waves) == 0:
        return np.zeros((0, 1024), dtype=np.float32)
    return _embedder()(np.asarray(waves, dtype=np.float32)).numpy()


def segment_path(split: str, class_name: str, segment_id: str, augmented: bool) -> Path:
    slug = {c.name: c.slug for c in load_classes()}[class_name]
    root = DEFAULT_AUGMENTED_DIR if augmented else DEFAULT_SEGMENTS_DIR
    return root / split / slug / f"{segment_id}.wav"


def segment_embeddings(features_dir: Path = DEFAULT_FEATURES_DIR) -> dict[str, np.ndarray]:
    """Embeddings of every training/validation/test segment, cached by segment ID."""
    key = cache_key()
    if EMBEDDINGS.exists():
        cached = np.load(EMBEDDINGS)
        if "key" in cached and str(cached["key"]) == key:
            return dict(zip(cached["segment_id"], cached["embedding"]))
    with open(features_dir / "features.csv", newline="", encoding="utf-8") as f:
        rows = [r[:6] for r in csv.reader(f)][1:]  # segment_id, audio_id, class, split, is_augmented, source
    print(f"  computing YAMNet embeddings for {len(rows)} segments (once, then cached)")
    ids, vectors = [], np.zeros((len(rows), 1024), dtype=np.float32)
    start = time.perf_counter()
    for i, (segment_id, _audio, class_name, split, augmented, _src) in enumerate(rows):
        wave, _ = sf.read(segment_path(split, class_name, segment_id, augmented == "True"), dtype="float32")
        vectors[i] = embed(wave[np.newaxis])[0]
        ids.append(segment_id)
        if (i + 1) % 1000 == 0:
            rate = (i + 1) / (time.perf_counter() - start)
            print(f"    {i + 1}/{len(rows)} ({rate:.0f}/s, ~{(len(rows) - i - 1) / rate / 60:.0f} min left)")
    np.savez(EMBEDDINGS, segment_id=np.array(ids), embedding=vectors, key=np.array(key))
    return dict(zip(ids, vectors))


class YAMNetModel:
    def __init__(self, head, embeddings: dict[str, np.ndarray] | None = None):
        self.head = head
        self.embeddings = embeddings

    def _inputs(self, batch: dict) -> np.ndarray:
        if "rows" in batch:  # dataset segments: embeddings were computed once and cached
            if self.embeddings is None:
                self.embeddings = segment_embeddings()
            return np.stack([self.embeddings[s] for s in batch["data"].segment_id[batch["rows"]]])
        return embed(batch["wave"])

    def predict_proba(self, batch: dict) -> np.ndarray:
        x = self._inputs(batch)
        if len(x) <= SMALL_BATCH:  # the web app's case: a direct call skips predict()'s set-up cost
            return np.asarray(self.head(x, training=False))
        return self.head.predict(x, batch_size=512, verbose=0)

    def save(self, model_dir: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        self.head.save(model_dir / "head.keras")
        (model_dir / "backbone.json").write_text(json.dumps({"yamnet": YAMNET_URL, "embedding": 1024}), encoding="utf-8")


def load(model_dir: Path) -> YAMNetModel:
    import keras

    return YAMNetModel(keras.models.load_model(model_dir / "head.keras"))


def _build_head(n_classes: int, hidden: int, dropout: float, lr: float):
    import keras

    model = keras.Sequential([
        keras.Input((1024,)),
        keras.layers.Dense(hidden, activation="relu"),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(data, train_rows: np.ndarray, val_rows: np.ndarray):
    import keras
    from sklearn.metrics import f1_score

    keras.utils.set_random_seed(0)
    cache = segment_embeddings()
    E_tr = np.stack([cache[s] for s in data.segment_id[train_rows]])
    E_va = np.stack([cache[s] for s in data.segment_id[val_rows]])
    y_tr, y_va = data.y[train_rows], data.y[val_rows]

    results, best, best_f1 = [], None, -1.0
    print(f"  grid search: {len(GRID)} classifier settings on frozen YAMNet embeddings")
    for params in GRID:
        start = time.perf_counter()
        head = _build_head(len(data.classes), **params)
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        history = head.fit(E_tr, y_tr, validation_data=(E_va, y_va), epochs=100, batch_size=128,
                           callbacks=[stop], verbose=0)
        macro_f1 = float(f1_score(y_va, head.predict(E_va, verbose=0).argmax(1), average="macro"))
        results.append({**params, "epochs": len(history.history["loss"]), "val_segment_macro_f1": round(macro_f1, 4),
                        "seconds": round(time.perf_counter() - start, 1)})
        print(f"    hidden {params['hidden']:>3}  dropout {params['dropout']}  lr {params['lr']:.0e}  "
              f"epochs {results[-1]['epochs']:>3}  val macro F1 {macro_f1:.3f}")
        if macro_f1 > best_f1:
            best, best_f1 = (head, params), macro_f1
    head, params = best
    results.sort(key=lambda r: -r["val_segment_macro_f1"])
    return YAMNetModel(head, cache), params, results
