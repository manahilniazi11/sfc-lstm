"""The selected Python model, loaded once and used for uploads and live windows (SRS Step 8).

`analyse()` runs exactly the steps used when the models were evaluated:
preprocessing in inference mode (every 1 s window, silent windows skipped),
feature extraction, window probabilities calibrated with the model's window
temperature, then the model's clip aggregation and clip temperature.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..features import extract, load_feature_settings
from ..models.evaluate import aggregate, calibrate
from ..models.registry import MODELS_DIR
from ..preprocessing.pipeline import INFERENCE, preprocess
from ..preprocessing.settings import load_settings


@dataclass
class PythonResult:
    version: str
    classes: list[str]
    window_times: list[tuple[float, float]]  # (start, end) seconds in the recording
    window_probs: np.ndarray  # (windows, classes), calibrated
    clip_probs: np.ndarray | None  # (classes,), None when every window was silent
    elapsed_ms: float  # preprocessing + features + prediction (not the one-off model loading)
    preprocessing: dict | None = None  # what was done to the audio, for display

    def clip_scores(self) -> dict[str, float] | None:
        if self.clip_probs is None:
            return None
        return {c: round(float(p), 5) for c, p in zip(self.classes, self.clip_probs)}

    def window_scores(self) -> list[dict[str, float]]:
        return [{c: round(float(p), 5) for c, p in zip(self.classes, row)} for row in self.window_probs]


def selected_version() -> str:
    return json.loads((MODELS_DIR / "selected.json").read_text(encoding="utf-8"))["python_model"]


class PythonClassifier:
    """Loads the model on first use (TensorFlow start-up takes a few seconds) and serialises predictions."""

    def __init__(self, version: str | None = None):
        self._version = version
        self._model = None
        self._meta: dict | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from ..models.compare import load_model  # imports TensorFlow

            version = self._version or selected_version()
            self._model, self._meta = load_model(version)

    @property
    def version(self) -> str:
        with self._lock:
            self._ensure_loaded()
        return self._meta["version"]

    @property
    def classes(self) -> list[str]:
        with self._lock:
            self._ensure_loaded()
        return list(self._meta["classes"])

    def warm_up(self) -> None:
        """Load the model and run one prediction now (e.g. when the server starts).

        TensorFlow builds its computation graphs on the first prediction, which
        takes several seconds; doing it here keeps that out of the first upload.
        """
        with self._lock:
            if self._model is not None:
                return
            self._ensure_loaded()
            settings, fsettings = load_settings(), load_feature_settings()
            noise = (0.05 * np.random.default_rng(0).standard_normal(int(1.5 * settings.sample_rate))).astype(np.float32)
            segments = [s for s in preprocess(noise, settings.sample_rate, settings, INFERENCE) if not s.silent]
            feats = [extract(s.samples, settings.sample_rate, fsettings) for s in segments]
            self._model.predict_proba({
                "X": np.stack([f.vector for f in feats]),
                "logmel": np.stack([f.log_mel for f in feats]).astype(np.float16),
                "wave": np.stack([s.samples for s in segments]).astype(np.float16),
            })

    def analyse(self, audio: np.ndarray, sr: int) -> PythonResult:
        """Audio shaped (frames[, channels]) at any sample rate -> window and clip probabilities."""
        self.warm_up()
        started = time.perf_counter()
        settings, fsettings = load_settings(), load_feature_settings()
        all_segments = preprocess(audio, sr, settings, INFERENCE)
        segments = [s for s in all_segments if not s.silent]
        steps_done = {
            "original_sample_rate": int(sr),
            "sample_rate": settings.sample_rate,
            "channels_in": int(audio.shape[1]) if audio.ndim == 2 else 1,
            "content_start_s": all_segments[0].start_s if all_segments else 0.0,
            "content_end_s": max((s.end_s for s in all_segments), default=0.0),
            "segment_seconds": settings.segment_seconds,
            "hop_seconds": settings.hop_seconds,
            "windows_total": len(all_segments),
            "windows_silent": len(all_segments) - len(segments),
            "windows_padded": sum(s.padded for s in all_segments),
            "target_rms_dbfs": settings.target_rms_dbfs,
            "trim_top_db": settings.trim_top_db,
            "noise_reduction": bool(settings.noise_reduction),
        }
        with self._lock:
            meta, classes = self._meta, list(self._meta["classes"])
            if not segments:
                return PythonResult(meta["version"], classes, [], np.zeros((0, len(classes))), None,
                                    (time.perf_counter() - started) * 1000, steps_done)
            feats = [extract(s.samples, settings.sample_rate, fsettings) for s in segments]
            batch = {
                "X": np.stack([f.vector for f in feats]),
                "logmel": np.stack([f.log_mel for f in feats]).astype(np.float16),
                "wave": np.stack([s.samples for s in segments]).astype(np.float16),
            }
            window = calibrate(self._model.predict_proba(batch), meta["temperature"])
        offsets = np.array([0, len(segments)])
        clip = calibrate(aggregate(window, offsets, meta["aggregation"]), meta["clip_temperature"])[0]
        return PythonResult(
            meta["version"], classes, [(s.start_s, s.end_s) for s in segments], window, clip,
            (time.perf_counter() - started) * 1000, steps_done,
        )


_default: PythonClassifier | None = None
_default_lock = threading.Lock()


def get_classifier() -> PythonClassifier:
    """The process-wide classifier (one model in memory per web server process)."""
    global _default
    with _default_lock:
        if _default is None:
            _default = PythonClassifier()
        return _default
