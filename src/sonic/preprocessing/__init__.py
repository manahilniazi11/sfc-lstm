"""Audio preprocessing for SonicSentinel AI (SRS Step 4, FR xi-xv).

The same code prepares training data and the audio analysed by the web app,
so the models always see audio processed the same way:

    load -> mono -> resample -> remove DC -> (noise reduction) -> trim silence
         -> faint noise floor -> fixed-length segments (short clips padded with
         faint noise) -> loudness normalization

Settings live in ``config/audio.json``. See ``pipeline.preprocess``.
"""

from .pipeline import Segment, preprocess, preprocess_file
from .settings import AudioSettings, load_settings

__all__ = ["AudioSettings", "Segment", "load_settings", "preprocess", "preprocess_file"]
