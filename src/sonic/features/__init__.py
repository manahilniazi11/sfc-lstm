"""Acoustic feature extraction for SonicSentinel AI (SRS Step 6, FR xx).

Shared by training and the web app. Settings: ``config/features.json``.
"""

from .extract import Features, extract, feature_names
from .settings import FeatureSettings, load_feature_settings

__all__ = ["FeatureSettings", "Features", "extract", "feature_names", "load_feature_settings"]
