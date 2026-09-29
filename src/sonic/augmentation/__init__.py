"""Training-data augmentation for SonicSentinel AI (SRS Hint, FR xix).

Each augmented copy of a training segment runs a random chain of 2-3
transforms from ``transforms.py``; background noise is always one of them,
drawn from a shared pool (Background Noise and Normal Machinery training
segments). This breaks the link between a background and a
class and balances the classes. Settings: ``config/augmentation.json``.
"""
