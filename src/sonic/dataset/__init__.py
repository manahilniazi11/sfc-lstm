"""Dataset builder for SonicSentinel AI.

Collects labelled clips from public datasets and the team's own recordings,
maps them onto our sound classes, removes duplicates, assigns Audio IDs and
creates a leakage-free, stratified train/validation/test split.

Run ``python -m sonic.dataset --help`` for the command-line interface.
"""
