"""Compare the Python and GTM predictions (SRS Step 11, FR xxxi-xxxiv).

The two models were trained separately and never see each other's output;
this module only looks at their finished scores.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rules import Thresholds

ACCEPTABLE = "Acceptable Match"
WEAK = "Weak Match"
DISAGREEMENT = "Model Disagreement"
UNCERTAIN = "Uncertain Result"


@dataclass
class Scores:
    """One model's confidence for every class it knows."""

    probs: dict[str, float]

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.probs.items(), key=lambda kv: kv[1], reverse=True)

    @property
    def top(self) -> str:
        return self.ranked()[0][0]

    @property
    def confidence(self) -> float:
        return self.ranked()[0][1]

    @property
    def margin(self) -> float:
        """Top-two difference: best minus second-best confidence."""
        ranked = self.ranked()
        return ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)

    def top_n(self, n: int = 3) -> list[tuple[str, float]]:
        return self.ranked()[:n]


@dataclass
class Comparison:
    python_class: str
    python_confidence: float
    python_margin: float
    gtm_class: str | None
    gtm_confidence: float | None
    gtm_margin: float | None
    match: bool
    confidence_difference: float | None  # |Python top confidence - GTM top confidence|
    status: str


def compare(python: Scores, gtm: Scores | None, t: Thresholds) -> Comparison:
    if gtm is None or not gtm.probs:
        return Comparison(python.top, python.confidence, python.margin, None, None, None, False, None, UNCERTAIN)

    match = python.top == gtm.top
    difference = abs(python.confidence - gtm.confidence)
    both_confident = python.confidence >= t.min_confidence and gtm.confidence >= t.min_confidence
    if match:
        status = ACCEPTABLE if both_confident and difference <= t.weak_match_difference else WEAK
    else:
        status = DISAGREEMENT if both_confident else UNCERTAIN
    return Comparison(
        python.top, python.confidence, python.margin,
        gtm.top, gtm.confidence, gtm.margin,
        match, difference, status,
    )
