"""Alert rules and decision thresholds, loaded from alert_rules/alert_rules.json or the database."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from ..dataset.config import REPO_ROOT

RULES_FILE = REPO_ROOT / "alert_rules" / "alert_rules.json"

SEVERITIES = ["Informational", "Low", "Medium", "High", "Critical"]
QUALITY_GRADES = ["Good", "Acceptable", "Poor", "Unusable"]
UNKNOWN = "Unknown"
BACKGROUND = "Background Noise"
# Classes that describe "nothing happening": never an overlapping *event*.
NON_EVENTS = {"Background Noise", "Normal Machinery"}


@dataclass
class Thresholds:
    min_confidence: float = 0.60
    top_two_margin: float = 0.15
    weak_match_difference: float = 0.30
    overlap_threshold: float = 0.25
    unknown_confidence: float = 0.35
    repeat_window_seconds: float = 10


@dataclass
class Rule:
    category: str
    severity: str = "Informational"
    critical: bool = False
    alert: bool = False
    alert_type: str = "environment"
    min_confidence: float = 0.6
    top_two_margin: float = 0.15
    consecutive_detections: int = 1
    require_model_agreement: bool = False
    min_quality: str = "Poor"  # worst quality grade that may still raise an alert
    recommended_action: str = ""
    manual_review: str = ""
    escalate_after_minutes: float | None = None
    raise_to_critical: dict | None = None  # {"min_confidence": x, "consecutive_detections": n}
    noise_limit_dbfs: float | None = None  # Background Noise only
    above_noise_limit: dict | None = None
    enabled: bool = True

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f"{self.category}: unknown severity {self.severity!r}")
        if self.min_quality not in QUALITY_GRADES:
            raise ValueError(f"{self.category}: unknown quality grade {self.min_quality!r}")
        if not 0 <= self.min_confidence <= 1 or not 0 <= self.top_two_margin <= 1:
            raise ValueError(f"{self.category}: confidences must be between 0 and 1")
        if self.consecutive_detections < 1:
            raise ValueError(f"{self.category}: consecutive_detections must be at least 1")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RuleSet:
    thresholds: Thresholds
    rules: dict[str, Rule] = field(default_factory=dict)

    def for_category(self, category: str) -> Rule:
        """The rule for a class; a class without an (enabled) rule is informational only."""
        rule = self.rules.get(category)
        if rule is None or not rule.enabled:
            return Rule(category=category)
        return rule


def _known(cls, data: dict) -> dict:
    """Keep the dataclass's fields; keys starting with "_" are comments in the JSON."""
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def rule_from_dict(data: dict) -> Rule:
    return Rule(**_known(Rule, data))


def load_rules(path: Path = RULES_FILE) -> RuleSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = [rule_from_dict(r) for r in data["rules"]]
    duplicates = {r.category for r in rules if sum(o.category == r.category for o in rules) > 1}
    if duplicates:
        raise ValueError(f"more than one rule for {sorted(duplicates)}")
    return RuleSet(Thresholds(**_known(Thresholds, data.get("settings", {}))), {r.category: r for r in rules})


def dump_rules(ruleset: RuleSet) -> dict:
    """The JSON structure of alert_rules.json (used for the admin's export)."""
    return {"settings": asdict(ruleset.thresholds), "rules": [r.to_dict() for r in ruleset.rules.values()]}
