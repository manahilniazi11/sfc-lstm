"""Model comparison, final decision and alert rules (alert-rule, comparison, low-confidence and overlap tests)."""

import json

import pytest

from sonic.detection import compare as C
from sonic.detection import decision as D
from sonic.detection.compare import Scores
from sonic.detection.rules import RULES_FILE, Rule, Thresholds, load_rules

CLASSES = ["Gunshot", "Glass Breaking", "Aggression", "Panic Scream", "Background Noise", "Machinery Fault", "Vehicle Horn"]


def scores(**probs):
    """Scores with the given probabilities; the remaining mass is spread over the other classes."""
    named = {k.replace("_", " "): v for k, v in probs.items()}
    rest = [c for c in CLASSES if c not in named]
    left = max(0.0, 1 - sum(named.values()))
    return Scores({**named, **{c: left / len(rest) for c in rest}})


@pytest.fixture(scope="module")
def rules():
    return load_rules()


def test_rule_file_loads_and_covers_every_mandatory_class(rules):
    mandatory = {"Machinery Fault", "Glass Breaking", "Alarm or Siren", "Vehicle Horn", "Animal Sound",
                 "Gunshot", "Panic Scream", "Aggression", "Person Asking for Help", "Background Noise"}
    assert mandatory <= set(rules.rules)
    for name in ["Gunshot", "Glass Breaking", "Panic Scream", "Aggression", "Person Asking for Help"]:
        assert rules.rules[name].critical and rules.rules[name].alert


def test_invalid_rules_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        Rule(category="X", severity="Catastrophic")
    with pytest.raises(ValueError):
        Rule(category="X", min_confidence=1.5)
    with pytest.raises(ValueError):
        Rule(category="X", consecutive_detections=0)
    bad = tmp_path / "rules.json"
    bad.write_text(json.dumps({"rules": [{"category": "Gunshot"}, {"category": "Gunshot"}]}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_rules(bad)


# --- comparison (SRS Step 11) ---

def test_confidence_difference_is_absolute_top_class_difference():
    cmp = C.compare(scores(Gunshot=0.9), scores(Gunshot=0.7), Thresholds())
    assert cmp.match and cmp.confidence_difference == pytest.approx(0.2)
    assert cmp.status == C.ACCEPTABLE


@pytest.mark.parametrize("py,gtm,status", [
    (scores(Gunshot=0.90), scores(Gunshot=0.85), C.ACCEPTABLE),
    (scores(Gunshot=0.95), scores(Gunshot=0.40), C.WEAK),  # same class, GTM unsure
    (scores(Gunshot=0.90), scores(Glass_Breaking=0.80), C.DISAGREEMENT),  # both sure, different classes
    (scores(Gunshot=0.90), scores(Glass_Breaking=0.30), C.UNCERTAIN),  # different classes, GTM unsure
])
def test_consistency_status(py, gtm, status):
    assert C.compare(py, gtm, Thresholds()).status == status


def test_missing_gtm_result_is_uncertain():
    cmp = C.compare(scores(Gunshot=0.9), None, Thresholds())
    assert cmp.status == C.UNCERTAIN and cmp.confidence_difference is None


def test_top_two_margin():
    s = Scores({"Gunshot": 0.5, "Glass Breaking": 0.3, "Aggression": 0.2})
    assert s.margin == pytest.approx(0.2) and s.top_n(2) == [("Gunshot", 0.5), ("Glass Breaking", 0.3)]


# --- final decision and alerts (SRS Steps 15-19) ---

def test_confident_repeated_gunshot_raises_a_critical_alert(rules):
    shot = scores(Gunshot=0.92)
    d = D.decide(shot, scores(Gunshot=0.8), "Good", rules, windows=[shot, shot, scores(Background_Noise=0.9)])
    assert d.final_class == "Gunshot" and d.alert and d.severity == "Critical"
    assert d.status == D.ALERT_GENERATED and d.confidence_level == "High"
    assert not d.review_required


def test_gunshot_in_only_one_window_is_not_confirmed(rules):
    shot, quiet = scores(Gunshot=0.92), scores(Background_Noise=0.9)
    d = D.decide(shot, scores(Gunshot=0.8), "Good", rules, windows=[quiet, shot, quiet])
    assert d.alert_status == D.NOT_CONFIRMED  # the rule needs 2 consecutive windows
    assert D.R_FALSE_ALARM in d.review_reasons and d.status == D.MANUAL_REVIEW


def test_a_one_window_clip_can_still_confirm(rules):
    shot = scores(Gunshot=0.92)
    d = D.decide(shot, scores(Gunshot=0.8), "Good", rules, windows=[shot])
    assert d.alert


def test_live_repeated_detections_count(rules):
    shot = scores(Gunshot=0.92)
    assert not D.decide(shot, scores(Gunshot=0.8), "Good", rules, windows=[shot, shot], repeated_detections=1).alert
    assert D.decide(shot, scores(Gunshot=0.8), "Good", rules, windows=[shot, shot], repeated_detections=2).alert


def test_live_short_sound_in_one_window_is_not_confirmed(rules):
    """A gunshot fills one model window of a 2 s live window; only a repeat in a later window confirms it."""
    shot = scores(Gunshot=0.92)
    d = D.decide(shot, scores(Gunshot=0.8), "Good", rules, windows=[shot], repeated_detections=1)
    assert d.alert_status == D.NOT_CONFIRMED and D.R_FALSE_ALARM in d.review_reasons


def test_low_confidence_critical_event_goes_to_review_without_alert(rules):
    t = rules.thresholds
    low = (t.unknown_confidence + t.min_confidence) / 2  # a class, but below the confidence threshold
    d = D.decide(scores(Gunshot=low), scores(Gunshot=0.6), "Good", rules)
    assert not d.alert and d.uncertain and d.status == D.UNCERTAIN_STATUS
    assert D.R_LOW_CONF in d.review_reasons and D.R_FALSE_ALARM in d.review_reasons
    assert d.confidence_level == "Low"


def test_critical_event_without_model_agreement(rules):
    glass = scores(Glass_Breaking=0.9)
    d = D.decide(glass, scores(Gunshot=0.7), "Good", rules)
    assert d.alert  # the rule does not require agreement: a real break-in must not be missed
    assert D.R_CRITICAL_NO_AGREEMENT in d.review_reasons and D.R_DISAGREE in d.review_reasons
    assert d.confidence_level == "Medium"  # disagreement never gives High


def test_agreement_requirement_blocks_the_alert(rules):
    rules.rules["Glass Breaking"].require_model_agreement = True
    try:
        glass = scores(Glass_Breaking=0.9)
        d = D.decide(glass, scores(Gunshot=0.7), "Good", rules)
        assert d.alert_status == D.NOT_CONFIRMED and d.conditions["both models agree"] is False
    finally:
        rules.rules["Glass Breaking"].require_model_agreement = False


def test_very_low_confidence_is_unknown(rules):
    d = D.decide(Scores({c: 1 / len(CLASSES) for c in CLASSES}), scores(Gunshot=0.5), "Good", rules)
    assert d.final_class == "Unknown" and D.R_UNSUPPORTED in d.review_reasons and not d.alert


def test_unusable_audio_is_unknown(rules):
    d = D.decide(scores(Gunshot=0.95), scores(Gunshot=0.9), "Unusable", rules)
    assert d.final_class == "Unknown" and not d.alert


def test_machinery_fault_needs_acceptable_quality(rules):
    fault = scores(Machinery_Fault=0.9)
    good = D.decide(fault, scores(Machinery_Fault=0.8), "Good", rules, windows=[fault, fault])
    poor = D.decide(fault, scores(Machinery_Fault=0.8), "Poor", rules, windows=[fault, fault])
    assert good.alert and not poor.alert and D.R_POOR_QUALITY in poor.review_reasons


def test_overlapping_sounds_are_detected(rules):
    both = Scores({"Gunshot": 0.45, "Glass Breaking": 0.4, "Background Noise": 0.15})
    d = D.decide(both, scores(Gunshot=0.6), "Good", rules)
    assert d.overlapping_classes == ["Glass Breaking", "Gunshot"] and D.R_OVERLAP in d.review_reasons


def test_background_noise_is_not_an_overlapping_event(rules):
    s = Scores({"Gunshot": 0.5, "Background Noise": 0.45, "Glass Breaking": 0.05})
    assert D.overlapping_classes([s], 0.25) == []


def test_loud_background_noise_raises_its_severity(rules):
    noise = scores(Background_Noise=0.9)
    quiet = D.decide(noise, scores(Background_Noise=0.8), "Good", rules, noise_level_dbfs=-35)
    loud = D.decide(noise, scores(Background_Noise=0.8), "Good", rules, noise_level_dbfs=-12)
    assert quiet.severity == "Informational" and not quiet.alert
    assert loud.severity == "Medium" and loud.alert


def test_repeated_confident_aggression_becomes_critical(rules):
    a = scores(Aggression=0.9)
    once = D.decide(a, scores(Aggression=0.8), "Good", rules, windows=[a])
    three = D.decide(a, scores(Aggression=0.8), "Good", rules, windows=[a, a, a])
    assert once.severity == "High" and three.severity == "Critical"


def test_vehicle_horn_is_stored_without_alert(rules):
    horn = scores(Vehicle_Horn=0.9)
    d = D.decide(horn, scores(Vehicle_Horn=0.85), "Good", rules)
    assert d.status == D.CLASSIFIED and not d.alert and d.severity == "Low"


def test_missing_gtm_routes_to_review(rules):
    d = D.decide(scores(Vehicle_Horn=0.9), None, "Good", rules)
    assert D.R_NO_GTM in d.review_reasons and d.status == D.UNCERTAIN_STATUS


def test_rules_file_path_is_in_the_repository():
    assert RULES_FILE.parent.name == "alert_rules" and RULES_FILE.exists()
