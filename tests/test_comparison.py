"""The model comparison report: clip selection, GTM results, major disagreements and the summary."""

import csv
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from sonic.comparison import report, runner
from sonic.detection import decision as D
from sonic.detection.compare import Scores
from sonic.detection.rules import load_rules

RULES = load_rules()
CLASSES = ["Gunshot", "Glass Breaking", "Vehicle Horn", "Background Noise"]


def probs(top, p):
    rest = [c for c in CLASSES if c != top]
    return {top: p, **{c: (1 - p) / len(rest) for c in rest}}


def row(actual, python, p_conf, gtm, g_conf, quality="Good", windows=None):
    r = report.Row("X-1", "x.wav", actual, 2.0, quality=quality)
    r.python, r.gtm = probs(python, p_conf), probs(gtm, g_conf) if gtm else None
    r.python_windows = windows or [python]
    r.decision = D.decide(Scores(r.python), Scores(r.gtm) if r.gtm else None, quality, RULES,
                          windows=[Scores(r.python)] * len(r.python_windows))
    r.explanation = report.explain(r, RULES) if report.is_major(r, RULES) else ""
    return r


def test_per_class_selection_is_balanced_and_stable():
    clips = runner.test_clips(per_class=10)
    counts = {}
    for c in clips:
        counts[c["class_name"]] = counts.get(c["class_name"], 0) + 1
    assert set(counts.values()) == {10} and len(counts) >= 10  # at least ten from every class (SRS)
    assert clips == runner.test_clips(per_class=10)
    assert all(c["split"] == "test" for c in runner.test_clips())


def test_gtm_errors_and_bad_results_count_as_no_result():
    assert report._gtm_scores({"error": "every window is silent"}) == (None, "every window is silent")
    assert report._gtm_scores({"labels": ["A", "B"], "clip": [1.0]})[0] is None
    assert report._gtm_scores({"labels": ["A", "B"], "clip": [0.3, 0.7]}) == ({"A": 0.3, "B": 0.7}, "")


def test_major_disagreement_rules():
    assert not report.is_major(row("Gunshot", "Gunshot", 0.9, "Gunshot", 0.8), RULES)  # agreement
    assert report.is_major(row("Vehicle Horn", "Vehicle Horn", 0.4, "Gunshot", 0.4), RULES)  # critical class involved
    assert report.is_major(row("Vehicle Horn", "Vehicle Horn", 0.7, "Background Noise", 0.3), RULES)  # a confident model
    assert not report.is_major(row("Vehicle Horn", "Vehicle Horn", 0.5, "Background Noise", 0.4), RULES)


def test_explanation_uses_the_recordings_facts():
    r = row("Gunshot", "Gunshot", 0.91, "Glass Breaking", 0.45, quality="Poor",
            windows=["Gunshot", "Gunshot", "Background Noise"])
    r.quality_issues = ["clipping"]
    text = report.explain(r, RULES)
    assert "Python heard Gunshot (91%), GTM heard Glass Breaking (45%)" in text and "Python was right" in text
    assert "short, broadband impulses" in text  # the pair's reason
    assert "Gunshot in 2, Background Noise in 1" in text and "Poor (clipping)" in text and "GTM was unsure" in text
    neither = row("Vehicle Horn", "Gunshot", 0.8, "Glass Breaking", 0.7)
    assert "neither model was right" in neither.explanation


def test_summary_counts():
    rows = [row("Gunshot", "Gunshot", 0.9, "Gunshot", 0.8),
            row("Gunshot", "Gunshot", 0.9, "Glass Breaking", 0.8),
            row("Glass Breaking", "Gunshot", 0.6, "Glass Breaking", 0.7),
            row("Vehicle Horn", "Vehicle Horn", 0.8, None, 0)]
    s = report.summarise(rows, RULES)
    assert s["python"]["accuracy"] == 0.75 and s["gtm"]["accuracy"] == 0.5
    assert (s["compared"], s["agree"], s["disagree"]) == (3, 1, 2)
    assert s["disagree_python_right"] == 1 and s["disagree_gtm_right"] == 1
    assert s["python"]["per_class"]["Gunshot"][3] == 2
    lines = "\n".join(report.summary_sentences(s))
    assert "Python model** is clearly the stronger" in lines


def test_csv_has_every_class_for_both_models(tmp_path):
    rows = [row("Gunshot", "Gunshot", 0.9, "Glass Breaking", 0.8)]
    path = tmp_path / "c.csv"
    report.write_csv(rows, path)
    header, values = list(csv.reader(open(path, encoding="utf-8-sig")))
    record = dict(zip(header, values))
    assert all(f"Python: {c}" in header and f"GTM: {c}" in header for c in CLASSES)
    assert record["Class match"] == "No" and record["Correct"] == "Yes" and record["Explanation of major disagreement"]


def test_serve_saves_posted_results(tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: runner._Handler(*a, directory=str(tmp_path), **k))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        body = json.dumps({"version": "v", "results": [{"audio_id": "A"}]}).encode()
        urllib.request.urlopen(urllib.request.Request(url + runner.RESULTS, data=body, method="POST"))
        assert json.loads((tmp_path / runner.RESULTS).read_text())["results"] == [{"audio_id": "A"}]
        with pytest.raises(urllib.error.HTTPError):  # nothing else can be written
            urllib.request.urlopen(urllib.request.Request(url + "other.json", data=body, method="POST"))
    finally:
        server.shutdown()
