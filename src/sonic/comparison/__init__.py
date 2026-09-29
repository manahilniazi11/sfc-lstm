"""Model prediction and confidence comparison report (SRS deliverable 6).

Both models classify the same unseen test recordings independently, and
each recording then goes through the app's own decision (quality, rules,
severity, alert, manual review):

1. ``python -m sonic.comparison prepare`` copies the test clips and a GTM
   page into a run folder.
2. ``python -m sonic.comparison serve`` serves that folder; open the page in
   Chrome and press Run. GTM runs in the browser, exactly as in the web app
   (same ``static/js/gtm.js``), and the results are saved to
   ``gtm_results.json``. The page never sees the Python results.
3. ``python -m sonic.comparison report`` runs the Python model on the same
   files, applies the decision and writes ``reports/model_comparison.md``
   and ``reports/model_comparison.csv``.
"""
