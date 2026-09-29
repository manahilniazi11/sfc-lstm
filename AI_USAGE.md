# AI Tool Usage Declaration

Required by the SRS (§1.8 item 10, §1.10 item 15). Add an entry for every
use of an AI tool. The final sound classification is produced only by our
trained Python model and our Google Teachable Machine model, never by a
generative-AI API.

## Entry 1: Dataset builder

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5) |
| Purpose | Summarise the SRS; suggest public datasets per class; write the dataset-builder module |
| Assistance requested | Download (resumable, selective extraction from remote zips via HTTP Range), Freesound API fetcher, label mapping, de-duplication, group-aware stratified split, metadata and statistics, with tests |
| Files affected | `src/sonic/dataset/**`, `config/sound_classes.json`, `config/dataset_label_map.json`, `config/freesound_queries.json`, `tests/test_dataset_builder.py`, `tests/test_download.py`, `tests/test_remote_zip.py`, `.env.example`, `pyproject.toml`, `documentation/dataset.md`, `documentation/data_dictionary.md`, `data/raw/custom/recordings.csv`, `.gitignore` |
| Student modifications | _TODO: team fills in_ |
| Testing completed | 27 automated tests (`uv run pytest`) on synthetic audio and local HTTP servers; selective download verified against the real FSD50K archives (CRC-checked); _TODO: team adds results of the first real build_ |
| Verifying team members | _TODO: names of members who reviewed and can explain this module_ |

## Entry 2: Audio preprocessing and shortcut check

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5) |
| Purpose | Write the preprocessing pipeline shared by training and the web app; measure and reduce dataset-format bias |
| Assistance requested | Mono/resample/trim/segment/normalize steps, event-window selection, noise floor, segment export, shortcut check, with tests |
| Files affected | `src/sonic/preprocessing/**`, `config/audio.json`, `tests/test_preprocessing.py`, `documentation/preprocessing.md`, `pyproject.toml` |
| Student modifications | _TODO: team fills in_ |
| Testing completed | 19 preprocessing tests (46 total); shortcut check on the real dataset (results in `documentation/preprocessing.md`) |
| Verifying team members | _TODO_ |

## Entry 3: Augmentation

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5) |
| Purpose | Balance classes and break the background-to-class link with offline augmentation of training segments |
| Assistance requested | Noise/shift/pitch/stretch/gain/reverb/distance/device transforms, transform chain, noise pool, copy planning, metadata, with tests |
| Files affected | `src/sonic/augmentation/**`, `config/augmentation.json`, `tests/test_augmentation.py`, `documentation/augmentation.md`, `src/sonic/dataset/download.py` (mimii-valve-normal item) |
| Student modifications | _TODO: team fills in_ |
| Testing completed | 10 augmentation tests (56 total); build on the real segments (results in `documentation/augmentation.md`) |
| Verifying team members | _TODO_ |

## Entry 4: Feature extraction

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5) |
| Purpose | Extract the SRS acoustic features for classical models and log-mel spectrograms for the CNN |
| Assistance requested | Feature layout, single-STFT extraction, parallel build, tests that each feature measures what it claims |
| Files affected | `src/sonic/features/**`, `config/features.json`, `tests/test_features.py`, `documentation/features.md` |
| Student modifications | _TODO: team fills in_ |
| Testing completed | 8 feature tests (64 total); build on all 15,957 segments with no non-finite values; quick Random Forest baseline |
| Verifying team members | _TODO_ |

## Entry 5: Audio quality analysis

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5) |
| Purpose | Implement SRS Step 13 quality grading and the dataset quality report |
| Assistance requested | Seven checks (encoding, missing frames, duration, silence, level, clipping, noise, sample rate), grading, dataset report, builder integration, with tests |
| Files affected | `src/sonic/quality/**`, `config/quality.json`, `src/sonic/dataset/builder.py`, `tests/test_quality.py`, `documentation/quality.md`, `documentation/data_dictionary.md` |
| Student modifications | _TODO: team fills in_ |
| Testing completed | 12 quality tests (76 total); graded all 2,700 dataset clips; rebuilt dataset verified identical apart from the new grade columns |
| Verifying team members | _TODO_ |

## Entry 6: Model training, evaluation and comparison

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5) |
| Purpose | Train and compare five candidates (SVM, XGBoost, YAMNet transfer learning, CNN, YAMNet + CNN ensemble) under one evaluation protocol |
| Assistance requested | Shared training protocol (tuning on validation, temperature calibration, clip-level decisions as in the app), the model modules, ensemble, hidden-test simulator, factory-shortcut test, pre-registered selection rule, reports; diagnosing the Machinery Fault shortcut and adding the Normal Machinery class |
| Files affected | `src/sonic/models/**`, `config/sound_classes.json` (critical flags), `tests/test_model_evaluation.py`, `tests/test_cnn.py`, `python_models/**`, `reports/**` |
| Student modifications | _TODO: team fills in_ |
| Testing completed | Evaluation, selection-rule and CNN architecture tests; all four models trained on the real data; final comparison on the untouched test split (`reports/comparison.md`) |
| Verifying team members | _TODO_ |

Note: YAMNet (Google, Apache 2.0) is a pretrained network used only to turn audio into embeddings; its own AudioSet predictions are discarded. Every classification comes from classifiers we trained on our data. No generative-AI API is used for any prediction.

## Entry 7: GTM export and own-recordings importer

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5) |
| Purpose | Prepare GTM training audio from the Python model's training recordings; tooling and guide for recording the Help class |
| Assistance requested | Research on how GTM accepts audio (microphone only), playback-file export with a manifest, recording importer (ffmpeg conversion, metadata from file names), volunteer recording guide |
| Files affected | `src/sonic/gtm/**`, `src/sonic/dataset/recordings.py`, `src/sonic/dataset/sources/custom.py`, `tests/test_gtm_export.py`, `tests/test_recordings.py`, `documentation/gtm.md`, `documentation/help_recordings.md` |
| Student modifications | _TODO: team fills in_ |
| Testing completed | Export and importer tests (including a real .m4a conversion); export run on the real training split |
| Verifying team members | _TODO_ |

## Entry 8: GTM model training (automated recording)

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5) with Claude in Chrome |
| Purpose | Record the GTM training samples and train/export the GTM audio model |
| Assistance requested | Drove Teachable Machine in the team's Chrome (class names, recording length, record, extract, silent-sample removal, training, export) while PowerShell played the exported training audio into Stereo Mix; diagnosed a hidden-page capture failure and a silent-output problem |
| Files affected | `gtm_model/` (exported model), `documentation/gtm.md`, `documentation/development_log.md` |
| Student modifications | _TODO: team fills in_ |
| Testing completed | Per-class sample counts and silence check (peak level of every sample); GTM "Under the hood" accuracy/loss charts reviewed |
| Verifying team members | _TODO_ |

## Entry 9: Live-alert confirmation fix, alert handling and manual review

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5) |
| Purpose | Fix false live alerts found in the Stereo Mix test; build alert handling (FR lv, lvi, escalation) and the manual-review workflow (FR lvii-lxii) |
| Assistance requested | Analysed the live test windows and found why three false alerts were raised; wrote the repeat rule fix, the alert status changes and automatic escalation, the review queue, confirm/correct/override, comments, close/reopen, the pages and the tests |
| Files affected | `src/sonic/detection/decision.py`, `src/sonic/web/events/live.py`, `src/sonic/web/alerts/` (`handling.py`, `views.py`, `context.py`, `escalate_alerts` command), `src/sonic/web/events/` (`review.py`, `forms.py`, `views.py`, models and migration), `templates/alerts/`, `templates/events/review_*.html`, `tests/test_web_alerts.py` |
| Student modifications | _TODO: team fills in_ |
| Testing completed | 192 automated tests pass (21 new: every alert transition, refused transitions, permissions per role, escalation timing, override keeps the model outputs, queue order, save-and-next); pages checked in the browser |
| Verifying team members | _TODO_ |

## Entry 10: Dashboards, analytics, search, reports, export, admin settings and anomaly notices

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5) |
| Purpose | Milestone 6 of the web app: FR lxiii, lxv-lxx, lxxvii, lxxviii, lxxx and the configurable thresholds and rules (FR xxxv, xxxvi, liii) |
| Assistance requested | Wrote the search filters and timeline, the dashboard and analytics statistics and charts, the PDF analysis report, the CSV/Excel export (with formula-injection protection), the settings and alert-rule pages, data retention, the anomaly notices, and the error pages for model and database failures, with tests |
| Files affected | `src/sonic/web/events/` (`search.py`, `stats.py`, `dashboard.py`, `report.py`, `export.py`, `views.py`), `src/sonic/web/alerts/` (`forms.py`, `settings_views.py`, `retention.py`, `anomalies.py`, `AdminNotice` model, commands `apply_retention` and `check_anomalies`), `src/sonic/web/errors.py`, `static/js/charts.js`, `templates/dashboard/`, `templates/settings/`, `templates/errors/`, tests `test_web_search.py`, `test_web_dashboard.py`, `test_web_export.py`, `test_web_settings.py`, `test_web_anomalies.py` |
| Student modifications | _TODO: team fills in_ |
| Testing completed | 246 automated tests pass (54 new); every page checked in the browser; a real PDF report generated and read back |
| Verifying team members | _TODO_ |

## Entry 11: Model prediction and confidence comparison report

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5) |
| Purpose | SRS deliverable 6: compare the Python model and GTM on the unseen test recordings |
| Assistance requested | Wrote `python -m sonic.comparison` (prepare the run folder, a GTM page run in the browser with the app's own `gtm.js`, and the report that runs the Python model and the app's decision), the class-pair explanations of disagreements, and the tests; ran it on all 450 test clips |
| Files affected | `src/sonic/comparison/`, `src/sonic/gtm/model_info.py`, `src/sonic/web/events/gtm.py`, `reports/model_comparison.md`, `reports/model_comparison.csv`, `tests/test_comparison.py`, `.claude/launch.json` |
| Student modifications | _TODO: team fills in_ |
| Testing completed | 7 new automated tests; report figures cross-checked (agreement counts add up to the 449 scored clips; Python accuracy matches the earlier test-set evaluation within 0.2%) |
| Verifying team members | _TODO_ |

## Entry 12: Synthetic Person Asking for Help clips and adding them to the dataset

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5); Microsoft Edge text-to-speech through the `edge-tts` library (generates the audio, used with the organisers' written permission) |
| Purpose | Create the mandatory Person Asking for Help class, which no public dataset has |
| Assistance requested | Wrote `sonic.dataset tts` (47 English voices, the five SRS safety phrases with varied rate, pitch and loudness, plus ordinary sentences as negatives) and `sonic.dataset extend` (appends clips without changing the built dataset, one split per voice), with a test; ran them and started the retraining |
| Files affected | `src/sonic/dataset/tts.py`, `src/sonic/dataset/extend.py`, `src/sonic/dataset/__main__.py`, `src/sonic/dataset/sources/custom.py`, `src/sonic/dataset/report.py`, `data/metadata/dataset_metadata.csv`, `data/metadata/dataset_statistics.json`, `documentation/dataset.md`, `tests/test_dataset_builder.py`, `pyproject.toml` |
| Student modifications | _TODO: team fills in_ |
| Testing completed | Builder tests pass (17, one new); after extending, the 3,000 existing metadata rows are byte-identical and Help is split 210/45/45 by voice |
| Verifying team members | _TODO_ |

## Entry 13: Frontend migration from Bootstrap to Tailwind CSS

| Field | Details |
|---|---|
| Tool name | Claude Code (Anthropic, Claude Opus 5.5) |
| Purpose | Replace Bootstrap 5 with Tailwind CSS (CDN) without changing features or layout behaviour |
| Assistance requested | Planned and carried out the migration: theme and component layer (`templates/partials/tailwind.html`), all 26 templates, the page scripts, `ui.js` for the mobile menu and messages; checked every page for overflow and undefined classes |
| Files affected | `templates/` (all pages and partials), `static/js/ui.js`, `static/js/monitor.js`, `static/js/upload.js`, `src/sonic/web/*/forms.py` and `events/search.py` (mixin renamed), `static/css/app.css` (removed) |
| Student modifications | _TODO: team fills in_ |
| Testing completed | 254 automated tests pass; every page rendered at 375 px and 1280 px (no sideways scrolling), every class checked against the generated CSS, mobile menu and Escape key tested |
| Verifying team members | _TODO_ |

## Entry 14: Project report and GitHub submission preparation

| Field | Details |
|---|---|
| Tool name | OpenAI Codex |
| Purpose | Organize the SRS project report and prepare the full SonicSentinel project for a public GitHub repository |
| Assistance requested | Reviewed the SRS and project files; drafted Markdown and Word project reports; wrote uv-based setup and usage instructions; organized a separate local deliverables copy; updated package metadata, repository ignore rules, contribution record, and a small licensed sample-audio set |
| Files affected | `documentation/project_report.md`, `documentation/project_report.docx`, `README.md`, `.env.example`, `.gitignore`, `pyproject.toml`, `requirements.txt`, `database/.gitkeep`, `documentation/team_contributions.md`, `sample_audio/`, `src/sonic/__init__.py`, and the local ignored `submission/` directory |
| Student modifications | Team review and any edits after this preparation are to be recorded by the team |
| Testing completed | Markdown links checked; `uv lock --check` and Django system check passed; `uv run --locked pytest -q` completed with 251 passed and 3 skipped. The staged-file audit is completed before publication. |
| Verifying team members | To be confirmed by Saeed Ahmed, Manahil Khan, Muhammad Kaif, and Sahil Karim Lakahni |
