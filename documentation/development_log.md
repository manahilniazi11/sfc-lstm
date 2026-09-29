# Development log

Required by SRS §1.8 item 3. One entry per day: work completed, problems
encountered, dataset changes, model failures, code changes and tests
performed. AI assistance is declared separately in `AI_USAGE.md`.

---

## Day 1 — 2026-09-24

### Work completed
- Read the SRS and fixed the scope: Django web app, Python model + Google
  Teachable Machine (GTM) model, 10 sound classes.
- Chose the class list (`config/sound_classes.json`) as the single source of
  truth for class names. The SRS spells one class two ways ("Aggression" and
  "Aggression or Violent Conflict"); we use **Aggression** everywhere.
- Built the dataset builder (`sonic.dataset`): downloads, label mapping,
  de-duplication, group-aware 70/15/15 split, metadata, statistics.
- Built the dataset: **2,700 clips, 300 unique recordings for each of 9
  classes**. Person Asking for Help has no public source and needs our own
  recordings.
- Built audio preprocessing (`sonic.preprocessing`), augmentation
  (`sonic.augmentation`) and feature extraction (`sonic.features`).

### Dataset sources
ESC-50, UrbanSound8K, FSD50K (dev + eval), MIMII (valve), RAVDESS,
Nonspeech7k and Freesound (API). All metadata, including source and license
for every clip, is in `data/metadata/dataset_metadata.csv`.

### Problems encountered and how they were solved
| Problem | Cause | Fix |
|---|---|---|
| UrbanSound8K download "finished" at 1.08 GB of 5.6 GB, then failed to unpack | The VPN dropped the connection; the downloader treated a closed connection as a finished download | Downloader now keeps a `.part` file until the size matches, resumes automatically (HTTP Range) and never unpacks an incomplete file. It later survived a real drop at 3.94 GB. |
| PC shut down in the middle of a download | Power off | Resumed from the partial file; the last 64 MB were trimmed first in case they were half-written |
| Full archives far too big (FSD50K dev 17 GB, MIMII 6–10 GB each) | We only need a fraction of the clips | Wrote a selective zip reader: it reads the archive's table of contents over HTTP and fetches only the needed files. FSD50K dev: 0.6 GB instead of 17 GB; the whole queue about 2 GB instead of about 29 GB. |
| C: drive short of space | Audio is tens of GB | Audio moved to `E:\sonic_data`, configured with `SONIC_DATA_DIR` in `.env` |
| Two guessed FSD50K labels did not exist (`Fire_alarm`, `Smoke_detector…`) | Guessed from AudioSet, not checked | Checked every label against FSD50K's `vocabulary.csv`; the builder now warns about unknown labels |
| FSD50K "Alarm" includes telephones, ringtones, doorbells and bicycle bells | "Alarm" is a parent label in the AudioSet ontology | Clips with those labels are excluded (`_exclude_if_present`) |
| Vehicle-horn clips skipped as "ambiguous" | They also carry the parent label "Alarm" | "Generic" labels only decide the class when no specific label is present |
| Unique-recording count too low for RAVDESS/CREMA-D | Clips are grouped by actor (to keep a speaker in one split), and the count used the group | Separated `origin_id` (unique recording) from `group_id` (split group) |
| Vehicle Horn short of 300 unique recordings (232) | Public datasets exhausted | 200 horns from the Freesound API (CC0/CC BY/CC BY-NC only). The name/tag filter removed non-vehicle results ("Melodica Car Horn Imitation", "Car Alarm Horn", vocal imitations). |
| Panic Scream only just above 300 | Few scream sources | Added Nonspeech7k screams (selective download) |
| Download crashed at 762/773 clips | A truncated server response (`IncompleteRead`) was not retried | Retried like other connection errors; one failed file no longer stops a run |

### Dataset bias found and reduced
Each class comes mostly from one dataset, and datasets differ in sample
rate, channels, padding and level, so a model could learn *the dataset*
instead of *the sound*. We measured this with a "shortcut check" that uses
only recording-format features:

- Raw clips: format alone predicted the class 48 % of the time (chance
  11 %), and within one class told the source dataset apart 75–100 % of the
  time.
- Fixes in preprocessing: resample everything to 16 kHz (MIMII is recorded
  at 16 kHz), mono (first mic of MIMII's 8-mic array), trim silence,
  normalize loudness, and a faint −70 dBFS noise floor so no segment
  contains exact digital zeros.
- After preprocessing, dataset identification within a class dropped to
  near guessing (e.g. Gunshot 98.8 % → 60 %, guess 50 %).
- A first version added the noise floor *before* normalization, so its
  final level still depended on the source; moved after normalization.
- The factory background in every Machinery Fault clip (all MIMII) cannot
  be removed; augmentation mixes MIMII *normal* factory recordings into
  every class instead.

### Code changes
Modules added: `sonic.dataset`, `sonic.preprocessing`, `sonic.augmentation`,
`sonic.features`. Config files in `config/`.

### Tests performed
64 automated tests (`uv run pytest`): synthetic audio for every
preprocessing, augmentation and feature step; fake servers that drop
connections for the downloader; a locally served zip for the selective
reader. A quick untuned Random Forest reached 68 % on single 1 s
validation segments, confirming the features carry class information.

---

## Day 2 — 2026-09-25

### Work completed
- Committed the Day 1 work to git (`main` branch), one commit per module.
- Built audio quality analysis (`sonic.quality`, SRS Step 13): seven checks,
  Good / Acceptable / Poor / Unusable grades. Produced the dataset quality
  report: 84 % Good, 10 % Acceptable, 5.5 % Poor, 0 Unusable.
- Connected quality grading to the dataset builder; the rebuilt dataset was
  verified identical apart from the new grade columns.
- Marked the SRS critical classes (Gunshot, Glass Breaking, Panic Scream,
  Aggression, Person Asking for Help) in the class config.
- Started model training (four models, one at a time on the 4-core CPU).

### Problems encountered and how they were solved
| Problem | Cause | Fix |
|---|---|---|
| All 300 Machinery Fault clips graded Poor for "low signal" | MIMII records quietly (≈ −45 dBFS) but cleanly on 16-bit audio; threshold too strict | Low-signal thresholds relaxed to −48 / −55 dBFS |
| Chroma extraction printed warnings on some segments | Per-segment tuning estimation fails on unpitched sounds (noise, gunshots) | Fixed standard tuning (A = 440 Hz); also makes chroma consistent between segments |
| `tensorflow-hub` failed to import | It needs `pkg_resources`, removed from current `setuptools` | Dropped `tensorflow-hub`; YAMNet is loaded directly with TensorFlow |

| SVM run used a deprecated option | scikit-learn 1.9 deprecates `SVC(probability=True)` | Switched to `CalibratedClassifierCV(SVC(), ensemble=False)` before training |
| Clip-level confidence poorly calibrated (ECE 0.24) | Averaging windows makes clip scores less extreme than window scores | Second calibration temperature fitted on clip decisions (ECE 0.10) |
| XGBoost search would take ~2 hours | One trial with lr 0.05 / depth 9 took ~4 minutes on the CPU | Narrowed to 12 trials, lr 0.08–0.3, depth 4–8 |
| CNN: 3 variants would take ~3.5 hours | ~2 minutes per epoch on the CPU, no GPU | Trained the main variant only; the others are kept for a Colab GPU run |
| Final comparison crashed on YAMNet test segments | A model loaded from disk had no embedding cache attached | Cache now loaded on demand |
| GTM cannot import audio files | Teachable Machine audio projects only record from a microphone | Export one playback WAV per class of training segments, played into a virtual audio cable while GTM records (`documentation/gtm.md`) |

### Model results (test split, per clip, 9 classes)
| Model | Accuracy | Macro F1 | Robust macro F1 |
|---|---|---|---|
| SVM | 74.8 % | 0.749 | 0.603 |
| XGBoost | 72.8 % | 0.724 | 0.593 |
| YAMNet | 82.7 % | 0.823 | 0.673 |
| CNN (selected) | 82.2 % | 0.820 | 0.722 |

### Model failure found
The factory-shortcut test failed for every model: 98–100 % of 60 unseen
MIMII normal-operation recordings were called Machinery Fault. The models
learned "MIMII-style factory recording", not "fault". Augmentation with
factory noise under other classes was not enough; labelled normal-machinery
examples are needed (see `documentation/models.md`).

### Fix and retraining
- Added the optional SRS class **Normal Machinery** (300 MIMII normal clips:
  200 valve, 50 pump, 50 fan). Placed last in the class list so the
  existing 2,700 clips kept their Audio IDs and splits (verified).
- The augmentation noise pool now uses only training segments of Background
  Noise and Normal Machinery (raw MIMII files would have leaked
  validation/test clips into training).
- Found that model caches (test windows, YAMNet embeddings) were keyed only
  on feature settings, so a data rebuild would have silently reused stale
  inputs; they are now keyed on the data files too.
- Added a YAMNet + CNN ensemble (soft voting). Disclosed: the idea came from
  the first test results (complementary errors).
- Retrained all models (12:33–13:58) and re-ran the comparison.
- Tooling slip: two scripted edits of this log used Windows' default text
  encoding instead of UTF-8; one emptied the file (restored from git), the
  other left a mis-encoded dash (repaired). All other files were checked.

| Model (10 classes) | Test accuracy | Test macro F1 | Factory test: called fault |
|---|---|---|---|
| SVM | 75.3 % | 0.753 | 22 % |
| XGBoost | 77.1 % | 0.769 | 3 % |
| YAMNet | 78.2 % | 0.780 | 48 % |
| CNN | 81.6 % | 0.814 | 0 % |
| **Ensemble (selected)** | **86.2 %** | **0.861** | 5 % |

The ensemble meets the SRS accuracy (≥ 85 %) and macro F1 (≥ 0.80) targets.
Panic Scream and Aggression recall (82 %) are still below the 85 % target.

### Decisions
- Poor-quality clips stay in the training data on purpose: the SRS hidden
  tests include noisy, clipped and low-volume audio.
- Models train one at a time: each already uses all 4 CPU cores.
- The test split is used once, for the final comparison of all models.

- Model selection rule fixed before the test set was scored (validation only).

### Tests performed
90 automated tests (quality, evaluation, selection rule, CNN architecture,
GTM export, recordings importer). Final comparison of four models on the
untouched test split with a 10-condition hidden-test simulator.

## Day 3 — 2026-09-26 / 2026-09-27

### Work completed
- Rebuilt the GTM playback files (10 classes × 150 one-second training
  segments) and trained the Google Teachable Machine audio model on them
  through Stereo Mix; exported it to `gtm_model/` (TensorFlow.js).
- Recording was automated (PowerShell playback + scripted GTM clicks), with
  a silence check after every class. Details in `documentation/gtm.md`.

### Problems encountered and how they were solved
- GTM's *Upload* accepts only its own sample zips, so the training audio has
  to be played into a virtual microphone.
- First attempt captured only ~4 s of a 145 s recording: Chrome pauses
  hidden (covered) pages. Fix: keep the Chrome window visible.
- Mid-session the PC's output device changed and was muted; ~30 s of
  Alarm or Siren were silent. Cleared and re-recorded the class; added a
  level check and an automatic silent-sample count per class.

### Model results
GTM v1: training accuracy ~0.90, GTM's internal held-out accuracy ~0.46
(10 classes, ~145 samples each, 50 epochs). Over-fits; clearly weaker than
the Python ensemble. The proper comparison on the unseen test recordings
follows in the web app.

### Live monitoring test (Stereo Mix, 20 unseen test clips)
- Python model heard 19/20 clips correctly in at least one window, GTM
  17/20; result median 1.1 s (max 1.8 s) after each 2 s window ended.
- Problem: three false alerts. A lone Gunshot window during an Animal
  Sound, a Machinery Fault heard between Normal Machinery windows, and an
  Alarm or Siren in the gap between two clips. Cause 1: the live path
  capped "consecutive detections needed" at the number of model windows
  inside one live window, so a short sound (one model window) never needed
  a repeat. Cause 2: repeats were counted regardless of what was heard in
  between. Fix: live mode always needs the rule's number of detections,
  and a different sound in between breaks the chain (background noise and
  silence do not). Replaying the session: all three false alerts go to
  manual review, every real alert remains (Gunshot is confirmed 2 s
  later, by its second window).
- Still open, for the retraining round: one Alarm or Siren clip was
  missed, and one window of it was heard as Aggression.

## Day 4 — 2026-09-28

### Work completed
- Alert handling: acknowledge, escalate, dismiss (reason required) and
  resolve, with the full history of every alert and the audit trail.
  Alerts nobody acknowledges are escalated automatically after the rule's
  `escalate_after_minutes` (checked while the app is used, and by
  `python manage.py escalate_alerts` for a scheduler).
- Manual-review queue: most severe first, filters by reason, severity and
  source. Reviewers listen to the audio, confirm or correct the class,
  override severity and recommended action, comment, close or reopen, and
  can work through the queue with "Save and review next". The models'
  outputs are never changed; the decision is stored beside them.
- Menu badges show open alerts and waiting reviews.

### Milestone 6 (same day)
- Event history: search by Audio ID, event ID or file name, and filter by
  category, date range, confidence range, severity, quality, review
  status, source and user; a timeline view by day.
- Dashboard for every role (own data for normal users, the whole system
  for operators) and system charts for reviewers and administrators;
  an analytics page (category and critical-event frequency, confidence
  distribution, disagreement, quality, false positives and negatives,
  alert response times).
- PDF analysis report per event; CSV and Excel export for administrators.
- Settings pages for thresholds, alert rules and data retention.
- Administrator notices for failed uploads and logins, duplicates, alert
  floods, low-confidence spikes and model failures; readable errors
  instead of error pages when a model or the database fails.

### Problems encountered and how they were solved
- The retention preview counted the oldest records twice (as audio and
  as whole records); each record is now counted once, so the preview
  matches what is deleted.
- Filtering audit rows with `ip_address=""` matched nothing because
  Django turns an empty IP into NULL; the notice check filters set
  addresses explicitly.

### Person Asking for Help and retraining (same day)
- 300 synthetic Help clips (edge-tts, 47 voices) and 47 ordinary sentences
  (Background Noise) were appended to the dataset with the new
  `sonic.dataset extend`, which leaves the 3,000 existing clips and their
  splits unchanged. Help test clips come from 7 unseen voices.
- All five Python models retrained on 11 classes (2 h, unattended). The
  selected ensemble-20260928-1 reaches 87.6% test accuracy and macro F1
  0.877 (was 86.2% / 0.861); Panic Scream recall rose to 89% (target met),
  Help F1 is 0.98, Aggression stays at 80%.
- The threshold study was re-run: the new model is better calibrated, and
  the thresholds from alert_rules.json still separate reliable results
  from the ones worth a review, so they were kept.
- Caveat: the Help result is measured on synthetic (unseen) voices, which
  are cleaner than a real person calling for help.

### Frontend: Bootstrap to Tailwind CSS (same day)
- All 26 templates and the page scripts moved from Bootstrap 5 to Tailwind
  CSS 4 (browser build from jsdelivr, pinned to 4.3.3). One partial,
  `templates/partials/tailwind.html`, holds the theme, a base layer and the
  component classes; `static/css/app.css` is gone. Bootstrap Icons stay.
- To change nothing a user would notice, the theme uses Bootstrap's screen
  widths, the base layer restores Bootstrap's typographic defaults, spacing
  was converted by size (Bootstrap `mb-3` = 16 px = Tailwind `mb-4`), and
  the class names generated by Python and the scripts (form fields, badge
  colours, severity) are kept as components.
- Bootstrap's JavaScript is replaced by `static/js/ui.js` (the mobile menu
  and dismissible messages).
- Checked: every page at phone and desktop width for sideways overflow, and
  every class on every page against the generated CSS. Problems found and
  fixed: visually hidden labels escaping scrolling tables (now inside
  `.table-wrap`), the history view toggle (`.btn-group`) and bullets on the
  alert tabs.
- Note: the browser build compiles the CSS in each page; for deployment the
  Tailwind standalone CLI can produce the same CSS as a static file.
