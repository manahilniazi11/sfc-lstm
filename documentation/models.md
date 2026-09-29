# Python models: training, evaluation and selection

Five candidates were trained and compared under one protocol (SRS Step 7).
Full numbers: `reports/comparison.md` (and `comparison.json`); confusion
matrices: `reports/confusion/`; robustness chart: `reports/robustness.png`;
training logs: `reports/train_*_v2.log`; hyperparameter searches: `reports/tuning/`.

> **Status (2026-09-25):** 10 classes: the 9 mandatory classes that have data
> plus the optional **Normal Machinery**. Person Asking for Help has no
> recordings yet; all models must be retrained once it does.

## The candidates

| Model | Input | Architecture | Tuning |
|---|---|---|---|
| **SVM** | 123 summary features, standardized | RBF-kernel SVM; probabilities via sigmoid calibration (5-fold) | Grid C × gamma (20 settings) |
| **XGBoost** | 123 summary features | Gradient-boosted trees, `multi:softprob` | Random search (12 settings), early stopping |
| **YAMNet transfer** | 1 s waveform | Frozen YAMNet (AudioSet, MobileNet v1) → 1024-d embedding → Dense 256 → Dropout 0.5 → softmax | Grid hidden × dropout × lr (8 settings) |
| **CNN** | 64 × 63 log-mel spectrogram | 4 × [2 × (Conv3×3, BN, ReLU), MaxPool, Dropout] (32–256 filters) → avg + max pooling → Dense 128 → softmax; 1.24 M parameters | SpecAugment, mixup 0.2, label smoothing, AdamW + cosine decay, early stopping |
| **Ensemble** | both of the above | Mean of YAMNet's and the CNN's calibrated probabilities (soft voting), then its own calibration and clip aggregation | none (combines the saved models) |

Why these: two classical models on hand-made features, one network that
learns its own features, one that transfers knowledge from 2 million
AudioSet clips, and their combination. Random Forest and sklearn Gradient
Boosting were left out (same family as XGBoost, usually weaker or slower);
a CRNN was left out because 1 s windows are short enough for a CNN to see
the whole event. YAMNet's own AudioSet predictions are never used.

**Disclosure:** the ensemble was added after the first test comparison
showed that YAMNet and the CNN miss *different* critical classes. It was
judged on validation like every other candidate, but the idea came from
test results.

Compute: a 4-core CPU without a GPU. XGBoost's search was narrowed after
one trial took ~4 minutes; the CNN trained one variant (~2 min per epoch).

## Protocol

1. **Train** on the training split (original + augmented segments, 1,500
   per class); **tune** on validation segments.
2. **Calibrate** confidence on validation with two temperatures: one for
   single 1 s windows (live monitoring), one for whole-clip decisions
   (uploads), because averaging windows makes clip scores less extreme.
3. **Clip decisions as in the app:** every 1 s window of a recording is
   scored and combined per clip (mean / max / top-3, chosen on validation).
4. **Select** with a rule fixed before the test set was scored (validation
   only): critical-class gate → validation macro F1 → within 0.01, higher
   mean critical recall, then speed.
5. **Test once:** every candidate on the untouched test split, plus a
   10-condition hidden-test simulator and a factory-shortcut test.

## Results (test split, per clip, 10 classes, 450 clips)

| Model | Accuracy | Macro F1 | Macro precision | Macro recall | Robust macro F1 | Factory test (called fault) | 30 s upload |
|---|---|---|---|---|---|---|---|
| SVM | 75.3 % | 0.753 | 0.760 | 0.753 | 0.580 | 22 % ⚠ | 3.1 s |
| XGBoost | 77.1 % | 0.769 | 0.772 | 0.771 | 0.587 | 3 % | 0.3 s |
| YAMNet | 78.2 % | 0.780 | 0.782 | 0.782 | 0.622 | 48 % ⚠ | 1.2 s |
| CNN | 81.6 % | 0.814 | 0.824 | 0.816 | 0.714 | **0 %** | 0.5 s |
| **Ensemble** (selected) | **86.2 %** | **0.861** | **0.863** | **0.862** | **0.733** | 5 % | 1.2 s |

**Critical-event recall** (target ≥ 85 %):

| Model | Glass Breaking | Gunshot | Panic Scream | Aggression |
|---|---|---|---|---|
| SVM | 87 % | 80 % | 78 % | 73 % |
| XGBoost | 89 % | 82 % | 71 % | 80 % |
| YAMNet | 87 % | 93 % | **87 %** | 78 % |
| CNN | 80 % | **98 %** | 69 % | 69 % |
| Ensemble | **89 %** | 96 % | 82 % | **82 %** |

**Against the SRS targets (NFR 4):** the ensemble meets accuracy ≥ 85 %
(86.2 %) and macro F1 ≥ 0.80 (0.861). Critical recall ≥ 85 % is met for
Glass Breaking and Gunshot; Panic Scream and Aggression are at 82 %
(these two loud human-voice classes are confused with each other most).

**Hidden-test simulator:** the ensemble is best or close to best in every
condition (e.g. MP3 0.846, partial event 0.813, phone 0.761). The CNN is the
most robust single model and slightly better than the ensemble at 0 dB
noise, low volume and distance, where YAMNet degrades most. All models
struggle with *low volume over microphone hiss* (0.34–0.55).

**Speed:** every model meets SRS NFR 1 (live window ≤ 3 s; 30 s upload ≤ 8 s).
The ensemble needs ~13 ms per 1 s window.

## Selection

**Selected: `ensemble-20260925-1`** (members `yamnet-20260925-2` and
`cnn-20260925-2`; `python_models/selected.json`). No candidate reached 85 %
on every critical class on validation, so all were eligible; the ensemble
had the best validation macro F1 (0.859 vs 0.821 for the CNN).

## The Machinery Fault shortcut, found and fixed

The first comparison (9 classes, `*-20260925-1` models) tested every model
on 60 MIMII *normal-operation* recordings never used anywhere else: **all
four called 98–100 % of them Machinery Fault**. The models had learned to
recognise MIMII's factory recordings, not the fault, and Machinery Fault's
near-perfect scores were mostly that shortcut. Mixing factory noise under
other classes in augmentation had not been enough.

Fix: the optional SRS class **Normal Machinery** (SRS Step 14: "machinery
fault vs. normal machinery") with 300 MIMII normal-operation clips (200
valve, 50 pump, 50 fan), and a noise pool built only from training
segments. After retraining, the CNN called **0 %** of the unseen normal
recordings a fault, XGBoost 3 %, the ensemble 5 %. Machinery Fault F1 is
now a believable 0.92–0.96 instead of an inflated 0.98–1.00.

YAMNet still calls 48 % of them faults: its AudioSet embeddings describe
"a machine" well but not the subtle difference between a healthy and a
faulty valve, while the CNN, trained from scratch on our spectrograms, learns
it. The ensemble inherits the CNN's judgement here.

## Earlier version (for the record)

First comparison, 9 classes, before the fix (`reports` history in git):
YAMNet 82.7 % / 0.823, CNN 82.2 % / 0.820 (selected then), SVM 74.8 %,
XGBoost 72.8 %; factory test failed for every model.

## Next improvements

1. Record the Help class and retrain everything (ideally on a Colab GPU).
2. Raise Panic Scream / Aggression recall: more varied scream and shouting
   data (own recordings), or a lower alert threshold for these classes in
   the alert rules.
3. Add microphone hiss / low-level recordings to augmentation, and test the
   (currently disabled) noise-reduction step on the low-volume condition.
4. Train the remaining CNN variants on a GPU.
