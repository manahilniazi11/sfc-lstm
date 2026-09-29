# Model comparison (test set)

Selected: **ensemble-20260928-1** (no model reached 85 % on every critical class, so all were eligible). Rule: see `src/sonic/models/compare.py`.

## Overall

| Model | Val macro F1 | Test accuracy | Test macro F1 | Test macro precision | Test macro recall | ECE | Robust macro F1 | Retained | ms / window | 30 s upload (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| svm-20260928-1 | 0.756 | 74.3% | 0.741 | 0.746 | 0.746 | 0.080 | 0.582 | 79% | 1.9 | 3213 |
| xgboost-20260928-1 | 0.792 | 78.7% | 0.784 | 0.787 | 0.789 | 0.036 | 0.602 | 77% | 0.0 | 408 |
| yamnet-20260928-1 | 0.838 | 80.5% | 0.804 | 0.809 | 0.808 | 0.075 | 0.633 | 79% | 5.5 | 717 |
| cnn-20260928-1 | 0.857 | 84.5% | 0.847 | 0.852 | 0.845 | 0.025 | 0.738 | 87% | 1.7 | 508 |
| ensemble-20260928-1 | 0.863 | 87.6% | 0.877 | 0.879 | 0.879 | 0.037 | 0.752 | 86% | 8.1 | 787 |

## Critical-event recall (test, target >= 85 %)

| Model | Glass Breaking | Gunshot | Panic Scream | Aggression | Person Asking for Help |
|---|---|---|---|---|---|
| svm-20260928-1 | 87% | 80% ✗ | 69% ✗ | 69% ✗ | 98% |
| xgboost-20260928-1 | 89% | 84% ✗ | 78% ✗ | 69% ✗ | 98% |
| yamnet-20260928-1 | 89% | 93% | 84% ✗ | 73% ✗ | 96% |
| cnn-20260928-1 | 91% | 89% | 71% ✗ | 78% ✗ | 100% |
| ensemble-20260928-1 | 91% | 93% | 89% | 80% ✗ | 100% |

## Class-wise F1 (test)

| Class | svm-20260928-1 | xgboost-20260928-1 | yamnet-20260928-1 | cnn-20260928-1 | ensemble-20260928-1 |
|---|---|---|---|---|---|
| Machinery Fault | 0.86 | 0.96 | 0.68 | 0.99 | 0.97 |
| Glass Breaking | 0.83 | 0.80 | 0.92 | 0.91 | 0.90 |
| Alarm or Siren | 0.63 | 0.66 | 0.79 | 0.75 | 0.80 |
| Vehicle Horn | 0.66 | 0.67 | 0.87 | 0.79 | 0.84 |
| Animal Sound | 0.66 | 0.74 | 0.87 | 0.84 | 0.89 |
| Gunshot | 0.83 | 0.78 | 0.88 | 0.91 | 0.91 |
| Panic Scream | 0.70 | 0.79 | 0.82 | 0.70 | 0.82 |
| Aggression | 0.67 | 0.69 | 0.77 | 0.74 | 0.80 |
| Person Asking for Help | 0.85 | 0.91 | 0.88 | 1.00 | 0.98 |
| Background Noise | 0.62 | 0.68 | 0.67 | 0.73 | 0.75 |
| Normal Machinery | 0.85 | 0.97 | 0.70 | 0.97 | 0.98 |

## Hidden-test simulator (test macro F1)

| Condition | svm-20260928-1 | xgboost-20260928-1 | yamnet-20260928-1 | cnn-20260928-1 | ensemble-20260928-1 |
|---|---|---|---|---|---|
| clean | 0.741 | 0.784 | 0.804 | 0.847 | 0.877 |
| noise_20dB: background noise at 20 dB SNR | 0.669 | 0.697 | 0.755 | 0.817 | 0.840 |
| noise_10dB: background noise at 10 dB SNR | 0.647 | 0.656 | 0.689 | 0.767 | 0.794 |
| noise_5dB: background noise at 5 dB SNR | 0.567 | 0.594 | 0.617 | 0.717 | 0.735 |
| noise_0dB: background noise at 0 dB SNR (as loud as the event) | 0.435 | 0.468 | 0.499 | 0.607 | 0.624 |
| echo: strong reverb (RT60 0.8 s, 50 % wet) | 0.716 | 0.738 | 0.656 | 0.816 | 0.829 |
| low_volume: -30 dB quieter over a -60 dBFS microphone hiss | 0.347 | 0.407 | 0.484 | 0.490 | 0.492 |
| phone: phone microphone (300-3400 Hz) with clipping | 0.641 | 0.662 | 0.658 | 0.785 | 0.803 |
| distant: far-away source (low-pass 2.5 kHz, reverb) | 0.510 | 0.464 | 0.501 | 0.747 | 0.707 |
| partial: only 40-60 % of the recording | 0.656 | 0.703 | 0.744 | 0.791 | 0.821 |
| mp3_64k: re-encoded as low-bitrate MP3 | 0.636 | 0.633 | 0.731 | 0.842 | 0.873 |

## Factory-shortcut test

60 MIMII normal-operation recordings never used in the dataset, training or augmentation. A model that learned the fault (not the factory background) should not call them Machinery Fault; the correct answer is Normal Machinery.

| Model | Called Machinery Fault | Predictions |
|---|---|---|
| svm-20260928-1 | 23% ⚠ flagged | Normal Machinery: 46, Machinery Fault: 14 |
| xgboost-20260928-1 | 8% | Normal Machinery: 55, Machinery Fault: 5 |
| yamnet-20260928-1 | 48% ⚠ flagged | Machinery Fault: 29, Normal Machinery: 27, Glass Breaking: 2, Background Noise: 2 |
| cnn-20260928-1 | 0% | Normal Machinery: 60 |
| ensemble-20260928-1 | 0% | Normal Machinery: 60 |

## Accuracy by source dataset (test)

| Source | svm-20260928-1 | xgboost-20260928-1 | yamnet-20260928-1 | cnn-20260928-1 | ensemble-20260928-1 |
|---|---|---|---|---|---|
| edge-tts (52 clips) | 85% | 98% | 87% | 100% | 100% |
| esc50 (19 clips) | 63% | 63% | 84% | 68% | 84% |
| freesound (13 clips) | 62% | 62% | 92% | 77% | 85% |
| fsd50k (252 clips) | 70% | 69% | 79% | 77% | 81% |
| mimii (90 clips) | 86% | 97% | 72% | 99% | 98% |
| nonspeech7k (16 clips) | 81% | 88% | 81% | 81% | 94% |
| ravdess (2 clips) | 50% | 100% | 100% | 100% | 100% |
| urbansound8k (58 clips) | 72% | 79% | 90% | 90% | 91% |
