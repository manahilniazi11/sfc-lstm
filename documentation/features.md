# Feature extraction

`sonic.features` turns each preprocessed 1 s segment (16 kHz mono) into two
representations, both computed from one short-time Fourier transform
(512-sample frames every 256 samples, about 63 frames). Settings are in
`config/features.json`; the web app calls the same `extract` function.

```bash
uv run python -m sonic.features build   # all segments -> <data>/features/
uv run python -m sonic.features list    # the 123 feature names
```

## 1. Summary vector (classical models)

Each frame-level feature is summarised over the segment. 123 values:

| Feature | Statistics | Values | What it captures | Helps with |
|---|---|---|---|---|
| MFCC (20 coefficients) | mean, std | 40 | Timbre: the shape of the spectral envelope | Telling voices, engines, animals apart |
| MFCC delta | std | 20 | How quickly the timbre changes | Speech and screams vs steady hums |
| Chroma (12 pitch classes) | mean, std | 24 | Energy per musical pitch class | Sirens, horns, alarms (tonal) |
| Mel bands (64 → 16 coarse) | mean | 16 | Loudness per frequency region | Low rumble vs high shatter |
| Zero-crossing rate | mean, std | 2 | How often the waveform changes sign | Noisy vs tonal sounds |
| RMS energy | mean, std, max | 3 | Energy and how impulsive it is | Gunshots, glass, clicks |
| Spectral centroid | mean, std | 2 | "Centre of mass" of the spectrum (Hz) | Bright vs dull sounds |
| Spectral bandwidth | mean, std | 2 | Spread of energy around the centroid | Broadband vs narrow sounds |
| Spectral roll-off (85 %) | mean, std | 2 | Frequency below which 85 % of the energy lies | Brightness |
| Spectral contrast (7 bands) | mean | 7 | Peaks vs valleys per octave | Tonal (horn) vs noisy (rain) |
| Spectral flatness | mean, std | 2 | 1 = white noise, 0 = pure tone | Background noise vs events |
| Onset strength | mean, std, max | 3 | Sudden increases in energy | Gunshots, glass, impacts |

Chroma uses fixed standard tuning (A = 440 Hz): estimating tuning per segment
fails on unpitched sounds and makes chroma inconsistent between segments.

**Tempo** is not extracted: a 1 s segment holds at most one or two beats, so
a tempo estimate would be noise (the SRS asks for tempo "where relevant").

## 2. Log-mel spectrogram (CNN)

64 mel bands from 20 Hz to 8 kHz × 63 frames, in dB (80 dB range). The CNN
learns its own patterns from this image. Stored as float16.

## Outputs (`<data>/features/`, same row order)

| File | Content |
|---|---|
| `features.csv` | `segment_id, audio_id, class_name, split, is_augmented, source_dataset` + 123 features |
| `logmel.npy` | float16 array (rows, 64, 63) |
| `feature_info.json` | Feature names and settings, so a model can check its input matches |

15,957 rows: 8,150 original segments (all splits) and 7,807 augmented
training segments. Extraction takes about 7 ms per segment.

## First check

An untuned Random Forest on the training features reaches 68.2 % accuracy
(macro F1 0.66) on single 1 s validation segments (chance 11 %). The most
useful features were RMS energy, spectral contrast, MFCC 0 variation,
bandwidth, zero-crossing rate and roll-off. Clip-level decisions, tuning and
the CNN come in the model-training step.
