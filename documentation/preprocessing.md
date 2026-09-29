# Audio preprocessing

`sonic.preprocessing` turns any recording into fixed-length segments that
look the same whatever device or dataset they came from. Training, the GTM
sample export and the web app all call the same code, with settings from
`config/audio.json`.

```bash
uv run python -m sonic.preprocessing build   # dataset clips -> <data>/segments + data/metadata/segments.csv
uv run python -m sonic.preprocessing check   # shortcut check (below)
```

## Steps

| Step | What | Why |
|---|---|---|
| Mono | Stereo is averaged; microphone arrays (>2 channels, e.g. MIMII's 8 mics) use the first mic | Averaging an array acts like a beamformer and changes the sound |
| Resample | 16 kHz for everything (soxr, high quality) | MIMII is recorded at 16 kHz; a higher rate would leave 8–16 kHz empty only for Machinery Fault |
| Remove DC | Subtract the mean | Offsets distort energy measures |
| Noise reduction | Spectral gating, **off by default** | It would erase the Background Noise class and smear gunshots; compared on/off in the noise-robustness tests |
| Trim silence | Drop leading/trailing audio 40 dB below the peak | Removes ESC-50's digital-silence padding |
| Segment | 1 s windows, 0.5 s hop; the last window is aligned to the end | 1 s matches Google Teachable Machine's input |
| Event windows (training only) | Keep windows within 20 dB of the loudest, non-overlapping, max 5 per clip | A 0.5 s gunshot in a 4 s clip gives the window with the shot, not background labelled "Gunshot" |
| Normalize | Content scaled to −20 dBFS RMS, peaks capped at −1 dBFS; windows below −60 dBFS are flagged silent | Level differs per dataset and per distance |
| Place + noise floor | Short clips placed in a 1 s buffer (random offset in training, centred in inference); −70 dBFS noise added to every segment | No segment contains exact zeros, which some datasets have and others do not |

Inference mode (web app) returns every window in time order with its
timestamps and a `silent` flag; training mode returns only event windows.

## Shortcut check: why these steps matter

Each class comes mostly from one dataset, and datasets differ in format.
`python -m sonic.preprocessing check` trains a classifier on recording-format
features only (sample rate, channels, energy above 7 kHz, fraction of exact
zeros, level) and asks, within each class, *which dataset* a clip came from.
Accuracy above the majority guess means a dataset fingerprint a model could
learn instead of the sound.

| Class | Raw clips | Processed segments | Majority guess (processed) |
|---|---|---|---|
| Gunshot | 98.8% | 60.0% | 50.0% |
| Panic Scream | 100.0% | 57.5% | 50.0% |
| Animal Sound | 98.8% | 62.5% | 50.0% |
| Alarm or Siren | 97.2% | 38.3% | 33.3% |
| Vehicle Horn | 75.8% | 28.6% | 26.0% |
| Aggression | 100.0% | 70.6% | 54.1% |

Predicting the class from format alone fell from 48.3% to 32.2% (chance
11.1%). What remains is mostly real sound content: brightness and
impulsiveness genuinely differ between glass breaking, gunshots and horns.
Estimates use 40 clips per source and vary by about ±10 points between runs.
The latest output is in `data/metadata/shortcut_check.txt`.

**Known remaining bias:** every Machinery Fault clip is from MIMII, recorded
with factory background noise, and Aggression mixes studio speech (RAVDESS)
with Freesound shouts. Preprocessing cannot remove a background that is
mixed into the recording; augmentation (mixing other backgrounds into every
class) and the team's own recordings address it.
