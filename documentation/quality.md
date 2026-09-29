# Audio quality analysis

`sonic.quality` implements SRS Step 13 (and FR xii silence detection, FR xiii
clipping detection, FR xiv background-noise estimation, FR xxxvii quality
classification). It runs on the **raw** recording, before normalization, and
grades it **Good, Acceptable, Poor or Unusable**. Thresholds are in
`config/quality.json`.

```bash
uv run python -m sonic.quality report          # grade every dataset clip
uv run python -m sonic.quality file some.wav   # grade one file
```

## Checks

The grade is the worst result of seven checks; the issues list says why.

| Check (SRS) | How | Acceptable | Poor | Unusable |
|---|---|---|---|---|
| Encoding problems | File must decode; no NaN/infinite samples | | | cannot decode |
| Missing audio frames | Readable length vs declared length; interior runs of exact zeros ≥ 20 ms (dropouts) | 1–2 dropouts | 3+ dropouts, or >1 % missing | >50 % missing |
| Unsuitable duration | | | > 10 min | < 0.3 s |
| Silence | Peak level; share of 25 ms frames below −60 dBFS | | >95 % silent | peak < −60 dBFS |
| Low signal strength | Level of the non-silent part | < −48 dBFS | < −55 dBFS | |
| Clipping | Samples at ≥ 99 % of full scale for 3+ consecutive samples (a single peak is not clipping) | ≥ 0.1 % | ≥ 1 % | |
| Excessive noise | Steady (small gap between loud and quiet frames) **and** noise-like (high spectral flatness) | range < 6 dB, flatness > 0.3 | range < 3 dB, flatness > 0.5 | |
| Sample rate | | < 16 kHz | < 8 kHz | |

The noise check needs both conditions so that steady *tonal* sounds (sirens,
horns, machine hums) are not called noisy. It is skipped for the Background
Noise class, which is noise by definition. The estimated noise floor (10th
percentile of frame levels) is always reported (FR xiv).

## Dataset quality report

The dataset builder grades every candidate clip: Unusable clips are rejected,
and the grade and issues are stored in `dataset_metadata.csv`
(`quality_grade`, `quality_issues`). Full results:
`data/metadata/quality_report.csv` and `quality_summary.json`.

| Class | Good | Acceptable | Poor |
|---|---|---|---|
| Aggression | 281 | 10 | 9 |
| Alarm or Siren | 252 | 23 | 25 |
| Animal Sound | 257 | 28 | 15 |
| Background Noise | 275 | 18 | 7 |
| Glass Breaking | 254 | 29 | 17 |
| Gunshot | 201 | 51 | 48 |
| Machinery Fault | 228 | 72 | 0 |
| Panic Scream | 254 | 24 | 22 |
| Vehicle Horn | 273 | 22 | 5 |
| **Total** | **2,275 (84 %)** | **277 (10 %)** | **148 (5.5 %)** |

No clip in the final dataset is Unusable: the builder already rejected 41
files (17 too short, 1 silent, 23 label conflicts).

Most common issues: clipping (204 clips; real gunshot and scream recordings
overload the microphone), missing audio frames (124; mostly edited FSD50K
uploads with stretches of exact zeros), low signal strength (97).

**Poor clips are kept for training on purpose.** The SRS hidden tests include
low-quality, clipped, noisy and distant audio; a model that never saw such
audio would do worse on it. Unusable clips are always removed.

**Calibration note.** The first low-signal thresholds (−35 / −45 dBFS) graded
all 300 Machinery Fault clips Poor: MIMII records at about −45 dBFS. On
16-bit audio that is quiet but clean (about 50 dB of dynamic range), so the
thresholds were moved to −48 / −55 dBFS.

In the web app the same grade feeds manual review (FR lvii: poor audio
quality) and the dashboard's quality warnings.
