# Dataset guide

The SRS asks for at least 3,000 unique original clips, 300 per class, split
70/15/15 with the same split used by the Python model and by Google Teachable
Machine. `python -m sonic.dataset` builds that dataset from public datasets
plus the team's own recordings.

## 0. Choose where the audio lives

The raw downloads and the built dataset take tens of GB, so they can live on
any drive. Copy `.env.example` to `.env` and set:

```
SONIC_DATA_DIR=E:/sonic_data
```

Below, `<data>` means that folder (default: `data/` in the repository). The
metadata in `data/metadata/` always stays in the repository.

## 1. Get the raw data

```bash
uv run python -m sonic.dataset download --list
uv run python -m sonic.dataset download esc50 urbansound8k ravdess fsd50k-labels
uv run python -m sonic.dataset download fsd50k-eval fsd50k-dev mimii-valve nonspeech7k
uv run python -m sonic.dataset freesound          # needs FREESOUND_API_KEY in .env
```

Everything lands in `<data>/raw/<source>/`.

Items marked `[selective]` in `--list` (FSD50K, MIMII, Nonspeech7k) do not
download whole archives. The downloader reads the zip's table of contents
with HTTP Range requests and fetches only the clips whose labels map to one
of our classes, so FSD50K dev needs about 0.6 GB instead of 17 GB. Every file
is checked against the CRC stored in the archive. `--classes` limits FSD50K
to some classes; `--full` downloads the whole archive instead.

**Freesound** tops up classes the datasets leave short (Vehicle Horn). Get a
free API key at https://freesound.org/apiv2/apply/ and add
`FREESOUND_API_KEY=<key>` to `.env`. Queries and excluded tags are in
`config/freesound_queries.json`; only CC0, CC BY and CC BY-NC sounds are
kept, and the high-quality preview (lossy OGG) is downloaded. Listen to a
sample of the results before building.

Recommended per class:

| Class | Sources |
|---|---|
| Machinery Fault | `mimii-valve` (479 abnormal clips); add `mimii-pump` for variety |
| Normal Machinery (optional SRS class) | `mimii-valve-normal` (200), `mimii-pump-normal` (50), `mimii-fan-normal` (50); added because without it every model called normal machines "Machinery Fault" (see `models.md`) |
| Glass Breaking | `esc50`, `fsd50k-eval` (Shatter), MIVIA, own recordings |
| Alarm or Siren | `urbansound8k`, `esc50`, `fsd50k-eval` |
| Vehicle Horn | `urbansound8k`, `esc50`, `fsd50k-eval`, `fsd50k-dev`, `freesound`, own recordings |
| Animal Sound | `esc50`, `urbansound8k` |
| Gunshot | `urbansound8k`, `fsd50k-eval`, MIVIA |
| Panic Scream | `fsd50k-eval`, `fsd50k-dev` (Screaming), `nonspeech7k`, MIVIA, own recordings |
| Aggression | `ravdess`, CREMA-D, `fsd50k-eval` (Shout, Yell) |
| Person Asking for Help | own recordings + TTS only (fixed safety phrases) |
| Background Noise | `esc50`, `urbansound8k` (air conditioner), `fsd50k-*` (rain, wind, traffic, crowd, ...) |

Run `uv run python -m sonic.dataset scan` at any time to see how many
candidate files each class has so far.

## 2. Add your own recordings

Put files in `<data>/raw/custom/<class slug>/`, using the slugs from
`config/sound_classes.json` (`help_request`, `panic_scream`, ...). WAV, FLAC,
OGG and MP3 are read directly; convert phone `.m4a` files first:

```bash
ffmpeg -i input.m4a output.wav
```

Describe each file in `<data>/raw/custom/recordings.csv` (every column except
`path` is optional):

| column | example | meaning |
|---|---|---|
| path | `help_request/ali_01.wav` | relative to `<data>/raw/custom` |
| environment | `indoor, small room` | where it was recorded |
| device | `Samsung A52` | recording device |
| distance_m | `3` | approximate source distance |
| speaker | `ali` | who is speaking (speech classes) |
| group | `ali_session1` | files cut from one recording share a group, so they stay in one split |
| freesound_id | `123456` | set for Freesound downloads so duplicates across datasets are detected |
| license | `CC BY 4.0` | defaults to "own recording (team)" |
| source | `mivia` | defaults to `custom` |

**Help phrases:** record "Help me", "Somebody help", "Please help", "Call for
help" and "Emergency" from as many volunteers, devices, rooms and distances
as possible, only with each speaker's consent. Your own recordings are always
selected before public clips.

## 3. Build

```bash
uv run python -m sonic.dataset build
```

Options: `--per-class 300`, `--seed 42`, `--hardlink` (saves disk space; same
drive only), `--force` (rebuild over an existing dataset), `--sources ...`.

What the build does:

1. Maps each source's labels to our classes using `config/dataset_label_map.json`.
2. Drops recordings that two datasets label differently (ESC-50, UrbanSound8K
   and FSD50K all come from Freesound and overlap).
3. Per class, picks clips round-robin across original recordings, so 300
   clips come from as many different recordings as possible.
4. Rejects undecodable, empty, too-short (< 0.3 s), silent (peak < 0.001) and
   byte-identical duplicate files.
5. Splits each class 70/15/15 **by recording group**, so slices of one
   Freesound upload, one MIMII recording at three SNRs, or one actor's
   utterances never appear in both training and test.
6. Assigns Audio IDs (`GUN-00001`), copies files to
   `<data>/audio_dataset/<split>/<class slug>/`, and writes the metadata.

Outputs in `data/metadata/` (committed to git):

- `dataset_metadata.csv`: one row per clip (see `data_dictionary.md`)
- `dataset_statistics.json`: counts per class and split, sources, warnings
- `rejected_files.csv`: every rejected file and why

The same seed and the same raw data always give the same dataset.

## 3b. Person Asking for Help (synthetic) and adding clips later

No public dataset has this class. Following the SRS ("recorded voluntarily,
generated synthetically where permitted, or obtained from ethically licensed
sources"), and with the organisers' written permission, the team generated it
with Microsoft Edge text-to-speech (edge-tts):

```bash
uv run python -m sonic.dataset tts
uv run python -m sonic.dataset extend help_request background_noise
```

- **300 Help clips**: all 47 English voices (14 accents, 24 female, 23 male)
  say each of the five SRS phrases, plus 65 repeated calls ("Help me! Help
  me!"). Each clip has its own speaking rate, pitch and loudness; most are
  faster, higher and louder (urgent), about 15% calmer. 0.3 s of silence is
  added before and after, like a real recording.
- **47 ordinary sentences** (one per voice, e.g. "Have you seen my keys
  anywhere?") are added to Background Noise. Without them, clear
  single-voice speech would only ever mean Help, and a model could learn
  "a voice" instead of "a call for help" (the SRS's help request vs.
  ordinary speech confusion).
- **One voice = one group**: all clips of a voice are in one split. The 45
  Help test clips are spoken by 7 voices never used in training or
  validation, so the test result measures unseen voices.
- Every clip is marked in `dataset_metadata.csv`: source `edge-tts`, the
  voice as speaker, the text, rate, pitch and volume in the notes.

**Limitation:** synthetic voices are clean and less panicked than a real
person calling for help, and the test voices are synthetic too. Real
recordings from volunteers would give a more realistic test; the recording
guide is in `help_recordings.md`.

`extend` appends new `custom/` clips to the built dataset without changing
any clip already in it. A full `build` would redraw the selection and split
of the classes after Help (Background Noise, Normal Machinery), moving clips
GTM was trained on into the test set. New clips get the next Audio IDs of
their class (HLP-00001..., BGN-00301...).

## 4. Rules to keep (SRS §Hint)

- Augmented copies are made later from `train` clips only, keep their
  parent's Audio ID in `parent_audio_id`, and never count as originals.
- GTM samples must be cut from the same `train` recordings; never upload
  `validation` or `test` audio to Teachable Machine.
- Compare both models on the same `test` split.
