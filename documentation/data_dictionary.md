# Data dictionary: `data/metadata/dataset_metadata.csv`

One row per original clip in the built dataset.

| Column | Type | Description |
|---|---|---|
| audio_id | text | Unique Audio ID, `<class code>-<5 digits>`, e.g. `GUN-00012`. Segments and augmented copies derived from this clip keep it. |
| filename | text | Path inside `<data>/audio_dataset/` (see SONIC_DATA_DIR), e.g. `train/gunshot/GUN-00012.wav` |
| class_name | text | Sound class, exactly as in `config/sound_classes.json` |
| class_slug | text | Folder-safe class name |
| split | text | `train`, `validation` or `test` |
| source_dataset | text | `esc50`, `urbansound8k`, `fsd50k`, `mimii`, `crema_d`, `ravdess`, `musan`, `nonspeech7k`, `freesound`, `custom` or the `source` given in `recordings.csv` |
| source_label | text | The label the source dataset used |
| source_path | text | Path of the original file inside `<data>/raw/` |
| original_filename | text | Original file name |
| origin_id | text | Original recording the clip came from; the SRS counts unique `origin_id`s as unique original clips |
| group_id | text | Split group: all clips of a group are in the same split (one Freesound upload, one MIMII recording at three SNRs, one actor) |
| freesound_id | text | Freesound sound ID, when known |
| license | text | License of the clip |
| format | text | Container format (WAV, FLAC, OGG, MP3); Freesound previews are lossy OGG |
| duration_s | number | Duration in seconds |
| sample_rate | integer | Sampling rate in Hz |
| channels | integer | Number of channels |
| bit_depth | integer | Bits per sample, blank for compressed formats |
| file_size_bytes | integer | File size |
| sha256 | text | SHA-256 of the file bytes, used for duplicate detection |
| peak_amplitude | number | Peak absolute sample value (0 to 1) |
| quality_grade | text | SRS Step 13 grade: Good, Acceptable or Poor (Unusable clips are rejected); see `documentation/quality.md` |
| quality_issues | text | Why the grade is not Good, e.g. `clipping: 0.8% of samples are clipped` |
| environment | text | Recording environment, `unknown` when the source does not say |
| device | text | Recording device, `unknown` when the source does not say |
| distance_m | number | Approximate source distance in metres, blank if unknown |
| speaker | text | Speaker ID for speech classes |
| is_augmented | boolean | Always `False` here; augmented copies are listed separately |
| parent_audio_id | text | For derived/augmented audio: Audio ID of the original |
| notes | text | Source-specific details (salience, machine ID, SNR, sentence) |
