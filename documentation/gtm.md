# Google Teachable Machine (GTM) audio model

SRS Steps 9–10: a separately trained GTM audio model, with the same class
names as the Python model, trained on samples from **the same training
recordings**.

## Why the audio is "played" into GTM

GTM audio projects cannot import audio files: samples can only be recorded
through a microphone in the browser (GTM then turns each 1 s sample into
its own frequency frames). So the training audio is **played into a
virtual microphone** while GTM records. The sound stays digital (no speaker,
no room), and every sample comes from our training split.

`python -m sonic.gtm export` builds the playback files in
`<data>/gtm_export/` (`E:\sonic_data\gtm_export` on the development PC):

- one WAV per class, 150 s long: 150 one-second **training** segments back
  to back, taken from as many different recordings as possible
- `manifest.csv`: which segment and Audio ID plays at which second of which
  file (evidence that GTM used the same training recordings)

Validation and test audio are never exported, so both models are later
compared on the same unseen test recordings (SRS Hint).

## One-time setup: a virtual audio cable

Pick **one** option.

**A. VB-Audio Virtual Cable** (free, recommended): install from
https://vb-audio.com/Cable/ (run the installer as administrator, then reboot).
It adds a playback device *CABLE Input* and a recording device *CABLE Output*.

**B. Stereo Mix** (no install, if your sound card has it): Control Panel →
Sound → Recording → right-click → *Show Disabled Devices* → enable
*Stereo Mix*. Stereo Mix records whatever the PC plays, so mute
notifications and other audio while recording.

## Recording the samples

1. Open https://teachablemachine.withgoogle.com/train/audio in Chrome.
2. When the browser asks for the microphone, choose **CABLE Output** (or
   *Stereo Mix*). To change it later: the camera/mic icon in the address bar.
3. Create the classes with **exactly** these names (they must match
   `config/sound_classes.json`):
   `Background Noise` (GTM's built-in first class), `Machinery Fault`,
   `Glass Breaking`, `Alarm or Siren`, `Vehicle Horn`, `Animal Sound`,
   `Gunshot`, `Panic Scream`, `Aggression`, `Normal Machinery`, and later
   `Person Asking for Help`.
4. For each class:
   - Set Windows' playback device to **CABLE Input** (Settings → System →
     Sound → Output), or with Stereo Mix leave your normal speakers.
   - Open the class's WAV (e.g. `gunshot.wav`) in a media player, ready to play.
   - In the GTM class, click **Mic**, open the settings (gear), set
     **Duration** to `145` seconds and click *Save Settings* (the setting
     is per class). 145 s, a little shorter than the 150 s file, so the
     recording ends before playback does and no silent samples are added.
   - Start playback, then immediately click **Record**. GTM keeps each
     second as one sample. Delete any black (silent) samples afterwards.
   - **Keep the Chrome window visible** (e.g. snapped to half the screen)
     while recording. When the window is covered or minimized, Chrome
     reports the page as hidden and pauses its drawing loop; GTM then
     captures only the first few seconds (seen on 2026-09-26).
   - Aim for **all 150 samples** per class; GTM requires at least 20 for
     Background Noise and 8 for the others.
5. Click **Train Model**. Keep the tab in front until training finishes.
6. Test with **Preview** by playing a few *test* clips (from
   `<data>/audio_dataset/test/`) the same way. Screenshot the results.

## Export and hand-over

1. **Export Model → TensorFlow.js → Download**. Put `model.json`,
   `weights.bin` and `metadata.json` in the repository's `gtm_model/` folder
   (the web app loads it from there).
2. **Save the project** (the menu next to the title → *Save project to
   Drive* or *Download project file*) so it can be retrained later.
3. Screenshots for the report (SRS evidence list): every class with its
   sample count, the training settings, the Preview results including wrong
   predictions.
4. Note in the development log: date, samples per class, training settings
   (epochs, batch size, learning rate from *Advanced*), anything unusual.

## Training record: version 1 (2026-09-27)

Recorded through **Stereo Mix** (Realtek) in Chrome. The recording was
automated: PowerShell played each class's WAV (`System.Media.SoundPlayer`)
while a script clicked *Record* in Teachable Machine and afterwards removed
silent samples (the second before playback started, and two Glass Breaking
windows that fell between two breaks, peak below −60 dB).

| Class | Samples |
|---|---|
| Background Noise | 145 |
| Machinery Fault | 145 |
| Glass Breaking | 143 |
| Alarm or Siren | 144 |
| Vehicle Horn | 145 |
| Animal Sound | 145 |
| Gunshot | 144 |
| Panic Scream | 144 |
| Aggression | 144 |
| Normal Machinery | 144 |

Training settings: GTM defaults (50 epochs, batch size 16, default learning
rate). Person Asking for Help is missing until its recordings exist.

**Observations ("Under the hood"):** training accuracy rose to about 0.90,
while accuracy on GTM's own held-out samples levelled off at about **0.46**
from epoch 20 on (test loss flat at about 1.65). That is far above chance
(0.10 for 10 classes) but clearly weaker than the Python models, and the gap
between the two curves shows over-fitting. Likely reasons: GTM's audio model
is a small network pre-trained for spoken commands, it sees only about 145
one-second samples per class, and its only training option is the number of
epochs. The fair comparison with the Python model is on the same unseen test
recordings, in the web app.

Problems met:
- On the first attempt (2026-09-26) only ~4 s of a 145 s recording were
  captured, because the Chrome window was covered (page hidden).
- During Alarm or Siren the PC's output was switched away from the Realtek
  speakers and muted, so Stereo Mix received silence for ~30 s. The class
  was cleared and recorded again; a level check now runs before recording.

Export: TensorFlow.js, `gtm_model/` (`model.json`, `weights.bin`,
`metadata.json`; speech-commands 0.4.0, model name TMv2). Labels in the
export are sorted alphabetically; the web app maps them by name, never by
position.

## Keeping the two models independent (SRS 1.8 items 13-14)

The GTM model is trained only in Teachable Machine on its own samples, and
in the web app it receives only the audio window, never the Python model's
prediction or confidence. The comparison happens after both have predicted.
