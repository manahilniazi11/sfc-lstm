# Recording guide: Person Asking for Help

No public dataset contains this class, so the team records it (SRS Hint:
"recorded voluntarily"). Target: **300 unique recordings**, from as many
different people, phones, rooms and distances as possible.

## Consent (do this first)

Before recording anyone, tell them: what the recordings are for (a student
competition sound-detection project), that the clips are stored in the
project dataset and may be submitted with it, and that they can ask for
their clips to be deleted. Record only people who agree, and note their
first name or a nickname only. No recordings of anyone who did not agree,
and no children without a parent's consent.

## What to record

The SRS limits this class to fixed safety phrases:

| Phrase | File-name word |
|---|---|
| Help me | `helpme` |
| Somebody help | `somebodyhelp` |
| Please help | `pleasehelp` |
| Call for help | `callforhelp` |
| Emergency | `emergency` |

Say them as in a real emergency: urgent, loud, some shouted, some breathless
or panicked. Also a few calmer ones: the model should still catch a
quieter "help me".

## How many

A plan that reaches 300:

**15 people × 5 phrases × 4 takes = 300**, where the 4 takes of each phrase differ:

| Take | Distance | Place |
|---|---|---|
| 1 | close (about 0.5 m) | quiet room |
| 2 | far (3–5 m) | quiet room |
| 3 | close | noisy place (kitchen, street, TV on) |
| 4 | far or another room | a room with echo (hall, bathroom, stairwell) |

More people with fewer takes each is **better** than fewer people with many
takes: the test set is judged on voices the model has never heard (each
person's recordings stay in one split).

## How to record

- Any phone's voice-recorder app. Different phones is a plus.
- One phrase per file, about **1–3 s** with a little silence before and after.
- Don't clip: if the recording distorts when shouting, step back.
- Record the take, listen once, keep it if the phrase is clearly audible.

## File names

`<person>_<phrase>_<place>_<distance>m_<phone>_<take>.<ext>`

Examples:
`ali_helpme_bedroom_0.5m_redmi9_1.m4a`,
`sara_emergency_street_4m_iphone13_3.m4a`

Use one word per part (no spaces or extra underscores). The importer
reads the person, place, distance and phone from the name.

## Adding them to the dataset

1. Copy the files into `E:\sonic_data\raw\custom\help_request\`.
2. Run:

   ```bash
   uv run python -m sonic.dataset recordings
   ```

   It converts phone formats (.m4a, .aac, .opus, …) to WAV with ffmpeg, and
   adds one row per file to `raw/custom/recordings.csv` (person, place,
   distance, phone, phrase). Each person becomes one split group.
3. Check `recordings.csv` and fix anything the name did not say.
4. Rebuild the dataset, then preprocessing, augmentation and features, and
   retrain (see the README). Our own recordings are always picked before
   public clips.

The same steps work for any class: put files in its folder (e.g.
`panic_scream/`, `vehicle_horn/`); own recordings add the device and
distance variety the hidden tests check for.
