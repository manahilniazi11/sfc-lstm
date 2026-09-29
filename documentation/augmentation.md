# Augmentation

`python -m sonic.augmentation build` creates augmented copies of **training
segments only** (settings: `config/augmentation.json`). Each copy keeps its
parent's Audio ID and split, is listed in `data/metadata/augmented_segments.csv`
with `is_augmented = True`, and never counts as a unique original clip (SRS Hint).

## Why

1. **Class balance.** Every class is topped up to 1,500 training segments,
   so small classes (Gunshot: 376 originals) get more copies (about 3 each)
   than large ones (Machinery Fault: 1,050 originals, under 1 each).
2. **Backgrounds belong to no class.** Every copy is mixed with background
   noise drawn from one shared pool: Background Noise training segments plus
   200 MIMII *normal-operation* factory recordings. Every Machinery Fault
   clip comes from MIMII with factory noise in it; mixing factory noise into
   every other class stops "factory hum" from meaning "fault".
3. **Hidden-test robustness.** The SRS hidden tests use noise, echo, low
   volume, other devices and distant sources: noise, reverb, gain, device
   and distance transforms imitate exactly these.

## Transforms

A copy applies 2–3 transforms, always including background noise, in this
physical order (source → room → environment → microphone):

| Transform | Range | Imitates |
|---|---|---|
| Time shift | ±0.3 s | the event falling anywhere in a live window |
| Pitch shift | ±2 semitones | different voices, engines, sirens |
| Time stretch | 0.85–1.15× | faster or slower events |
| Reverb | RT60 0.2–0.8 s, 10–40 % wet | rooms and halls (synthetic impulse response) |
| Distance | low-pass 2–6 kHz, extra reverb, −3 to −8 dB | a far-away source |
| Gain | −6 to +6 dB | louder or quieter source |
| Background noise | SNR 3–20 dB | noisy environments |
| Device | band-pass 300–3400 / 100–7000 / 200–5000 Hz, soft clipping | phones and cheap microphones |

Noise is mixed relative to the segment's level *before* gain and distance,
so a quieter or distant source really ends up closer to the noise. Limits
keep the worst case around −5 dB SNR so the event stays audible. Each copy
finishes exactly like preprocessing (normalized to −20 dBFS plus the
−70 dBFS noise floor) and is reproducible from its segment ID.

## Result

| Class | Originals | Copies | Training total |
|---|---|---|---|
| Aggression | 669 | 831 | 1,500 |
| Alarm or Siren | 751 | 749 | 1,500 |
| Animal Sound | 549 | 951 | 1,500 |
| Background Noise | 906 | 594 | 1,500 |
| Glass Breaking | 383 | 1,117 | 1,500 |
| Gunshot | 376 | 1,124 | 1,500 |
| Machinery Fault | 1,050 | 450 | 1,500 |
| Panic Scream | 440 | 1,060 | 1,500 |
| Vehicle Horn | 569 | 931 | 1,500 |

The noise pool held 1,906 segments; 53 % of copies were mixed with factory
noise, 47 % with other backgrounds. A background clip is never mixed with itself.

**To verify during model evaluation:** run the trained model on MIMII normal
recordings that were *not* in the noise pool. A model that learned the fault,
not the factory, should not call them Machinery Fault.
