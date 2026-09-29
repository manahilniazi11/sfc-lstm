# Decision-threshold study

Model `ensemble-20260928-1`, 502 validation clips (clip decision as in the app). Accuracy 86.3%, median top confidence 0.93.

## Top-class confidence

| Threshold | Clips at or above | Accuracy at or above | Accuracy below |
|---|---|---|---|
| 0.30 | 99% | 87.1% | 0.0% |
| 0.35 | 99% | 87.3% | 14.3% |
| 0.40 | 98% | 87.6% | 33.3% |
| 0.45 | 96% | 88.6% | 33.3% |
| 0.50 | 94% | 89.6% | 31.0% |
| 0.55 | 90% | 90.7% | 43.8% |
| 0.60 | 88% | 91.1% | 51.6% |
| 0.70 | 82% | 93.2% | 54.9% |
| 0.80 | 76% | 95.0% | 59.0% |

## Top-two margin

| Threshold | Clips at or above | Accuracy at or above | Accuracy below |
|---|---|---|---|
| 0.05 | 99% | 87.1% | 16.7% |
| 0.10 | 97% | 87.7% | 30.8% |
| 0.15 | 95% | 88.5% | 39.1% |
| 0.20 | 94% | 89.2% | 40.0% |
| 0.30 | 88% | 91.6% | 45.8% |

## Critical classes: share of real events that would reach an alert threshold

| Class | Recall (any confidence) | at >= 0.45 | at >= 0.50 | at >= 0.60 | at >= 0.70 |
|---|---|---|---|---|---|
| Gunshot | 96% | 93% | 91% | 84% | 80% |
| Glass Breaking | 100% | 98% | 96% | 91% | 89% |
| Panic Scream | 82% | 80% | 80% | 69% | 62% |
| Aggression | 73% | 71% | 71% | 67% | 58% |
| Person Asking for Help | 98% | 98% | 98% | 98% | 98% |

## Chosen values (alert_rules.json)

- `min_confidence` 0.50: above it clips are clearly more reliable; below it the model is right only about half the time, so those results go to manual review.
- `unknown_confidence` 0.35: below it the model is mostly wrong, so the sound is reported as Unknown.
- `top_two_margin` 0.10: below it accuracy drops sharply (two classes are almost tied).
- Panic Scream and Aggression alert at 0.45: their confidences are lower than the other critical classes, and a missed scream is worse than a reviewed false alarm.
