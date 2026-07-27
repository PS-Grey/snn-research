# Pre-registration: confusion-guided margin pressure (unbuilt)

Written **before** building, per the mechanism-first rule. Nothing here is a result.

## Signal (established, Exp 24)

`L[c][d]` = mean activation of output unit `d` over samples of true class `c` ("leakage").
Predicts where errors actually go at **r = 0.72**, robust over 3 seeds. Directional: `L[9][3]` is
large while `L[3][9]` is not (Exp 23: symmetry r = 0.05). Available during EP's nudged phase, since
the label is present there. Accumulated over samples, so it is a *class-level, slow* signal.

## The naive version is ALREADY DEAD — killed by reasoning, not compute

Naive proposal: "push harder on the competitor that class c leaks into most."

EP's readout update is `ΔW2 ∝ ρ(y_nudged)ρ(h) − ρ(y_free)ρ(h)`. With a one-hot target, a competitor
`d` that is strongly active in the free phase has further to fall, so it **already receives a
correction proportional to its activation** — which is exactly the leakage. So targeted suppression
re-derives what EP already does per-sample. **Redundant. Do not build it.**

(This is the same trap as Exp 20's learned payload, where the global readout absorbed the new
channel, and the same as the old notes' "study the error harder" backfire.)

## The non-redundant version — the only one worth testing

The gap EP leaves: **when the sample is already classified correctly, the error is ~0 and EP does
nothing** — even if the sample sits a hair from the 9/3 boundary. EP has no notion of *margin*.
But the leakage map knows 9 systematically threatens 3 *across the class*, including on samples
where 9 currently wins.

So the mechanism is **preventive margin pressure along known-confusable directions**:
for a sample of class `c` that is currently CORRECT, apply a small extra negative nudge to the
specific competitor `d*` = argmax `L[c][·]`, proportional to `L[c][d*]`, scaled by how *narrow* the
current margin is. Nothing is applied to already-safe samples or to non-confusable competitors.

This is not what EP does (EP is inert when correct) and not uniform regularisation (it is directional
and class-specific). That is the claim.

## Falsifiable prediction

At matched total nudge strength, versus plain EP:
- **accuracy on the top-5 confused pairs improves**, and
- the gain is *concentrated* there, not spread evenly (a spread gain means we just regularised).

## Controls the design MUST carry (both learned the hard way this week)

1. **Random-guidance arm** — same extra nudge magnitude, same number of samples touched, but the
   competitor `d` chosen at RANDOM instead of from the leakage map. This is the arm that decides it.
   Beating plain EP proves nothing (extra pressure ≈ regularisation); beating *random guidance* is
   the only thing that proves the *confusion structure* is doing the work. Cf. Exp 22, where
   `learned` lost to a random `fixed` payload.
2. **5 seeds, variance bars, from line one.** Cf. Exp 21, where a +15pp single-seed "win" was RNG.

## Simplest ways it fails (state now, so we cannot rationalise later)

- **F1 (most likely).** Margin pressure is just label smoothing / weight decay in a fancy hat.
  Symptom: beats plain EP, ties random-guidance. That is a null, and I will call it one.
- **F2.** Pushing `c` off `d` damages `d` itself, since confusable classes share hidden features.
  Symptom: hot-pair accuracy rises, the competitor's own accuracy falls, net zero.
- **F3.** The leakage map is a lagging class-level average; the boundary may move faster than it
  updates. Symptom: helps early in training, decays or hurts later.

## Status

UNBUILT. Do not run until the design (especially control 1) is agreed.
