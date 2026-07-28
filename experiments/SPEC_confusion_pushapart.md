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

## ~~The non-redundant version~~ — ALSO DEAD (2026-07-28, killed by inspection)

**The premise below is FALSE and the whole method dies with it. Do not build this.**

> ~~The gap EP leaves: when the sample is already classified correctly, the error is ~0 and EP does
> nothing — even if the sample sits a hair from the 9/3 boundary. EP has no notion of *margin*.
> So the mechanism is preventive margin pressure along known-confusable directions: for a sample of
> class `c` that is currently CORRECT, apply a small extra negative nudge to `d*` = argmax `L[c][·]`.~~

**Why it is false.** EP's nudge (`equilibrium_prop.py:66`) is `β(y_target − y)` — a *soft pull toward
the one-hot target*, NOT an error-gated correction. For competitor `d` (target 0) the force is
`−β·y[d]`, applied on EVERY sample regardless of correctness; EP keeps pushing until `y` exactly
equals one-hot. Working it through: **`ΔW2[d] ∝ − y_free[d] · ρ(h)`**. EP already applies margin
pressure, per-sample, proportional to activation.

And `L[c][d]` is *literally the class-average of `y_free[d]`* — a lagging, averaged copy of the exact
quantity already in the update. Worse, it is **mis-targeted**: on an unambiguous 9 (unit 3 at 0.05)
the class map pushes hard where there is nothing to fix, while the 9/3 boundary lives in the
ambiguous samples where `y_free[3]` is high and EP already acts hardest. **Per-sample strictly
dominates class-average.**

**The specified control was also broken:** the random-guidance arm picks a non-confusable competitor
whose activation ≈ 0, so its update is ≈ no-op. "Matched nudge magnitude" ≠ matched effect;
confusion-guided would have beaten it trivially by doing something vs nothing. A win would have
proved nothing.

Steelmen considered and rejected: (a) with a *hinge/margin* target EP would be inert on safe samples
— but then the fair baseline is "EP with a hinge target", not this method; (b) the class average has
lower variance than a single sample — but averaging blurs the instance-specificity that makes the
correction land where the boundary is.

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

**KILLED 2026-07-28 by reasoning, before any compute.** Both versions (naive suppression, preventive
margin pressure) are redundant with EP's own per-sample update. Zero GPU time spent.

## The pattern this completes — a general negative principle

Four method-directions have now died the SAME death:

| direction | what we tried to inject | what already did it |
|---|---|---|
| learned graded payload (Exp 20/22) | a learned per-spike payload | the readout absorbs it; membrane (computed) carries the real info |
| energy scheduling (Exp 19a/b) | basin depth as forgetting signal | accuracy measures correctness directly; energy measures confidence |
| cost / MIR scheduling (Exp 21, 25) | which-class-to-replay priority | nothing to exploit — forgetting is *proportional*, not selective |
| confusion-guided pressure (this) | class-level leakage map | EP's per-sample nudge IS that quantity, better targeted |

**Principle: an error-proportional local rule already implements the optimal per-sample version of
any aggregate statistic computed from its own outputs. Injecting that statistic back is redundant at
best, mis-targeted at worst.** Corollary — stop proposing re-weightings of the existing error signal.

**Where that leaves room** (things that are NOT re-weightings):
1. **Add DATA the rule does not have** — replay/recall. Note this is the *one* thing that has worked
   all week (Exp 17), and the principle explains why: it supplies gradient signal that does not exist,
   rather than re-scaling signal that does. **Generative recall** (settle-to-generate old classes) is
   the coherent next target.
2. **Add CAPACITY / structure** — separate pathways per confusable group (parameter isolation).
   Escapes the principle, but is a known technique and a bigger build.
3. **Change what is REPRESENTED** (features), not how the readout is corrected.
