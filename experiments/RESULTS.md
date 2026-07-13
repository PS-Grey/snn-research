# Baseline results — gradient-SNN vs matched ANN

Pure surrogate-gradient SNN (conv-BN-LIF, atan surrogate, subtract reset, **no membrane
noise**) against a parameter-matched ANN sharing an identical conv backbone. The point is a
fair SNN↔ANN gap under modern gradient training, replacing the legacy membrane-noise regime.

Script: `baseline_gradient_snn.py`. Device: MPS (M2). Both models 50,378 params.

| Dataset | Epochs | T | ANN best | SNN best | Gap (pp) | SNN spikes/img |
|---|---|---|---|---|---|---|
| MNIST | 5 | 10 | 99.13% | 98.89% | 0.24 | ~22,700 |
| Fashion-MNIST | 10 | 10 | 92.30% | 92.41% | **−0.11** | ~26,000 |

## Timestep (T) sweep — accuracy vs energy (Fashion-MNIST, 10 epochs each)

Script: `timestep_sweep.py`.

| T | Test acc | Spikes/img |
|---|---|---|
| 1 | 91.60% | 1,892 |
| 2 | 90.85% | 5,078 |
| 4 | 92.15% | 9,604 |
| 8 | 92.43% | 19,498 |

Accuracy is flat across T (90.8–92.4%) while spikes scale ~linearly with T. T=1→T=8 buys
+0.83 pp for 10x the spikes. On a static frame there is no temporal structure to integrate
(extra steps just re-present the same image), so **T=1 is the deployment operating point**:
~1,900 spikes/img at 91.6%. The temporal dimension only earns its cost on genuinely temporal
input (DVS / event vision), which is where higher T should pay off.

## Surrogate-width schedule — fixed vs annealed atan slope (Fashion-MNIST, T=10, 10 epochs)

Script: `surrogate_schedule.py`. Tests the 2024-25 literature claim that scheduling the
surrogate slope (wide early for gradient flow, sharp late to cut gradient mismatch) improves
direct training. Identical seed/init/batch order across conditions.

| Condition | Best test | Spikes/img (final) |
|---|---|---|
| anneal α 2→16 (linear/epoch) | 92.37% | ~23,400 |
| fixed α=16 | 92.45% | ~25,900 |
| fixed α=2 (baseline regime) | **92.67%** | ~24,100 |

All three within 0.30 pp (seed noise); the baseline nominally wins. Re-confirmation, not
discovery: the legacy sigmoid-surrogate scale sweep (vault, 30 May) already showed the smooth
band is a flat plateau (flat SNN 89.4% @ scale=1, conv 92.5% @ scale=5) with the drop-off only
at the sharp extreme (scale 25→100 → ~77-81%). The atan α 2→16 range sits on that plateau, so
a null is expected.

**Parked**: surrogate width is a non-lever within the atan family at this depth; scheduling is
a depth-scaling tool, revisit when the net is deep enough to show gradient attenuation.

## Surrogate tail shape — tested head-to-head, NOT load-bearing at 2-conv depth

Script: `surrogate_tail.py`. Tests whether the surrogate's *tail* (not its width) is the
load-bearing variable, motivated by the old sigmoid collapse. atan (Lorentzian tails,
`1/(1+x²)`) vs true logistic sigmoid (exponential tails), **matched at equal gradient
half-width** so only the tail differs, then both sharpened. Fashion-MNIST, T=10, 6 epochs,
seed 0.

| half-width | atan best | sigmoid best | atan tail@0.3 | sig tail@0.3 |
|---|---|---|---|---|
| 0.35 (smooth) | 92.17% | 92.27% | 0.58 | 0.59 |
| 0.10 | 92.23% | 91.92% | 0.10 | 0.02 |
| 0.05 | 92.09% | 91.66% | 0.027 | 0.0001 |
| 0.02 (sharp) | 91.66% | 91.35% | 0.0044 | **0.0000** |

**Hypothesis falsified at this depth.** Both families hold ~91–92% across the whole range. At
hw=0.02 the sigmoid's off-threshold gradient is literally zero (tail@0.3 = 0.0000) and it still
reaches 91.35%, tracking atan epoch-for-epoch. Starving off-threshold neurons of gradient does
not cause collapse at 2-conv depth. There is a whisper in the tail direction (sigmoid slips
from +0.10 to −0.4 behind atan as both sharpen) but it is within the ±0.3 pp seed noise.

Consequence: the old 30-May sigmoid collapse (76–81% at sharp scale, 3-conv net) is
**unreproduced here** with matched-width surrogates of either tail. Since atan and sigmoid are
interchangeable at 2 layers, the old collapse was not the surrogate tail — the remaining
suspect is **depth** (3 conv layers vs 2; gradient must propagate further). Ties back to the
depth-scaling experiment.

Gotcha found: snnTorch's stock `surrogate.sigmoid` computes its backward with a bare `exp()`
that overflows to inf→NaN for neurons more than ~20 below threshold — routine in a BN
conv-SNN — silently pinning training at chance regardless of slope. Use a `torch.sigmoid`-based
surrogate (as here / the 30-May script). This bug invalidated a first run of this experiment.

## Backprop-free: unsupervised STDP baseline + noise (post-pivot, 2026-07-09)

Script: `stdp_diehl_cook.py`. First experiment of the backprop-free direction — MNIST learned
with a local, on-chip-implementable rule only (trace-STDP + winner-take-all + adaptive-threshold
homeostasis + weight normalisation). No backprop, no labels during learning; neurons labelled
post-hoc by response. CPU (tiny per-step tensors beat MPS launch overhead).

**Mechanism confirmed.** Unsupervised STDP learns digit-selective neurons and classifies well
above chance:

Getting-it-working progression (100 neurons, no ramping → the config below), backprop-free
throughout. Two levers took it from ~45% to the mid-80s:
1. **Intensity ramping** (+18 pp alone at 100 neurons) — D&C's trick of re-presenting an image
   at higher intensity until it drives ≥5 spikes. Without it, thin low-pixel digits never won
   neurons, capping accuracy via class imbalance. Raising `max_tries` 4→6 also cut silent-eval
   images (a direct auto-fail) from ~13% to ~2% (0% by 1600 neurons).
2. **Scale + revisiting** — more neurons and a 2nd pass sharpen and spread the receptive fields.

**Scaling curve (30k imgs × 2 passes, T=100, ramping, theta_plus 5e-4):**

| neurons | 100 | 400 | 800 | 1600 | 3200 |
|---|---|---|---|---|---|
| test acc | ~46% | 70.5% | 86.04% | **88.68%** | 88.58% |

**Pure STDP plateaus at ~88.6%.** Doubling neurons 1600→3200 buys nothing (−0.1 pp); the
~10 pp gap to the surrogate reference (98.9%) is **not** closable by adding neurons. Mechanism:
unsupervised STDP has no signal to allocate neurons by *class usefulness*, so extra capacity
piles onto already-popular prototypes — at 3200, class 2 hogs 729 neurons while class 1 is
starved at 110. This is exactly the failure a global reward signal (three-factor / R-STDP) is
meant to fix, so the three-factor experiment is now empirically motivated: **neurons are
exhausted as a lever at ~88.6%.**

All primitives here are on-chip-implementable, with two caveats flagged for deployment:
weight-normalisation and hard-WTA are not cleanly local.

Getting-it-working diagnostics (all specific failure modes hit and fixed): (1) weak
same-timestep lateral inhibition → **WTA collapse**, all neurons learn one shared "mean digit"
→ fixed with single-winner-per-timestep WTA; (2) `theta_plus` too large vs input drive →
homeostasis **silences neurons** by eval (0 spikes/img) → fixed by tuning `theta_plus` to the
drive scale (sweet spot 5e-4: 15 spikes/img, all classes). Bounding the pre-trace to [0,1]
*hurt* (33.7% vs 46.3%): the larger unbounded-trace updates concentrate weights more sharply
and drive firing harder.

**Noise result — membrane noise HURTS STDP** (opposite of the surrogate-gradient / legacy
regimes where noise helped). 100 neurons, seed 0:

| membrane noise σ | test acc | eval spikes/img |
|---|---|---|
| 0.0 | 46.3% | 15.1 |
| 0.05 | 35.9% | 7.1 |
| 0.10 | 35.3% | 6.1 |
| 0.20 | 32.4% | 5.5 |

Robust across seeds (noise 0.1 → 35.7% / 35.1% at seeds 1–2 vs 42.8 / 46.9 no-noise). Mechanism:
in hard-WTA STDP the learning signal **is** the winner selection (`argmax` of membrane), so
membrane noise corrupts which neuron wins → neurons get STDP updates for images they don't match
→ misattributed credit blurs their receptive fields → lower drive, fewer spikes, lower accuracy.
The surrogate regime tolerated/liked noise because there the gradient channel is separate from
and robust to it; here noise attacks the credit-assignment mechanism directly. **Deployment
implication:** intrinsic device noise on neuromorphic hardware could degrade hard-WTA STDP unless
the winner-selection is made noise-robust — worth carrying into the on-chip direction.

## Three-factor R-STDP readout — reward beats neurons (2026-07-09)

Script: `stdp_three_factor.py`. Adds the missing third factor to the plateaued pure-STDP net: a
global reward signal gating a local update (the Loihi-supported, on-chip form of goal-directed
learning). Frozen 800-neuron unsupervised-STDP features; a 10-class spiking readout trained by
reward-modulated STDP (rate form: predict by argmax, and on error push the correct class's
synapses up and the wrong winner's down — reward-gated pre×post Hebbian). No backprop.

| approach | neurons | test acc |
|---|---|---|
| pure STDP (unsupervised) | 800 | 86.04% |
| pure STDP ceiling | 3200 | 88.58% |
| **three-factor R-STDP** | **800** | **89.18%** |
| surrogate-gradient (ref) | — | 98.9% |

**Two wins, both predicted by the scaling-curve plateau:**
1. +3.1 pp over pure STDP on the *same* 800 features — the reward signal adds real value the
   unsupervised rule couldn't extract.
2. Beats the pure-STDP *3200-neuron* ceiling (88.58%) with only 800 neurons — **one global
   reward signal outperforms 4× the neurons.** This is the on-chip efficiency payoff: reward
   substitutes for neuron/synapse count, the expensive resource on hardware.

The ~10 pp gap to the surrogate reference remains — the frozen features cap it (readout train
error plateaus ~7.7%, so the linear-separability of the unsupervised features is the limit, not
the readout). Closing more needs the reward to shape the *features* too (both-layers-plastic,
deferred follow-up), or better features.

Caveat: this is the **rate form** of R-STDP (feature spike-counts → reward-gated Hebbian), not a
time-resolved eligibility-trace output layer. A spike-faithful version (`stdp_three_factor_spiking.py`:
LIF output neurons, temporal eligibility traces, reward-gated updates) was **attempted but not
confirmed** — across four design iterations (no-WTA / output-WTA / teacher-forcing) it landed
45–55% vs the rate form's 69% on the same weak features. The spiking WTA+eligibility+reward
readout has its own cold-start / WTA-monopoly / teacher-balance tuning problem (a mini D&C). So
the rate-form 89.18% stands, but **its spike-faithful equivalence is unconfirmed** — parked
(2026-07-10) in favour of the both-layers-plastic direction.

### Both-layers-plastic (reward shapes the features too) — WIP, unfinished

`stdp_both_plastic.py`. Lets the global reward reshape the feature layer via reward-modulated
STDP (each winning feature neuron accumulates an STDP eligibility; reward gates it — correct
reinforces, wrong pushes away). Protocol: unsup features → baseline readout, then fine-tune
features with reward, then fresh readout → compare.
- **Naive version collapsed** (baseline 79.4% → 31.3%, −48 pp on a small config). Mechanism:
  early on the co-adapting readout is unreliable, so most images read as "wrong" → `R=−1` →
  anti-Hebbian updates dominate → receptive fields scramble. A documented instability of
  reward-modulated feature STDP when negative rewards dominate.
- **Frozen-critic fix** (use the good baseline readout as a stable ~80%-correct reward source →
  mostly reinforcing) was implemented but the run stalled and was never completed. **Untested.**
- Status: parked unfinished (2026-07-13). Next: complete the frozen-critic run; if still
  unstable, add reward baselining (advantage) or correct-only reinforcement.

## Reading

- The legacy figure of **80.7% as the best pure SNN on MNIST** is an artefact of the old
  regime (membrane noise + bio-inspired stack at small scale), **not a ceiling**. With
  surrogate gradients and BatchNorm from the start, the pure SNN sits within a quarter of a
  point of its ANN on MNIST and ties (edges ahead) on the harder Fashion-MNIST.
- Mechanism: the SNN tracks the ANN learning curve epoch-for-epoch, so the surrogate
  gradient propagates cleanly through the conv-BN-LIF stack. No dead-neuron starvation, no
  collapse. Spike rate self-sparsifies and stabilises without any noise or spike-rate loss.
- Conclusion so far: the SNN↔ANN gap is a property of the training regime, not of spiking
  computation. Membrane noise is not needed once you train with surrogate gradients + BN.
