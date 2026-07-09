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

| config | test acc | notes |
|---|---|---|
| 100 neurons, 6k imgs, no ramping | ~45% (42.8 / 46.3 / 46.9, seeds 0–2) | initial version |
| 400 neurons, 12k imgs, no ramping | 47.5% | plateaus; specialisation imbalanced (class 2 hogs) |
| 100 neurons, 6k imgs, **+ intensity ramping** | 64.5% | class spread evens out, silent imgs 133→49 |
| 400 neurons, 30k imgs, T=100, + ramping | 70.5% | class 1 still light; 1353 silent test imgs |
| **800 neurons, 30k×2 passes, T=100, ramping (max_tries 6)** | **86.04%** | matches D&C regime; silent imgs → 210 |

Progression 47.5% → **86.04%**, backprop-free the whole way. Two levers did it:
1. **Intensity ramping** (+18 pp alone at 100 neurons) — D&C's trick of re-presenting an image
   at higher intensity until it drives ≥5 spikes. Without it, thin low-pixel digits never won
   neurons, capping accuracy via class imbalance. Raising `max_tries` 4→6 also cut silent-eval
   images (a direct auto-fail) from ~13% to ~2%.
2. **Scale + revisiting** — 800 neurons and a 2nd pass over the data sharpen and spread the
   receptive fields (class 1 rep 13→32 neurons).

**86.04% is now the honest pure-STDP floor** a three-factor / reward-modulated rule must beat
(surrogate-gradient reference: 98.9%). All primitives here are on-chip-implementable, with two
caveats flagged for deployment: weight-normalisation and hard-WTA are not cleanly local.

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
