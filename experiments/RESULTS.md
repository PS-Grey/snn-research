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

## Graded ('colour') spikes vs binary — confidence is redundant with count (2026-07-14)

Script: `graded_spikes_ab.py`. Idea: an SNN's weakness is a thin 1-bit message, not fewer
parameters. Give each spike a "colour" = membrane overshoot at firing (a confidence signal the
binary spike discards; = Loihi graded spike). Clean A/B: same STDP feature net, same R-STDP
readout, change only the message. Four encodings: A counts, B summed colour, C confidence alone
(mean overshoot), D counts+confidence as separate channels.

| encoding | 400 neurons (weak) | 800 neurons (strong) |
|---|---|---|
| A counts (binary) | 76.80% | 89.23% |
| B summed colour | 74.95% | 88.18% |
| C confidence alone | 75.35% | 86.60% |
| D counts + confidence | **78.25%** | 89.09% |
| **decisive D − A** | **+1.45 pp** | **−0.14 pp** |

**Negative result at the scale that matters.** The +1.45 pp gain at 400 neurons is a
weak-feature artefact and **vanishes on strong 800-neuron features** (−0.14 pp, noise). Reading:
confidence carries *real* class information (arm C alone = 86.6%, near the count's 89.2%), but it
is **redundant with the spike count** — a neuron that fires more is also more confident. It only
helped when the features were impoverished (count noisy) and the redundant-but-cleaner
confidence propped it up. Once features are strong the count already captures the class and
confidence adds nothing. (Sanity: counts 89.23% ≈ three-factor readout 89.18%.)

Two design lessons regardless: (1) do NOT fold colour into the spike value (arm B mixes count +
confidence + magnitude noise, strictly worse than counts); keep it a separate channel. (2) On a
single-axis task (MNIST) the colour is redundant *by construction* — a digit is just a digit.

Falsified: graded confidence as an extra *readout* feature. Follow-ups now done — see below.

**Follow-up 1: are count and overshoot even coupled?** Measured per-neuron correlation between
spike count and mean overshoot across images: **r = 0.045** (median 0.034) — nearly independent
channels. So on MNIST they were redundant only because both encode the *one* label (digit), not
because they are the same signal. (The 0.70 global count-vs-graded-sum figure is trivial:
graded-sum = count × overshoot.) This *reopened* the orthogonal-axis idea.

**Follow-up 2: two-axis test (`stdp_two_axis.py`).** Digit (axis 1) + independent contrast level
(axis 2, a magnitude property), ramping off. Binary vs graded on each head (400 neurons):

| head | binary | graded | Δ |
|---|---|---|---|
| digit | 59.05% | 60.10% | +1.05 |
| contrast | 57.00% | 56.00% | **−1.00** |

**Negative on the contrast head.** Contrast is a global-drive property, so it changes spike
*count* too (more drive → more spikes), and binary count already reads it (57% vs 33% chance);
overshoot adds nothing. **Synthesis:** count and overshoot ARE independent channels, but any
*input* property that drives overshoot also drives count, so you cannot load a controllable
second *label* into overshoot via pixels. Overshoot's independent content is an intrinsic
match-quality signal, not a dialable axis. The orthogonal-colour vision needs colour as a
**computed payload** the network *sets* on the spike (true Loihi graded spike), not membrane
overshoot *driven by* the input — a different, larger design. Idea 3 closed here.

## Surprise-weighted learning — prediction-error beats the perceptron (2026-07-14)

Script: `stdp_surprise_learning.py`. Tests confidence feeding the *learning rule* (not the
readout, which was redundant). Sergiy's concern: "learn more from confident spikes" overfloods
with easy cases and neglects the hard/uncertain ones. The fix is to learn from ERROR/SURPRISE,
not confidence — the delta rule / focal-loss / dopamine-as-prediction-error principle, still
local and backprop-free. Applied to the readout only (the recorded "spike-rate as difficulty →
unstable" failure mode is a circular feedback into the *representation*; the readout has no such
loop). Three rules on the same frozen 800-neuron features:

| rule | test acc | vs flat |
|---|---|---|
| A flat (perceptron on error = Exp 8 readout) | 89.18% | — |
| **B surprise (delta: W += lr·(onehot − softmax)·f)** | **91.27%** | **+2.09** |
| C confidence-weighted (B scaled by max p) | 91.01% | +1.83 |

**Confirmed positive, and it survived 400→800** (unlike the graded-spike gain). +2.09 pp over the
flat perceptron — the **first improvement over the three-factor 89.18% baseline**, still fully
backprop-free (single-layer delta rule, each output's local error × its input). Flat = 89.18%
matches Exp 8 exactly, so this is a clean drop-in upgrade to the three-factor readout.

**Multi-seed (seeds 0–3): robust.** surprise−flat = +2.09 / +3.15 / +1.98 / +2.16 (mean ≈ +2.3,
all positive). Surprise is also *less* variable than flat (surprise σ≈0.3 across 90.6–91.4; flat
σ≈0.6 across 88.1–89.5), so the delta rule is both better and steadier.

Also: surprise-weighting is **more stable** — the perceptron oscillates (ep1 88.4 → ep3 86.7 →
89.18) while the delta rule climbs monotonically (88.8 → 90.2 → 91.27).

On Sergiy's overflooding concern: **an earlier run showed confidence-weighting crashing to 45%
(−31 pp), but that was a learning-rate under-training artefact — retracted.** With a matched
effective rate, confidence-weighting is +1.83 (helps) but stays consistently ≤ surprise (margin
2.15 pp at 400 neurons, 0.26 pp at 800). So the intuition holds in *direction* (learn from error
> learn from confidence, because confidence neglects uncertain cases), but it is not the
catastrophe the buggy run implied on strong features.

Caveat: the delta rule's softmax is a mild cross-output (lateral) operation, like the WTA/argmax
the perceptron already needs — small, on-chip-feasible, flagged for deployment. Backprop-free
progression so far: pure STDP 88.6% → three-factor perceptron 89.18% → **surprise delta 91.27%**
(surrogate reference 98.9%).

## Surprise into FEATURE learning — fails, and instructively (2026-07-14)

Script: `stdp_surprise_features.py`. Extends Exp 9's surprise signal from the readout into the
feature layer, with an anti-instability design: global surprise from a frozen critic (not
per-neuron self-judgement, to dodge the "spike-rate as difficulty → circular" trap), magnitude
only (never sign, to dodge the anti-Hebbian collapse). Protocol: unsup features → surprise-delta
readout = baseline; surprise-weighted feature fine-tune; fresh readout = final.

**Negative across every variant** (400 neurons, baseline 80.65%): continuous surprise (−25.5 pp),
binary hard-example surprise (−29.2 pp), gentle eta 10× smaller (−39.6 pp — *worse* with more
fine-tuning). Degrades monotonically with the amount of fine-tuning, at every step size.

**Mechanism (why it must fail):** on a misclassified image the feature neurons that fired are
the ones causing the wrong answer; Hebbian only moves weights *toward* the input, so "learn more
from this error" makes those neurons respond *even more* to the image they already get wrong —
reinforcing the mistake, and compounding (more errors → higher surprise → more wrong-
reinforcement; `mean surprise` rises epoch-over-epoch). To *fix* an error you must weaken the
wrong responders (anti-Hebbian, sign-flip), which is exactly the direction that collapsed the
naive both-plastic (−48 pp). (A frozen-critic drift-mismatch inflates the effect too.)

**The tension, demonstrated from both sides:** stable (Hebbian-only) can't improve separability
and reinforces errors; corrective (anti-Hebbian) destabilises unsupervised STDP. So the **readout
is where the global surprise signal safely helps** (a supervised linear layer does the corrective
push-apart with signed error updates), while the **unsupervised feature layer resists** global-
signal improvement. This justifies the architecture: features unsupervised, surprise in the
readout. Parked; the remaining path to shape features would need a genuinely stable corrective
local rule (e.g. contrastive / predictive-coding), not reward-modulated Hebbian.

## Forward-Forward — breaks the feature ceiling, 95.1% backprop-free (2026-07-21)

Script: `forward_forward.py`. Exp 10 showed we could correct the *readout* but not the *features*
(Hebbian reinforces errors). Forward-Forward (Hinton 2022) is the fix: a *corrective* rule that
trains **every layer** with only a **local** objective, no backprop between layers. Each layer
learns to fire hard ("goodness" high) for a positive input (image + correct label) and weakly for
a negative one (image + wrong label). Inference: overlay each label, predict the one with highest
total goodness. Spiking form: goodness = mean(spike-count²) over T; the layer's own surrogate
gradient does the local update.

2 layers × 500, T=15, 40 epochs, MNIST: **95.1%** (peak ep30). Backprop-free progression:
pure STDP 88.6% → three-factor 89.18% → surprise-readout 91.27% → **Forward-Forward 95.11%**
(surrogate-gradient reference 98.9%). **First method to train the features backprop-free** — the
missing piece from Exp 10 — and +3.8 pp over the previous best, closing half the remaining gap.

Getting-it-working (three specific fixes): (1) **BatchNorm kills FF** — it normalises activity
magnitude, but magnitude *is* the goodness → g_pos ≈ g_neg, no learning. Removed. (2) label
drowned by the 784 image dims → amplify the one-hot (×10). (3) **fixed threshold far below the
operating goodness → trivial collapse** (loss only pushes one side, everything drifts to the
threshold ignoring the label) → **adaptive threshold at the running-mean goodness** forces
*separation* (pos above, neg below). That was the unlock: 9% → 95%.

Caveat/honesty: FF is Hinton's method, not ours; the spiking adaptation is modest. Value is the
head-to-head in a spiking/neuromorphic setting and the mechanism (corrective > unsupervised).

## Recurrent / top-down FF (the "FFF" idea) — no benefit (2026-07-21)

Script: `forward_forward_recurrent.py`. Tests whether *more/recurrent passes that refine each
other* (Sergiy's "FFF" idea, the bridge toward predictive coding) beat plain FF. Both layers run
in one T-step loop; layer 2 feeds back to layer 1 each step (settling). Each layer still trains
on its own local FF goodness, cross-layer signals detached (no backprop between layers). Clean
A/B, same net:

| | final MNIST |
|---|---|
| top-down OFF (feedforward FF) | 95.00% |
| top-down ON (recurrent settling) | 94.51% |

**Negative: top-down feedback is marginally worse (~−0.4 pp), consistent across the run.** Why:
adding a recurrent connection to FF is **not** predictive coding. Real PC has the top-down layer
*predict* bottom-up activity, uses the *error* between them to drive updates, and *iterates to
convergence* (the ~100× compute the lit-scout flagged). This version just adds a feedback wire
and keeps the FF goodness objective, so the feedback has no error-correcting role and only ~15
forming timesteps to "settle" — it adds noise, not refinement. **The cheap shortcut to
predictive coding doesn't exist; PC is a distinct method that has to be built properly.** Confirms
the survey. (Sanity: top-down-off reproduces the feedforward FF 95.1%.)

## Equilibrium Propagation — 97.6%, new best AND on-chip-deployable (2026-07-21)

Script: `equilibrium_prop.py`. EP is the **strictly two-neuron-local** rule: each synapse updates
from only the two neurons it connects, measured across two settling states — free phase (input
clamped, network relaxes to equilibrium) and nudged phase (output gently pulled toward the
target). Update ∝ (1/β)[ρ(sᵢ)ρ(sⱼ)|nudged − ρ(sᵢ)ρ(sⱼ)|free], symmetric ±β nudging (Laborieux
2021) to cancel the bias. No backprop; the error spreads through the settling itself.

784→500→10, 25 settling steps, symmetric weights, MNIST: **97.6%** (peak ep18). Backprop-free
line: STDP 88.6 → surprise 91.3 → FF 95.1 → **EP 97.6** (backprop reference 98.9%). **New best,
only ~1.3 pp from backprop** — and unlike FF it's *strictly local* (each weight sees only its two
neurons), the property that lets it run on-chip. Best accuracy *and* best hardware-fit; every
other method traded one for the other.

Caveat: this is **rate-based** (analog) EP, not spiking — its neuromorphic home is analog
hardware, and the load-bearing property is the local settling rule. Spiking-EP is the extension.
Also note MNIST is saturated (~99% ceiling); the method-separating test is CIFAR (→ Colab).

## Corrective STDP (contrastive Hebbian) — subsumed by EP (2026-07-21)

Script: `corrective_stdp.py`. Novelty attempt: give STDP the yes/no signal it lacks (Exp 10) by
adding Forward-Forward's positive/negative contrast, but with a *pure local Hebbian* update
(pre×post gated by ±sign) instead of a goodness gradient. Idea: a more hardware-faithful FF.

**Result: dropped, two reasons.** (1) The naive spike-count version does not learn (below chance).
(2) More importantly — **"contrast + Hebbian update" IS the Contrastive Hebbian Learning family
(Movellan 1991), and Equilibrium Propagation (Exp 13, 97.6%) is the mature, well-formulated member
of exactly that family, which we already have.** So corrective-STDP is at best a *cruder EP* — not
a novelty bet, and worse than what's built. The only genuinely fresh angle would be a *timing*-based
version (real STDP windows, not spike-count correlation), still EP-adjacent.

Lesson (novelty mode): **identify the method *family* before building.** This idea was subsumed by
a method we had already implemented. Effort redirected to Direction 2 (learned graded-spike payload,
genuinely less-explored; lit-scout first).

## Catastrophic forgetting demonstrated — the novel-direction setup (2026-07-22)

Script: `continual_forgetting.py`. Answers "does on-chip continual learning keep old knowledge?"
EP (Exp 13) trained on MNIST, then on Fashion-MNIST (no MNIST rehearsal):

| | MNIST | Fashion |
|---|---|---|
| after MNIST | 97.1% | — |
| after Fashion | **31.8%** (−65 pp) | 65.3% |

**Severe forgetting: MNIST 97→32%, near chance.** The net overwrites the old task to fit the new
one (and old knowledge interferes with the new task — Fashion only 65% vs ~90% fresh). Confirms:
naive on-chip continual learning does NOT retain old tasks. Locality doesn't fix this; it can make
it worse. This is the problem the novel direction attacks (see the vault note *SNN Novelty Target
— Learned Graded Payload + Continual Learning*): does a **learned graded-payload channel** and/or
**metaplasticity/stickiness** protect the old task? The scout confirmed that's open ground.

## Does freezing features reduce forgetting? NO — it's a READOUT problem (2026-07-22)

Script: `continual_compare.py`. Class-incremental MNIST (5 tasks × 2 classes each), EP full-plastic
vs features frozen after task 0 (proxy for "unsupervised frozen features + readout", the STDP
family). Tested "does the learning rule / where-plasticity-lives affect forgetting?"

| arm | task-0 acc (start → end) | final all-class |
|---|---|---|
| full (all plastic) | 100% → 6% | 14.5% |
| frozen (features locked) | 100% → **0%** | 22.2% |

**Hypothesis falsified: freezing the representation does NOT reduce forgetting** (frozen is no
better, worse on task-0). Both collapse task-0 to ~0% after just the *second* task. **The
forgetting is in the READOUT, not the features:** in class-incremental training the target for a
new-class example says the old-class outputs should be *zero*, so the negative nudge **actively
suppresses** old-class outputs — it doesn't merely neglect them. Feature stability can't help
because the class-competition that gets clobbered lives in the readout.

**Implications:** (1) "which rule forgets less" is the wrong axis here — the **output structure**
dominates, not the feature rule. (2) Adding feature *capacity* (graded payload) would NOT fix
this; the problem is the readout. (3) **Rehearsal/recall attacks the exact mechanism** — keep old
classes present (even self-generated) so their outputs stop being suppressed. Points the
anti-forgetting work at recall, not capacity. Caveat: EP-frozen is a proxy; a true STDP-vs-EP
comparison would confirm, but the readout-suppression mechanism is architecture-general.

## Recall fixes the forgetting — exemplar rehearsal (2026-07-22)

Script: `recall_continual.py`. Class-incremental EP, no-recall vs 20 stored exemplars/class mixed
into each new-task batch.

| arm | task-0 (start → end) | final all-class |
|---|---|---|
| no-recall (Exp 16 baseline) | 100% → 6% | 14.5% |
| **recall (20/class)** | 100% → **82%** | **59.9%** |

**Recall dramatically fixes it:** task-0 retention 6→82% (+76 pp), all-class 14.5→60% (+45 pp),
from only 20 exemplars/class. Confirms the Exp-16 readout-suppression diagnosis — keeping old
classes present stops their outputs being suppressed. This is the **known baseline** (stored-
exemplar replay); it validates the mechanism and direction, not novelty. 60% is still below ~97%
joint training, so recall helps hugely but not fully (more exemplars/rehearsal would push higher).

**Novel targets next (the actual contribution):** (a) **EP-native generative recall** — no stored
data; the network *settles to generate* old-class patterns itself (energy-based generation),
rehearsing without a buffer. (b) **Forgetting-curve-scheduled recall** — spaced-repetition timing,
recall effort proportional to a learned forgetting risk. Lit-scout both before claiming ground.

## Self-measured forgetting-scheduled replay — null at MNIST scale (2026-07-22)

Script: `recall_scheduled.py`. First genuine novelty test. Lit-scout (2026-07-22) confirmed the
open sub-question: does replay scheduled by the model's OWN measured forgetting beat uniform replay
at matched budget? (Settle-to-generate EP replay itself is now published — Cook et al. 2026 — so
the scheduler is the open piece.)

| arm (matched budget 16/step) | task-0 (end) | final all-class |
|---|---|---|
| uniform | 78% | 64.2% |
| forgetting-scheduled (measured retention) | 76% | 63.5% |

**Null — scheduled ≈ uniform (within noise).** Diagnosis: on MNIST the classes forget *uniformly*
(similar difficulty), so there's nothing for a which-to-replay scheduler to exploit; uniform
already covers everyone. A self-measured schedule only helps under **heterogeneous forgetting**
(more tasks, tighter budget, or mixed-difficulty classes — e.g. MNIST+Fashion). **Wrong-regime
null, not a dead idea.** Also: this is the *crude* version (measured retention as the signal); the
scout's *coherent* target — **energy/basin-depth as both the generator and the forgetting signal**,
plus **spiking** — is untested. The open problem stands; its cheapest proxy just doesn't bite at
MNIST scale.

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

## Heterogeneous forgetting — the scheduler win is ordering-dependent (Exp 19, 2026-07-22)

Exp 18 was a wrong-regime null (MNIST forgets uniformly). This builds the heterogeneous regime it
lacked — **MNIST+Fashion as one 20-class class-incremental problem** (0-9 easy, 10-19 hard), EP,
10 tasks of 2 classes — and re-tests self-scheduled replay at matched 16-sample budget, plus the
energy/basin-depth signal (direction 2). `continual_hetero.py`, two task orderings.

| arm (matched budget) | domain-order (MNIST→Fashion) | interleave-order (easy+hard per task) |
|---|---|---|
| uniform | 40.9% | 48.5% |
| forgetting-scheduled | **56.0% (+15.1)** | 48.9% (+0.4, null) |
| energy/basin-depth (dir-2) | 51.0% (+10.1) | 43.9% (−4.6, worse) |

**The +15pp is not general — it needs a structural old/new split.** Under domain-order there is a
clean pocket of recently-collapsing classes (Fashion at 0%) that uniform replay neglects and the
scheduler rescues (visible: 11→86, 13→70, 14→80). Interleave the difficulty and that pocket
disappears, so the scheduler ≈ uniform. Honest read: **self-measured priority replay helps only
when forgetting is structurally lumpy (domain shift), not when it's diffuse** — consistent with
known priority-replay results (MIR, Aljundi 2019). So this is a known idea confirmed in a
backprop-free EP setting, and only in one regime.

**Energy signal (dir-2) underperforms uniform in both orderings.** As implemented (settled output
confidence for the true class) it's a noisier accuracy meter, not a fair test of the *basin-depth*
idea — the real EP energy of the settled state was not computed. Either that fair version is built,
or the premise (energy predicts forgetting better than accuracy) is abandoned; basin depth and
"still classified right" are correlated by construction, so even a fair version may only match
accuracy, never beat it. The scout's actual appeal for energy was *elegance* (same quantity does
generation AND forgetting-readout), not better prediction.

**Easy-vs-hard forgetting (recency-controlled, interleave arm):** with difficulty separated from
recency (MNIST class c and Fashion c+10 learned in the same task), **easier-learned classes forget
MORE** — MNIST 35pp vs Fashion 23pp at matched age. Caveat: partly a ceiling effect (MNIST learned
to higher accuracy → more absolute pp to lose); the relative-retention version is the fairer test,
and this is single-seed. Interesting direction anyway: sharp, easily-learned representations get
overwritten faster than the more distributed hard-learned ones.

**Net:** the recall/scheduling line is converging on known territory (priority replay, ordering-
dependent), and the one genuinely-novel angle (EP-native energy signal) does not yet beat uniform.
Forks worth deciding before more compute: (a) fair energy signal = actual settled-state EP energy;
(b) the untouched fully-open bet — a plastic learned graded-spike payload (Δpayload local rule);
(c) multi-seed the easy-forget-more effect (a clean, self-contained empirical result).

### Fair energy signal — dead, with a clean mechanism (Exp 19b)

Gave direction 2 its fair test: the REAL settled free-phase EP energy (`EPNet.energy`), min-max
normalised across seen classes, as the replay-priority signal (high energy = shallow basin =
replay more). Domain order, matched budget:

| arm | all-class |
|---|---|
| uniform | 40.9% |
| forgetting-scheduled (accuracy) | **56.0%** |
| energy (real EP energy) | 41.4% (≈ uniform) |

**Energy fails because it measures settling *confidence*, not *correctness*.** A catastrophically-
forgotten class does not fall into a shallow basin — it settles *confidently into a competing
class's basin* (low energy, wrong label). So the energy scheduler marks collapsed classes as "deep
basin, safe" and never rescues them: classes 10 and 12 stay at 0% under energy while the accuracy
signal recovers their neighbours to 70-86%. The early-warning hypothesis was backwards — energy
gives false *security*, not early warning. Both dir-2 signals (settled confidence, real energy) are
now killed for principled reasons. Energy's only remaining niche is the LABEL-FREE generative-recall
regime, where accuracy can't be computed — untested, and a bigger build.

**Where this leaves the recall line:** the one method that works (accuracy-priority replay, +15pp
domain-order) is known priority replay (MIR), regime-specific. Both novel signal ideas failed. The
freshest genuinely-different angle is now a *cost-aware* objective (Sergiy, 2026-07-22): schedule by
**relearning cost, not forgetting amount** — hold what's expensive to reacquire, cheaply relearn the
easy stuff on demand (mirrors Bjork's desirable difficulties + Anderson & Schooler's rational
forgetting). Novel-looking in a backprop-free SNN; needs a lit-scout before building.

## Graded-payload probe — learned payload adds nothing over computed, under end-to-end training (Exp 20)

Scaffold for the untouched open problem (`graded_payload.py`): a per-spike graded PAYLOAD that
rides on already-firing spikes (matched spike budget exact — payload adds zero spikes), transmitting
`s_j * payload_j`. Surrogate-gradient PROBE (is there capacity at all?). MNIST, 20k train, 4 epochs.

| mode | acc | spk/img | params | note |
|---|---|---|---|---|
| binary (rate code) | 95.90% | 1467 | 239k | baseline |
| computed (membrane = sigma-delta) | **96.70%** | 1360 | 239k | KNOWN method, wins |
| learned (projection of input x) | 96.51% | 1309 | 474k | redundant with rate code |
| learned_u (learned readout of membrane) | 95.96% | 1372 | 329k | collapsed toward constant (~binary) |

**A graded payload helps (+0.8pp) but it's the known sigma-delta (raw membrane); LEARNING the
payload does not beat COMPUTING it.** Both learned variants underperform computed despite more
params. Mechanism: under end-to-end training the readout W2 already learns to exploit the payload,
so a separately-learned payload transform is redundant — W2 absorbs its role. The payload only needs
to *carry* the graded state; downstream weights do the rest.

**Redirect, not death:** a learned payload can only earn capacity where no global readout absorbs it
— under a **local learning rule** (no end-to-end W2) or in **continual learning** (payload = a
protected second channel; the Hajizada-2025 convergence). The end-to-end probe is structurally blind
to payload value. Owed controls if pursued: param-matched binary (`--wide`), and a fix for the
learned_u collapse (init/scale of the payload layer). The honest next test is the LOCAL-rule regime.

## Cost-of-relearning replay — null over 3 seeds, and a correction to Exp 19 (Exp 21, 2026-07-22)

First implementation of the scout-confirmed-open criterion (`cost_replay.py`): schedule replay by
REACQUISITION COST (basin-narrowness proxy: margin collapse under input perturbation) vs uniform and
vs MIR (forgetting/low-retention), matched budget, 20-class MNIST+Fashion class-incremental, EP.

Seed 0 looked like the scout's prediction (cost beat MIR by +3.8pp on hard/Fashion classes, trading
MNIST). **It did not survive multi-seed.** Means over seeds 0/1/2:

| arm | overall | MNIST | Fashion |
|---|---|---|---|
| uniform | **55.9%** | 67.5% | **44.2%** |
| mir | 53.3% | 68.5% | 38.1% |
| cost | 47.6% | 63.2% | 32.0% |

**Cost-scheduling is the WORST arm on average** (best only at seed 0, worst at seeds 1&2). No
scheduler beats uniform robustly. Cost-of-relearning scheduling as implemented (basin-narrowness
proxy) actively hurts.

**CORRECTION to Exp 19.** Uniform's own overall accuracy swings 42.6%→65.4% across seeds — a 23pp
RNG range. Single-seed comparisons in this regime are unreliable at the pp level. Exp 19's headline
"forgetting-scheduled beats uniform +15pp" (56.0 vs 40.9, single seed) does NOT replicate: the same
rule (mir) at seed 0 here gives 43.9 vs 42.6 (+1.3pp), the difference being RNG ordering alone. **The
+15pp win was seed-luck, not a real effect.** Unified honest verdict across Exp 16–21: no replay
scheduler (MIR / energy / cost) robustly beats UNIFORM replay under multi-seed. Durable results are
only the *mechanism* (recall fixes forgetting, Exp 17) and the *observation* (easy-learned classes
forget faster, Exp 19), neither of which is a scheduling claim. Methodological lesson: every
class-incremental scheduling comparison needs multi-seed + variance reporting; this was not done for
Exp 16–19 and their pp-level rankings should be treated as noise.

## Learned graded payload under a LOCAL rule — negative, multi-seed, two regimes agree (Exp 22, 2026-07-22)

Exp 20's negative (learned payload useless) could have been an artefact of end-to-end training (W2
absorbs the payload). This tests the honest regime: fixed random reservoir (hidden NOT trained
end-to-end), LOCAL delta-rule readout, and a plastic Delta-payload rule (per-neuron input-dependent
graded value `softplus(a*rate+b)`, params learned by a three-factor DFA rule). `graded_payload_local.py`,
3 seeds, matched spikes.

| mode | mean acc | per-seed |
|---|---|---|
| binary (rate) | 94.47% | 94.39 / 94.52 / 94.51 |
| computed (sigma-delta) | **95.49%** | 95.53 / 95.49 / 95.44 |
| fixed (random payload nonlin) | 94.63% | 94.65 / 94.63 / 94.60 |
| learned (DFA Δpayload) | 94.48% | 94.44 / 94.54 / 94.45 |

`computed - binary = +1.01pp` (robust). `learned - fixed = -0.15pp` (learning the payload is WORSE
than a random fixed nonlinearity). `learned - computed = -1.01pp`. **The local rule did not rescue
the learned payload; two independent regimes (end-to-end Exp 20, local Exp 22) agree: a graded
payload helps only as the KNOWN computed/sigma-delta membrane value; LEARNING it adds nothing.**
Mechanism: the payload's value is membrane/temporal info the spike count lacks — that is computed,
not learned; a learned reshape of the rate carries no information a linear readout couldn't use.
To beat computed, a learned payload would have to read the membrane AND out-encode raw membrane —
Exp 20's `learned_u` already tried that end-to-end and failed. Both scout-open directions
(cost-scheduling Exp 21, graded-payload Exp 20/22) now fail their probes with clean mechanisms.
