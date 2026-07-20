# SNN chart-vision (finance project benchmark)

Application experiment: predict P(next-7d return > 0) from 64 daily OHLC bars, on the finance
project's fixed vision benchmark. The SNN angle (vs the from-scratch CNN that found nothing):
read the chart as a **time series** — 64 bars = 64 timesteps, LIF membrane carries memory
bar-to-bar — rather than a static image.

- `snn_chart.py` — surrogate-gradient SNN (backprop baseline; the online/STDP self-correcting
  version is the planned follow-up). Reads `~/Claude/finance/lab/data/vision_ds/`, writes
  `preds_snn.csv` there in arena format. Build lives here (snn-research); finance owns the
  dataset + scorer + dashboard.

## Task (fixed — do not redefine)
Input 64×4 price-normalised OHLC; output P(fwd7 > 0). Train only on `split=="train"` (< 2025-01-01,
9,964 rows). Test 7,819 rows (2025→2026), temporal split sacred, no peeking. Majority-class
baseline ~54.3% (test period is bear-tilted). Tradeable metric = decile spread (mean fwd7 of
top-10%-bullish minus bottom-10%).

## Results (2026-07-20) — all rows on the finance arena (single seed)
| model | test acc | decile spread |
|---|---|---|
| gbm-flow (finance, HGB on 7 channels) | 50.7% | **+1.43%** |
| SNN flow, raw channels (`snn-flow`) | 52.2% | +1.21% |
| SNN flow, hand event-encoding (`snn-flow-events`) | 50.5% | +0.62% |
| control (permuted labels) | 54.5% | +0.47% |
| CNN OHLC-only (finance) | 51.4% | +0.30% |
| SNN OHLC-only (`snn-ohlc`) | 49.2% | −0.88% |

**Conclusions (this task, this scale):**
1. **OHLC-only is empty** — spatial CNN (+0.30%) and temporal SNN (−0.88%) agree the price shape
   carries nothing. Temporal inductive bias rescued nothing.
2. **Flow carries real signal** — the SNN jumps to 3× the noise floor (+1.21%) with
   volume/whale/taker, independently confirming the GBM's finding via a different model class.
3. **Spikes do not beat summaries** — SNN flow (+1.21%) *matches but does not beat* the summary
   GBM (+1.43%); the −0.22 pp gap is within noise (sample overlap, single seed).
4. **Hand event-encoding HURTS** (+0.62%, below the raw-channel SNN). The suggested mappings
   (volume→magnitude, whale→polarity, taker→direction) *pre-multiply* channels
   (`up × whale × volume`), collapsing three separable signals into one number — the model can no
   longer weight them independently. Feeding raw channels and letting the learner find the
   combination is strictly better. The more "neuromorphic" the representation, the worse it did.

**Net:** the signal is in the *data* (flow), not the *representation* (spikes/events). A simple
learner captures it as well as, or better than, the SNN. The one card unplayed is the online /
self-correcting version (`snn-online`) — legitimate walk-forward (learn only from labels resolved
≥7 days before prediction) — which could add value by *tracking* the signal through regime shift,
not by extracting more of it. Tempered expectation after the batch results.

## Efficiency proxy (`efficiency_proxy.py`, 2026-07-20)
The SNN's genuine advantage is not accuracy but cost per inference. Can't be *measured* without
neuromorphic hardware (simulating an SNN on a Mac is slower than a GPU CNN — the win is a chip
property), so we measure the proxy that determines it:

| metric | value |
|---|---|
| spikes / inference | ~1,280 (14.6% sparsity) |
| event SynOps / inference | ~39,000 |
| dense input MACs (`fc1`, not event-driven) | ~33,000 |
| theoretical latency (Loihi-class est.) | ~320 µs |
| theoretical energy (est.) | ~0.8 µJ |
| GPU CNN, for contrast | ~single-digit ms, ~mJ |

**The real advantage is ENERGY (~1000×), not latency (~few×, same order)** — a GPU CNN is also
sub-10 ms, so "spikes react faster" overstates it; the dramatic gap is power. Caveats: estimate
not measurement; the input front-end is dense (not event-driven), undercutting the pure-spike
story; and per the finance ARENA, none of it helps retail trading (network + fees dominate). It
matters only for always-on power-constrained edge, or true HFT — both outside this project's
scope. Corrected SNN pitch: *same decision at ~1000× less energy*, not *faster reactions*.

## snn-online — walk-forward adaptive, no reliable gain (`snn_online.py`, 2026-07-20)
Frozen SNN feature extractor + online-adapting delta-rule readout, walking forward through the
test period learning only from resolved labels (date ≤ t−7d, the fairness rule). Online-lr chosen
on a train-period holdout (never test). 8-seed:

| | mean | range |
|---|---|---|
| frozen readout | +0.57% | −0.96 .. +2.44 |
| online readout | +0.78% | +0.16 .. +2.31 |
| **online − frozen** | **+0.22 pp** | **−2.11 .. +1.92** |

**No reliable benefit.** The paired online−frozen delta averages +0.22 pp but swings ±2 pp by
seed — dominated by noise. The flow signal (whale-follow/crowd-fade) is stationary enough across
the 2025 regime boundary that walk-forward tracking adds nothing reliable; online just adds
variance. Confirms the tempered expectation: the SNN's one structural edge over batch GBM
(on-the-fly adaptation) does not materialise as a gain here. Arena row `snn-online` = +0.90%
(8-seed ensemble), notes carry the honest no-gain verdict.

**Also exposed:** the SNN-flow readout is high-variance — single-seed +1.21% was optimistic;
multi-seed frozen mean ~+0.57% (range −1..+2.4). Multi-seeding caught what one seed hid.

## Next (where signal might actually be)
- **Volume / whale-crowd / taker-flow channels** (finance export, same 64-bar window + 7d label +
  split). New *information*, the biggest lever. Volume → event magnitude (graded spike);
  whale-vs-crowd → a second polarity channel; taker buy/sell → event direction. Things the
  OHLC-only CNN literally could not see. ~2-line change to `bar_features`.
- **Online / STDP self-correcting** version (walk-forward, adapts through the test period where
  the CNN is frozen). Only worth building once an informative input is confirmed — online
  learning can't create signal from empty data.
