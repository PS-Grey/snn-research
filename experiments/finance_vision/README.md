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

## Results (2026-07-20)
| model | test acc | decile spread |
|---|---|---|
| gbm-flow (finance, HGB on 7 channels) | 50.7% | +1.43% |
| **SNN + flow (naive channel-stack)** | **52.2%** | **+1.21%** |
| control (permuted labels) | 54.5% | +0.47% |
| CNN OHLC-only (finance) | 51.4% | +0.30% |
| SNN OHLC-only | 49.2% | −0.88% |

**OHLC-only is empty; flow carries signal.** The SNN goes from noise floor on OHLC-only (−0.88%)
to 3× the floor with volume/whale/taker channels (+1.21%) — the flow signal is real and the SNN
uses it, independently confirming the GBM finding. But it **matches, does not beat**, the summary
GBM (+1.21 vs +1.43, a −0.22 pp gap = within noise given sample overlap + single seed). So the
event-stream representation is **neutral vs window-summaries** — no better, no worse — on this
naive encoding. The temporal inductive bias rescued nothing on empty OHLC data (spatial CNN and
temporal SNN agree it's empty).

**Not yet tested — the actual event-stream hypothesis:** the flow channels are fed as plain
per-bar features, NOT as the suggested event mappings (volume → graded spike magnitude, whale →
second polarity, taker → event direction). That is the representation that could beat summaries;
"neutral" is the verdict for feature-stacking, not for event encoding proper.

## Next (where signal might actually be)
- **Volume / whale-crowd / taker-flow channels** (finance export, same 64-bar window + 7d label +
  split). New *information*, the biggest lever. Volume → event magnitude (graded spike);
  whale-vs-crowd → a second polarity channel; taker buy/sell → event direction. Things the
  OHLC-only CNN literally could not see. ~2-line change to `bar_features`.
- **Online / STDP self-correcting** version (walk-forward, adapts through the test period where
  the CNN is frozen). Only worth building once an informative input is confirmed — online
  learning can't create signal from empty data.
