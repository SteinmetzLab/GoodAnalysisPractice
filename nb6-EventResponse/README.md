# Notebook 6 — Baseline normalization can invent a response

**Pitfall:** averaging per-unit ratios is biased upward when the denominator (the
baseline) is noisily estimated -- Jensen's inequality, E[X/Y] > E[X]/E[Y]. In a
baseline-normalized PSTH this bias also arrives with a free alibi: the baseline
bins are pinned to exactly 1 by construction, so the figure has an immaculate flat
baseline and a sharp step at t = 0, which is the signature of a real response.

2000 units, 20 events, homogeneous Poisson firing at a constant rate throughout.
**No unit responds to the event.** The baseline-normalized population PSTH
nonetheless shows a sustained 8.8% elevation with onset in the first post-event
bin, at p = 9e-19.

## The two mechanisms

1. **Ratio bias.** Each post-event bin is divided by a baseline estimated from a
   handful of spikes, so the inflation is roughly `1 + 1/(baseline spike count)`.
   It is worst for exactly the low-rate units that normalization was introduced to
   rescue.
2. **The pinned baseline.** If the denominator is the mean of the baseline bins,
   those bins average to exactly 1 for every unit, algebraically, with zero
   variance. The flat baseline carries no information, and it hides the bias in
   precisely the region a reader would use to judge it. That is what converts a
   uniform multiplicative offset into an apparent onset at t = 0.

## Files
- `generate_data.py` -> `data/data.npz` (`spike_times`, `spike_clusters`,
  `event_times`, `n_neurons`, `recording_duration`) and `data/ground_truth.npz`
  (per-unit true rate). Standard spike-sorting layout; the notebook builds the
  PSTH itself.
- `notebook6_student.ipynb` -> build the PSTH, normalize to baseline, plot mean
  +/- SEM. The 8.8% "response".
- `notebook6_solution.ipynb` -> subtract instead of divide; the pinning identity;
  two ways to unpin the baseline (wider plot window, held-out trials); where the
  inflation lives and which summaries avoid it; bias vs amount of baseline data
  against the `1 + 1/N` prediction.
- `hints.md`.

## Reproduce
```bash
conda activate goodanalysis
python generate_data.py
python build_notebooks.py
```

## Realized numbers (seed 1)
Baseline-normalized post-event mean 1.0881 (+8.8%), t = 8.93, p = 9.2e-19 across
2000 units; baseline bins 1.00000000 exactly for every unit. Raw population rate
5.4023 -> 5.3800 spikes/s (-0.41%). Baseline-*subtracted* step -0.0223 spikes/s,
p = 0.39. Wider window: bins before the baseline window sit at 1.0869, i.e. the
"response" precedes the event. Held-out baseline: pre 1.1774 vs post 1.1727, step
-0.5 percentage points. Mean fold-change 1.0881, median 1.0000, ratio of means
0.9959, geometric mean 1.0243 with a 0.7 s post window but 0.9974 with a matched
0.3 s window. Bias vs held-out baseline duration (1 to 30 s): 1.1444, 1.1297,
1.0697, 1.0334, 1.0190, 1.0093, converging onto `1 + mean(1/baseline spikes)`
from below (that approximation overshoots badly when the baseline holds only a
handful of spikes).

## A note on the log transform
Averaging log fold-changes is the usual recommended fix and it is much better, but
it is only unbiased when numerator and denominator are estimated from **equal
amounts of data**. With a 0.7 s post window against a 0.3 s baseline the geometric
mean still reads 1.024; matching the windows brings it to 0.997. The solution
notebook shows both.
