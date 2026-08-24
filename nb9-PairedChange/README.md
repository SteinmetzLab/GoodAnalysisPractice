# Notebook 9 — A paired t-test that misses a real effect

**Pitfall:** the paired t-test tests the mean difference. When the effect is
*multiplicative* and the baseline values are heavy-tailed, the difference in
spikes/s is proportional to the baseline rate, so the mean difference is a
**rate-weighted** average dominated by a handful of fast units. The test stays
perfectly calibrated and loses almost all of its power.

Unlike the rest of this collection, the false conclusion here is a **negative**
one. 40 units, optogenetic suppression of a long-range input, interleaved
laser-off and laser-on blocks. The paired t-test returns
**t = -0.48, p = 0.63** and you would write that the input does nothing.

**The manipulation really does suppress firing, by about 20%, in 32 of the 40
units.**

## Why it fails

1. The effect is multiplicative: each unit's rate is scaled by a factor with
   median 0.80. That is what most neural manipulations do.
2. Baseline rates are lognormal and span 0.11 to 82 spikes/s.
3. So the absolute change scales with rate. The 82 spikes/s unit moves by about
   16 spikes/s, the 1 spikes/s unit by 0.2. The mean paired difference is a
   rate-weighted average with an **effective sample size of 10.0 of 40**
   (`(sum r)^2 / sum(r^2)`), and the top 5 units carry 48% of the total absolute
   change. The same units inflate the SD, so both halves of the t-statistic are
   set by a few neurons.

The same t-test on **log** rates goes from p = 0.63 to p = 1.7e-06 on identical
data, because the log scale turns the multiplicative effect into an additive one.

## This is a power failure, not a Type I failure

Under a strict null (measure the same units twice, nothing changes) the paired
differences are exchangeable and therefore symmetric, and the paired t-test is
essentially exact. Measured: 0.036 against a nominal 0.05. The test is honest, it
simply cannot see this kind of effect.

## The caveat that matters more than the fix

"The t-test failed, so use the non-parametric one" is a habit worth not forming.
The three tests have three **different** null hypotheses:

- paired t-test: `mean(difference) = 0`
- Wilcoxon signed-rank: the differences are **symmetric** about 0
- sign test: `median(difference) = 0`

They coincide when the differences are symmetric and diverge sharply when they
are not. With differences that are skewed but have a mean of *exactly* zero,
Wilcoxon rejects 42% of the time and the sign test 66%, while the t-test, the one
whose null is actually true, sits near 5%. Choose the summary that answers your
question first, then choose the test. If the effect is multiplicative, do the
whole thing on a log scale, where the mean, the median and the symmetry point all
coincide again.

## Files
- `generate_data.py` -> `data/data.npz` (`counts_off`, `counts_on`, `duration`)
  and `data/ground_truth.npz` (true baseline rate and true factor per unit).
- `notebook9_student.ipynb` -> the bar chart, the paired plot on a linear axis,
  and the paired t-test.
- `notebook9_solution.ipynb` -> the same data on a log axis; four tests compared;
  the rate-weighting and effective sample size; measured Type I error and power;
  and the three-nulls comparison.
- `hints.md`.

## Reproduce
```bash
conda activate goodanalysis
python generate_data.py
python build_notebooks.py
```

## Realized numbers (seed 144)

40 units, 10 min per condition. Baseline rates median 3.14, range 0.11 to 82.3
spikes/s. Mean rate 9.31 -> 9.02 spikes/s (-3.2%), but the **median per-unit
change is -22.4%** and **32 of 40 units decreased**.

| test | statistic | p |
|---|---|---|
| paired t on rate differences | t = -0.480 | **0.634** |
| Wilcoxon signed-rank | W = 163 | 0.00060 |
| sign test (32 of 40 down) | 32 | 0.00018 |
| paired t on log rates | t = -5.633 | 1.7e-06 |

Effect size from the log scale: **x0.756, 95% CI x0.684 to x0.836**, a 24%
reduction (16% to 32%). The population factor used to generate the data had
median 0.80.

Effective sample size of the rate-weighted mean: 10.0 of 40. Top 1, 3, 5, 10
units by rate carry 1.5%, 36.2%, 48.3%, 70.0% of the total absolute change. Mean
difference -0.295 spikes/s with SD 3.888 gives t = -0.480; the same data on the
log scale gives mean -0.280 with SD 0.314 and t = -5.633.

Over repeated experiments from this model:

| test | Type I (no effect) | power (real 20% drop) |
|---|---|---|
| paired t (rates) | 0.036 | **0.430** |
| Wilcoxon signed-rank | 0.041 | 0.949 |
| sign test | 0.037 | 0.942 |
| paired t (log rates) | 0.041 | 0.997 |

Three nulls, rejection rate at n = 40 pairs:

| difference distribution | paired t | Wilcoxon | sign |
|---|---|---|---|
| symmetric, mean 0 (all 3 nulls true) | 0.046 | 0.048 | 0.036 |
| symmetric heavy-tailed (all 3 true) | 0.045 | 0.047 | 0.039 |
| right-skewed, MEAN 0 (only t true) | 0.067 | 0.191 | 0.345 |
| lognormal, MEAN 0 (only t true) | 0.107 | 0.422 | 0.657 |
| right-skewed, MEDIAN 0 (only sign true) | 0.454 | 0.176 | 0.040 |

On right-skewed differences with a true mean of zero the paired t-test's errors
are also badly one-sided: at n = 10 pairs, 9.4% of experiments falsely report a
"decrease" against 0.4% reporting an "increase". A one-sided paired t-test in the
unlucky direction is inflated roughly twofold.

## Relation to the other notebooks

[`nb8-RegionRates/`](../nb8-RegionRates) is the other skew notebook, but it is an
*unpaired* Type I failure driven by unequal group sizes, and the test it breaks
is Welch's. This one breaks the paired t-test, which far more people actually
run, and it breaks it by removing power rather than by inventing an effect.
