# Notebook 8 — A t-test on skewed data with unequal group sizes

**Pitfall:** the two-sample t-test tolerates skew well when the groups are the
same size, because the skew in the two sample means cancels. It does not when the
groups are unequal. Welch's version, the usual recommended default, is the worse
of the two here, because it trusts each group's own standard deviation and a small
sample of skewed data gets its mean and its SD wrong *in the same direction at the
same time*.

150 units in one recording, 10 localized to region A and 140 to region B. Region
A appears to fire at half the rate of region B (5.1 vs 9.6 spikes/s, Welch's
t-test p = 0.009). **All 150 rates were drawn from one lognormal distribution and
region was assigned without reference to rate**, so the true difference is exactly
zero.

## The mechanism

Firing rates across neurons are lognormal, spanning orders of magnitude, so the
population mean is carried by the tail. Two consequences, and the pitfall needs
both:

1. A sample of 10 usually **misses the tail**, so its mean comes out below the
   population mean.
2. Its SD comes out low **at the same time**, because the same missing large
   values drive both. Sample mean and sample SD are strongly positively correlated
   for skewed data (r = 0.92 at n = 10 here).

Welch reads the shrunken SD as precision. In this dataset region A has SD 2.70
against region B's 17.07 purely because 10 draws did not reach the tail that 140
draws did.

The bias has a **consistent direction**: the small group almost always looks
quieter. Unlike a shared-nuisance false positive, which flips sign at random (see
[`nb7-ConditionAvsB/`](../nb7-ConditionAvsB)), this one points the same way every
time. Any literature comparing a rare, hard-to-sample population against an
abundant one, using means and t-tests, will tend to report the rare population as
less active.

## Three results that are easy to guess wrong

- **Student's pooled t-test is safer than Welch's here** (5.5% vs 17% false
  positive rate), the reverse of the usual advice. Pooling lets the large group's
  honest SD dominate so the standard error cannot collapse. This is two errors
  partly cancelling, not a recommendation.
- **Equal group sizes fix it.** At 30 vs 30 the skew barely matters and Welch is
  slightly conservative. It is skew **plus** imbalance that breaks the test.
- **Welch's rejection rate is not monotonic in the truth.** Over part of the range
  it *falls* as the real effect grows, because a genuine increase in region A
  cancels the downward bias that was generating the false positives.

## Files
- `generate_data.py` -> `data/data.npz` (`spike_counts`, `duration`, `region`) and
  `data/ground_truth.npz` (per-unit true rate).
- `notebook8_student.ipynb` -> the bar chart, the strip plot, and Welch's t-test.
- `notebook8_solution.ipynb` -> the distribution on linear and log axes; measured
  false positive rates for four tests plus a permutation test, with and without
  balanced groups; the mean/SD coupling that drives it; sweeps over imbalance and
  over skew; and a positive control.
- `hints.md`.

## Reproduce
```bash
conda activate goodanalysis
python generate_data.py
python build_notebooks.py
```

## Realized numbers (seed 290)

The dataset: 150 units, median rate 4.28 spikes/s, range 0.19 to 110.1.
Region A (n = 10) mean 5.10, median 4.69, SD 2.70, max 10.9 spikes/s.
Region B (n = 140) mean 9.57, median 4.11, SD 17.07, max 110.1 spikes/s.
So the **means differ by -47% while the medians differ by +14%**, in the opposite
direction.

On this dataset: Welch's t p = 0.009, Student's pooled t p = 0.41,
Mann-Whitney p = 0.81, t on log rates p = 0.65, permutation on the difference of
means p ~ 0.4. Only Welch's rejects.

False positive rates with both groups drawn from the same distribution
(sigma = 1.2): at 10 vs 140, Welch 0.166, Student 0.055, Mann-Whitney 0.050,
t on log 0.054. At 30 vs 30: Welch 0.035, Student 0.037, Mann-Whitney 0.051,
t on log 0.053. When Welch rejects, the small group is the lower one 99% of the
time.

Imbalance sweep (150 units total, sigma 1.2), Welch's false positive rate: 0.040
at 75 vs 75, then 0.053, 0.074, 0.102, 0.157, 0.219 as the split goes to 50/100,
30/120, 20/130, 10/140, 5/145. Skew sweep at 10 vs 140: 0.051 with no skew, then
0.063, 0.103, 0.157, 0.204 at sigma 0.4, 0.8, 1.2, 1.6. Mann-Whitney stays at
0.046 to 0.052 throughout both sweeps.

Power at 10 vs 140 for a real rate ratio: Mann-Whitney 0.40 at 2x, 0.76 at 3x,
0.92 at 4x; t on log rates 0.37, 0.72, 0.90. Welch's rejection rate over the same
range is 0.162, 0.052, 0.057, 0.178, 0.322, 0.517 at ratios 1, 1.5, 2, 3, 4, 6:
non-monotonic, and *lower* at a true 1.5x effect than under the null. With only 10
units in region A you need roughly a threefold difference before a
correctly-calibrated test is likely to find it, which is a constraint of the
design rather than a defect of the tests.
