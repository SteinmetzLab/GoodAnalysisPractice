"""
Generate the dataset for Notebook 9: "A paired t-test on heavy-tailed data."

What is simulated
-----------------
One recording, 40 units held throughout. A manipulation (optogenetic suppression
of a long-range input) is delivered in interleaved blocks, giving 600 s of
"off" and 600 s of "on" per unit. Every unit is measured in both conditions, so
the design is paired.

**There is a real effect and it is large.** Each unit's rate is multiplied by a
factor with median 0.80, a 20% decrease, with per-unit variability
(lognormal, sigma 0.30). In the delivered dataset 32 of 40 units go down and the
median unit drops 22%.

Why the paired t-test misses it
-------------------------------
Baseline rates are lognormal and very wide (0.11 to 82 spikes/s here). Because
the effect is multiplicative, the *absolute* change in spikes/s is proportional
to the unit's rate: an 80 spikes/s unit moves by 16 spikes/s, a 1 spikes/s unit
moves by 0.2. So the mean difference in spikes/s is a **rate-weighted** average,
and its effective sample size is far below 40 (about 10 in this dataset; the top
five units carry 48% of the total absolute change). The paired t-test divides
that rate-weighted mean by an SD that the same few units inflate, and gets
t = -0.48, p = 0.63.

This is a **power** failure, not a Type I failure. Under a strict null the
paired t-test is essentially exact, because differences between exchangeable
pairs are symmetric. Over repeated experiments from this model the paired t-test
detects the real effect only about 43% of the time, against 94% for the Wilcoxon
signed-rank test and 99% for a paired t-test on log rates.

data/data.npz stores:
  counts_off  (n_units,)  spikes during the off condition
  counts_on   (n_units,)  spikes during the on condition
  duration    scalar, s spent in EACH condition

Run:  python generate_data.py
"""

import os
import numpy as np

SEED          = 144
N_NEURONS     = 40
RATE_MEDIAN   = 4.0    # spikes/s, median baseline rate
RATE_SIGMA    = 1.7    # lognormal sigma of baseline rates, natural log units
FACTOR_MEDIAN = 0.80   # the real effect: a 20% multiplicative decrease
FACTOR_SIGMA  = 0.30   # unit-to-unit variability in the effect, in log units
DURATION      = 600.0  # s in each condition


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rate_off = RATE_MEDIAN * np.exp(rng.normal(0.0, RATE_SIGMA, N_NEURONS))
    factor = FACTOR_MEDIAN * np.exp(rng.normal(0.0, FACTOR_SIGMA, N_NEURONS))
    rate_on = rate_off * factor

    counts_off = rng.poisson(rate_off * DURATION)
    counts_on = rng.poisson(rate_on * DURATION)

    np.savez(os.path.join(data_dir, "data.npz"),
             counts_off=counts_off, counts_on=counts_on, duration=DURATION)
    np.savez(os.path.join(data_dir, "ground_truth.npz"),
             rate_off=rate_off, factor=factor, seed=SEED,
             factor_median=FACTOR_MEDIAN, factor_sigma=FACTOR_SIGMA,
             note=np.array("The manipulation really does suppress firing: every "
                           "unit's rate is multiplied by a factor with median "
                           "0.80, a 20% decrease. The paired t-test misses it "
                           "because the effect is multiplicative while the test "
                           "is applied to differences in spikes/s, which are "
                           "proportional to rate. With lognormal rates that mean "
                           "difference is a rate-weighted average with an "
                           "effective sample size far below 40. This is a power "
                           "failure, not a Type I failure."))

    a, b = counts_off / DURATION, counts_on / DURATION
    print(f"{N_NEURONS} units, {DURATION/60:.0f} min per condition")
    print(f"baseline rates: median {np.median(a):.2f}, "
          f"range {a.min():.2f} to {a.max():.1f} spikes/s")
    print(f"mean rate {a.mean():.2f} -> {b.mean():.2f} spikes/s "
          f"({(b.mean()/a.mean()-1)*100:+.1f}%)")
    print(f"median per-unit change: {np.median(b/a - 1)*100:+.1f}%")
    print(f"units that decreased: {int(np.sum(b < a))}/{N_NEURONS}")


if __name__ == "__main__":
    main()
