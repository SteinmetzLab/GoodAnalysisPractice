"""
Generate the dataset for Notebook 8: "A t-test on strongly skewed data with
unequal group sizes."

What is simulated
-----------------
One recording, 150 well-isolated units, each with a spike count over a 30 min
session. Units are labeled by anatomical region: 10 landed in region A (a small
target structure) and 140 in region B (the surrounding tissue).

**Every unit's rate is drawn from the SAME lognormal distribution**, median
4 spikes/s with sigma = 1.2 in natural log units, which puts the middle 90% of
units between about 0.6 and 29 spikes/s. Region is assigned first and has no
effect on rate whatsoever, so the true A-vs-B difference is exactly zero.

Why a t-test still finds a difference
-------------------------------------
Firing rates across neurons are lognormal, spanning orders of magnitude. Two
things follow:

1. A sample of 10 units usually **misses the heavy right tail**, so its sample
   mean comes out below the population mean.
2. Its sample standard deviation comes out low **at the same time**, because the
   same missing large values drive both. Sample mean and sample SD are strongly
   positively correlated for skewed data.

Welch's t-test divides by each group's own SD, so a group that missed the tail
gets a small mean *and* a small standard error: the test reads the missing tail
as precision. The result is a confident, systematically-signed false positive in
which the small group appears to fire much less.

Measured on this generative model: **Welch's t-test rejects at p < 0.05 about
17% of the time** when the null is exactly true, and in 99% of those cases the
small group appears lower. Student's pooled t-test (5.5%), the Mann-Whitney test
(5.0%) and a t-test on log rates (5.4%) are all correctly calibrated. With
**equal** group sizes even Welch is fine (3.5% at 30 vs 30), in fact slightly
conservative. It is the combination of skew and imbalance that breaks it.

data/data.npz stores:
  spike_counts  (n_units,)  total spikes in the session
  duration      scalar, s
  region        (n_units,)  0 = region A, 1 = region B

Run:  python generate_data.py
"""

import os
import numpy as np

SEED        = 290
N_A         = 10       # units in region A (a small structure)
N_B         = 140      # units in region B
RATE_MEDIAN = 4.0      # spikes/s
RATE_SIGMA  = 1.2      # lognormal sigma, natural log units
DURATION    = 1800.0   # s, 30 min recording


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    n_units = N_A + N_B
    # One population. Region plays no part in generating the rate.
    true_rate = RATE_MEDIAN * np.exp(rng.normal(0.0, RATE_SIGMA, n_units))
    spike_counts = rng.poisson(true_rate * DURATION)

    region = np.concatenate([np.zeros(N_A, dtype=np.int64),
                             np.ones(N_B, dtype=np.int64)])

    np.savez(os.path.join(data_dir, "data.npz"),
             spike_counts=spike_counts, duration=DURATION, region=region)
    np.savez(os.path.join(data_dir, "ground_truth.npz"),
             true_rate=true_rate, seed=SEED,
             rate_median=RATE_MEDIAN, rate_sigma=RATE_SIGMA,
             note=np.array("All 150 units' rates are drawn from ONE lognormal "
                           "distribution (median 4 spikes/s, sigma 1.2). Region "
                           "is assigned independently and has no effect on rate: "
                           "the true A-vs-B difference is exactly zero. Welch's "
                           "t-test is fooled because a sample of 10 from a "
                           "heavy-tailed distribution usually misses the tail, "
                           "which makes its mean AND its SD too small at once, "
                           "and the test reads the small SD as precision."))

    rate = spike_counts / DURATION
    a, b = rate[region == 0], rate[region == 1]
    print(f"{n_units} units ({N_A} in region A, {N_B} in region B), "
          f"{DURATION/60:.0f} min recording")
    print(f"all units: median {np.median(rate):.2f}, "
          f"range {rate.min():.2f} to {rate.max():.1f} spikes/s")
    print(f"region A : mean {a.mean():.2f}, median {np.median(a):.2f} spikes/s")
    print(f"region B : mean {b.mean():.2f}, median {np.median(b):.2f} spikes/s")
    print(f"  means differ by {(a.mean()/b.mean() - 1)*100:+.0f}% while the "
          f"medians differ by {(np.median(a)/np.median(b) - 1)*100:+.0f}%")


if __name__ == "__main__":
    main()
