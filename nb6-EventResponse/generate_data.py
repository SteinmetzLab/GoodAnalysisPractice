"""
Generate the dataset for Notebook 6: "Baseline normalization can invent an
increase" (averaging ratios / Jensen's inequality).

What is simulated
-----------------
A single session: `N_NEURONS` units recorded continuously for ~52 s, plus
`N_EVENTS` event times. Every unit is a homogeneous Poisson process with a
constant rate for the whole recording, so **no unit responds to the event and
nothing changes at t = 0.** Per-unit rates are lognormal (median ~3 spikes/s),
truncated below at RATE_FLOOR to mimic a routine "exclude very low-rate units"
QC step. The truncation also guarantees that essentially no unit has zero
baseline spikes, which keeps the exercise free of a division-by-zero side plot.

Why this still produces a convincing "response"
-----------------------------------------------
Build a PSTH, divide each unit's PSTH by that unit's own baseline (estimated
from the pre-event bins of the same trials), and average across units. Two
things happen at once:

1. Every post-event bin is inflated by roughly 1 + 1/(baseline spike count),
   because averaging a ratio with a noisy denominator is biased upward
   (Jensen's inequality: E[X/Y] > E[X]/E[Y]).
2. The baseline bins are *not* inflated. They are pinned to exactly 1: if the
   denominator is the mean of those same bins, their mean ratio is 1 by
   construction, for every unit, with no noise at all.

The result is a flat, immaculate baseline followed by a sharp step at t = 0,
which is exactly the signature of a real response.

Stored in data/data.npz, in the standard spike-sorting layout:
  spike_times      (n_spikes,)  s, ascending
  spike_clusters   (n_spikes,)  unit index for each spike
  event_times      (n_events,)  s
  n_neurons        scalar
  recording_duration  scalar, s

Run:  python generate_data.py
"""

import os
import numpy as np

SEED        = 1
N_NEURONS   = 2000     # Neuropixels-scale unit count for one session
N_EVENTS    = 20       # event repeats
RATE_MEDIAN = 2.5      # spikes/s, median of the lognormal rate distribution
RATE_SIGMA  = 1.1      # lognormal sigma (in natural log units)
RATE_FLOOR  = 0.8      # spikes/s, QC threshold: units below this are excluded
ISI_LO      = 2.0      # s, inter-event interval drawn uniformly from
ISI_HI      = 3.0      # s   [ISI_LO, ISI_HI]
PAD_START   = 1.2      # s before the first event
PAD_END     = 2.2      # s after the last event


def draw_rates(rng, n, median, sigma, floor):
    """Lognormal rates, rejection-sampled so all are >= floor."""
    out = np.empty(n)
    filled = 0
    while filled < n:
        cand = median * np.exp(rng.normal(0.0, sigma, 2 * n))
        cand = cand[cand >= floor]
        take = min(len(cand), n - filled)
        out[filled:filled + take] = cand[:take]
        filled += take
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    true_rate = draw_rates(rng, N_NEURONS, RATE_MEDIAN, RATE_SIGMA, RATE_FLOOR)

    isi = rng.uniform(ISI_LO, ISI_HI, N_EVENTS - 1)
    event_times = PAD_START + np.concatenate([[0.0], np.cumsum(isi)])
    duration = event_times[-1] + PAD_END

    # Homogeneous Poisson per unit: draw the spike count, then place the spikes
    # uniformly in time. Constant rate throughout, so no event-locked response.
    times, clusters = [], []
    for n in range(N_NEURONS):
        n_spk = rng.poisson(true_rate[n] * duration)
        times.append(np.sort(rng.uniform(0.0, duration, n_spk)))
        clusters.append(np.full(n_spk, n))
    spike_times = np.concatenate(times)
    spike_clusters = np.concatenate(clusters)

    order = np.argsort(spike_times, kind="stable")
    spike_times = spike_times[order].astype(np.float32)
    spike_clusters = spike_clusters[order].astype(np.int16)

    np.savez_compressed(
        os.path.join(data_dir, "data.npz"),
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        event_times=event_times,
        n_neurons=N_NEURONS,
        recording_duration=duration,
    )
    np.savez_compressed(
        os.path.join(data_dir, "ground_truth.npz"),
        true_rate=true_rate, seed=SEED,
        note=np.array("Every unit is a homogeneous Poisson process with a "
                      "constant rate for the whole recording. No unit responds "
                      "to the event; the true event-locked modulation is exactly "
                      "zero. Any apparent step in a baseline-normalized PSTH is "
                      "the ratio-averaging (Jensen) bias, plus the fact that the "
                      "baseline bins are pinned to 1 by construction."))

    print(f"{N_NEURONS} units, {N_EVENTS} events, {duration:.1f} s recording, "
          f"{len(spike_times):,} spikes")
    print(f"rate: median {np.median(true_rate):.2f}, mean {true_rate.mean():.2f}, "
          f"range {true_rate.min():.2f}-{true_rate.max():.1f} spikes/s")


if __name__ == "__main__":
    main()
