"""
Generate the dataset for Notebook 1: "Clusters from nothing".

The data are a (neurons x time) matrix. Every neuron is independent white
noise with a single smooth Gaussian "response" bump added at a randomly
chosen peak time. Crucially, the peak times are drawn from a CONTINUOUS
uniform distribution -- there is NO discrete group structure in the data.

Two files are written:
  data/activity.npz      -> what the students receive (activity + time only)
  data/ground_truth.npz  -> hidden generative parameters, used by the
                            solution notebook only.

Run:  python generate_data.py
"""

import os
import numpy as np

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
SEED         = 2026        # reproducibility
N_NEURONS    = 300         # number of neurons (rows)
N_TIME       = 100         # number of time bins (columns)
DT           = 0.05        # seconds per bin -> 5 s trial
NOISE_SD     = 1.0         # sd of the additive white noise
PEAK_AMP     = 4.0         # height of the response bump (in units of noise sd)
PEAK_WIDTH   = 9.0         # sd of the Gaussian bump, in time bins
# Keep peaks away from the very edges so the bump is fully visible.
PEAK_MARGIN  = 15          # bins of margin at each edge


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)

    time = np.arange(N_TIME) * DT  # seconds

    # --- Continuous, uniformly distributed peak times (the ground truth) ---
    # This is the key point: peak time is a CONTINUOUS variable with no gaps
    # or modes. There are no "early / middle / late" groups in the generator.
    peak_bins = rng.uniform(PEAK_MARGIN, N_TIME - 1 - PEAK_MARGIN, size=N_NEURONS)

    # --- Build the activity matrix ---
    # One smooth Gaussian bump per neuron at its peak time.
    t_idx = np.arange(N_TIME)[None, :]                 # (1, T)
    centers = peak_bins[:, None]                        # (N, 1)
    bumps = PEAK_AMP * np.exp(-0.5 * ((t_idx - centers) / PEAK_WIDTH) ** 2)

    # Small per-neuron amplitude jitter so bumps aren't all identical height
    # (a continuous nuisance -- still nothing that creates groups).
    amp_jitter = rng.uniform(0.7, 1.3, size=(N_NEURONS, 1))

    # White Gaussian noise everywhere, plus the (jittered) bump.
    noise = rng.normal(0.0, NOISE_SD, size=(N_NEURONS, N_TIME))
    activity = noise + amp_jitter * bumps

    # --- Save student-facing data (no ground truth leaked) ---
    np.savez(
        os.path.join(data_dir, "activity.npz"),
        activity=activity.astype(np.float64),
        time=time.astype(np.float64),
    )

    # --- Save hidden ground truth for the solution notebook ---
    np.savez(
        os.path.join(data_dir, "ground_truth.npz"),
        peak_bins=peak_bins,
        peak_times=peak_bins * DT,
        amp_jitter=amp_jitter.ravel(),
        seed=SEED,
        params=np.array([N_NEURONS, N_TIME, DT, NOISE_SD, PEAK_AMP, PEAK_WIDTH],
                        dtype=float),
    )

    print(f"Wrote activity.npz: activity shape {activity.shape}, "
          f"time {time[0]:.2f}-{time[-1]:.2f} s")
    print(f"Wrote ground_truth.npz: {N_NEURONS} continuous peak times "
          f"in [{peak_bins.min()*DT:.2f}, {peak_bins.max()*DT:.2f}] s")


if __name__ == "__main__":
    main()
