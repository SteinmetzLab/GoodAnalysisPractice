"""
Generate the datasets for Notebook 2: "Selection bias in tuning curves".

Scenario: an orientation-tuning experiment run in two conditions -- a control
condition and a test condition (some manipulation) -- for a population of V1
neurons responding to drifting gratings at 12 directions (spaced 30 deg), with
10 repeats each counted in a brief (0.1 s) window.

Two populations are generated, both with **identical control and test**
(no real condition difference anywhere):

  data/tuned.npz    -- neurons with REAL von Mises tuning (Act 1). Selecting the
                       preferred direction from the control data manufactures a
                       control > test difference even though tuning is identical.
  data/untuned.npz  -- neurons with NO tuning at all (Act 2). The same pipeline
                       manufactures the entire tuning curve AND the difference
                       from pure Poisson noise.

Run:  python generate_data.py
"""

import os
import numpy as np

# ---- shared design ----
SEED       = 20260706
N_NEURONS  = 200
N_ORI      = 12            # 12 directions spaced 30 deg
ORI_STEP   = 30           # degrees
DURATION   = 0.1          # s counting window per presentation
N_TRIALS   = 10           # presentations per direction per condition
COND_NAMES = ['control', 'test']

# ---- Act 1: real tuning ----
BASELINE   = 1.0          # sp/s
AMP        = 4.0          # sp/s (peak = baseline + amp)
KAPPA      = 2.0          # von Mises concentration (tuning width)

# ---- Act 2: no tuning ----
RATE       = 1.0          # sp/s, flat everywhere


def poisson_counts(rng, mean_rate_by_ori):
    """mean_rate_by_ori: (neuron, ori). Returns counts (neuron, ori, cond, trial)
    with the SAME mean in both conditions (independent Poisson draws)."""
    mean_counts = mean_rate_by_ori[:, :, None, None] * DURATION
    return rng.poisson(mean_counts,
                       size=(N_NEURONS, N_ORI, len(COND_NAMES), N_TRIALS))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    orientations = np.arange(N_ORI) * ORI_STEP
    theta = np.deg2rad(orientations)

    # --- Act 1: real von Mises tuning, random preferred direction per neuron ---
    pref_dir = rng.uniform(0, 2 * np.pi, N_NEURONS)
    tuning = BASELINE + AMP * np.exp(KAPPA * (np.cos(theta[None, :]
                                                     - pref_dir[:, None]) - 1))
    counts_tuned = poisson_counts(rng, tuning)

    # --- Act 2: no tuning (flat rate) ---
    flat = np.full((N_NEURONS, N_ORI), RATE)
    counts_untuned = poisson_counts(rng, flat)

    common = dict(duration=DURATION,
                  orientations=orientations.astype(float),
                  condition_names=np.array(COND_NAMES))
    np.savez(os.path.join(data_dir, "tuned.npz"),
             counts=counts_tuned.astype(np.int64), **common)
    np.savez(os.path.join(data_dir, "untuned.npz"),
             counts=counts_untuned.astype(np.int64), **common)

    np.savez(
        os.path.join(data_dir, "ground_truth.npz"),
        seed=SEED,
        kappa=KAPPA, baseline=BASELINE, amp=AMP, rate=RATE,
        true_pref_dir_deg=np.rad2deg(pref_dir),
        note=np.array("control and test share identical rates in BOTH datasets: "
                      "no condition difference. tuned.npz has real von Mises "
                      "tuning; untuned.npz is flat (no tuning)."),
    )

    for name, c in [("tuned", counts_tuned), ("untuned", counts_untuned)]:
        print(f"{name}.npz: counts {c.shape}, grand-mean rate "
              f"{c.mean()/DURATION:.2f} sp/s")


if __name__ == "__main__":
    main()
