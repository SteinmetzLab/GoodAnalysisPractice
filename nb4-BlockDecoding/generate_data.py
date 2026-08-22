"""
Generate the datasets for Notebook 4: "Decoding a slow variable under drift".

A task has a binary "block" variable that stays constant for a run of trials
(30-70, uniformly random) then flips: -1, +1, -1, ... Two populations of 100
neurons are recorded across the same block sequence:

  data/dataset_1.npz  -> NO block coding: activity is pure slow drift
                         (temporally-smoothed noise), independent of the block.
  data/dataset_2.npz  -> REAL block coding: a shared rank-1 block signal + noise,
                         no drift.

The student notebook receives dataset_1 and "decodes" the block at ~95% with
ordinary (trial-wise) cross-validation -- an artifact of the slow drift.

Each file stores:
  activity      (neurons x trials)
  block_values  (trials,)  the +/-1 block variable
  block_ids     (trials,)  which block each trial belongs to (0..9)

Run:  python generate_data.py
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter1d

SEED        = 6
N_NEURONS   = 100
N_BLOCKS    = 10
LEN_RANGE   = (30, 70)     # block length, trials (uniform, inclusive)
NOISE_SCALE = 0.2
DRIFT_SIGMA = 25           # trials; temporal smoothing that creates slow drift


def make_blocks(rng, n_blocks=N_BLOCKS, lo=LEN_RANGE[0], hi=LEN_RANGE[1]):
    """Alternating +/-1 blocks with uniform-random lengths and a random start
    sign. Returns (block_values, block_ids)."""
    sign = rng.choice([-1, 1])
    values, ids = [], []
    for i, L in enumerate(rng.integers(lo, hi + 1, size=n_blocks)):
        values += [sign * (-1) ** i] * int(L)
        ids += [i] * int(L)
    return np.array(values), np.array(ids)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    block_values, block_ids = make_blocks(rng)
    n_trials = len(block_values)

    # dataset_1: NO coding -- slow drift only (smoothed noise, block-independent)
    drift = gaussian_filter1d(rng.normal(0, 1, (N_NEURONS, n_trials)) * NOISE_SCALE,
                              DRIFT_SIGMA, axis=1)

    # dataset_2: REAL coding -- shared rank-1 block signal + white noise (no drift)
    weights = rng.uniform(-0.5, 0.5, N_NEURONS)
    coding = np.outer(weights, block_values) + rng.normal(0, 1, (N_NEURONS, n_trials)) * NOISE_SCALE

    common = dict(block_values=block_values, block_ids=block_ids)
    np.savez(os.path.join(data_dir, "dataset_1.npz"),
             activity=drift.astype(np.float64), **common)
    np.savez(os.path.join(data_dir, "dataset_2.npz"),
             activity=coding.astype(np.float64), **common)
    np.savez(os.path.join(data_dir, "ground_truth.npz"),
             seed=SEED, weights=weights,
             note=np.array("dataset_1: no block coding, slow drift only. "
                           "dataset_2: real rank-1 block coding, no drift. "
                           "Both share the same block sequence."))

    print(f"{n_trials} trials, {N_BLOCKS} blocks, {N_NEURONS} neurons")
    print("wrote dataset_1.npz (drift only) and dataset_2.npz (real coding)")


if __name__ == "__main__":
    main()
