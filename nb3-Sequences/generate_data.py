"""
Generate the dataset for Notebook 3: "You must cross-validate sorted plots".

The data are trial-resolved "activity" for a population of neurons over a short
interval: shape (neurons, time, trials). Every value is INDEPENDENT Gaussian
noise -- there is no temporal sequence and no consistent structure of any kind.

The student notebook smooths the trial-averaged activity, finds each neuron's
peak time, sorts the neurons by peak time, and plots the result -- producing a
gorgeous diagonal "sequence" out of pure noise.

Files:
  data/activity.npz      -> what students receive (activity + time)
  data/ground_truth.npz  -> note that it is pure noise (for the solution)

Run:  python generate_data.py
"""

import os
import numpy as np

SEED       = 3
N_NEURONS  = 100
N_TIME     = 100
DT         = 0.02          # s per bin  -> 2 s interval
N_TRIALS   = 10            # repeats (needed to cross-validate the sort)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    time = np.arange(N_TIME) * DT

    # Pure i.i.d. Gaussian noise: no sequence, no cross-trial consistency.
    activity = rng.normal(0.0, 1.0, size=(N_NEURONS, N_TIME, N_TRIALS))

    np.savez(os.path.join(data_dir, "activity.npz"),
             activity=activity.astype(np.float32),
             time=time.astype(float), dt=DT)

    np.savez(os.path.join(data_dir, "ground_truth.npz"),
             seed=SEED,
             note=np.array("i.i.d. Gaussian noise for every neuron, time bin, "
                           "and trial: no temporal sequence. Any diagonal after "
                           "peak-sorting is a selection artifact."))

    print(f"Wrote activity.npz: shape {activity.shape} "
          f"(neuron x time x trial), {time[0]:.2f}-{time[-1]:.2f} s")


if __name__ == "__main__":
    main()
