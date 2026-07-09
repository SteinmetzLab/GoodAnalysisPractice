"""
Generate the dataset for Notebook 6: "Baseline normalization can invent an
increase" (averaging ratios / Jensen's inequality).

For each neuron we have spike counts in a baseline (pre) window and a response
(post) window, over a few repeats. There is NO real change: pre and post are
drawn from the SAME per-neuron Poisson rate. Normalizing each neuron by its own
(noisy) baseline and averaging the fold-changes nonetheless yields a mean
"fold-change" well above 1.

data/data.npz stores pre_counts, post_counts (neurons x trials) and duration.

Run:  python generate_data.py
"""

import os
import numpy as np

SEED     = 1
N_NEURONS = 200
N_TRIALS  = 5        # repeats per neuron in each window
DURATION  = 0.4      # s per window
RATE_LO, RATE_HI = 1.0, 8.0    # sp/s, per-neuron true rate (same pre and post)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data"); os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    true_rate = rng.uniform(RATE_LO, RATE_HI, N_NEURONS)         # same for pre & post
    lam = true_rate[:, None] * DURATION
    pre_counts  = rng.poisson(lam, (N_NEURONS, N_TRIALS))
    post_counts = rng.poisson(lam, (N_NEURONS, N_TRIALS))

    np.savez(os.path.join(data_dir, "data.npz"),
             pre_counts=pre_counts.astype(np.int64),
             post_counts=post_counts.astype(np.int64),
             duration=DURATION)
    np.savez(os.path.join(data_dir, "ground_truth.npz"),
             true_rate=true_rate, seed=SEED,
             note=np.array("pre and post drawn from the SAME per-neuron Poisson "
                           "rate: no change. A mean fold-change > 1 is the "
                           "ratio-averaging (Jensen) bias."))
    print(f"{N_NEURONS} neurons, {N_TRIALS} trials/window, {DURATION}s windows")


if __name__ == "__main__":
    main()
