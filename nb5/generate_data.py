"""
Generate the dataset for Notebook 5: "Simpson's paradox".

Firing rate vs. pupil size (arousal), pooled across several sessions. WITHIN
each session the relationship is negative (higher arousal -> lower rate), but
sessions differ in both their mean pupil and their baseline rate in a way that
makes the POOLED relationship positive.

data/data.npz stores per-trial: pupil, firing_rate, session_id.

Run:  python generate_data.py
"""

import os
import numpy as np

SEED          = 0
N_SESSIONS    = 7
PER_SESSION   = 60
WITHIN_SLOPE  = -12.0     # true within-session slope (spikes/s per unit pupil)
BETWEEN_SLOPE = 30.0      # baseline rate rises with a session's mean pupil
BASE_RATE     = 15.0      # spikes/s


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data"); os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    mean_pupil = np.linspace(0.25, 0.75, N_SESSIONS)   # sessions differ in arousal
    pupil, firing, session = [], [], []
    for s in range(N_SESSIONS):
        p = np.clip(rng.normal(mean_pupil[s], 0.03, PER_SESSION), 0, 1)
        baseline = BASE_RATE + BETWEEN_SLOPE * mean_pupil[s]
        f = baseline + WITHIN_SLOPE * (p - mean_pupil[s]) + rng.normal(0, 1.2, PER_SESSION)
        pupil += list(p); firing += list(f); session += [s] * PER_SESSION

    np.savez(os.path.join(data_dir, "data.npz"),
             pupil=np.array(pupil), firing_rate=np.array(firing),
             session_id=np.array(session))
    np.savez(os.path.join(data_dir, "ground_truth.npz"),
             within_slope=WITHIN_SLOPE, seed=SEED,
             note=np.array("true within-session slope is NEGATIVE; the positive "
                           "pooled slope is a between-session (Simpson's) artifact"))
    print(f"{N_SESSIONS} sessions x {PER_SESSION} trials = {len(pupil)} trials")


if __name__ == "__main__":
    main()
