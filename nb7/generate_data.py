"""
Generate the dataset for Notebook 7: "Correlated observations" (pseudo-
replication / non-independence).

Two conditions, A and B. Neurons do NOT encode the condition. But all neurons in
a session share an arousal signal, and in this dataset arousal happened to be a
bit higher on A trials than B trials (a session-specific confound). Because
every neuron tracks arousal, every neuron shows A > B -- and a test that treats
the 600 neurons as independent samples reports an astronomically small p-value,
even though the neurons are highly correlated and the real unit of replication is
the session.

data/data.npz stores:
  activity   (sessions, neurons, trials)
  condition  (sessions, trials)   +1 = A, -1 = B
  arousal    (sessions, trials)

Run:  python generate_data.py
"""

import os
import numpy as np

SEED       = 2
N_SESSIONS = 6
N_NEURONS  = 100      # per session
N_TRIALS   = 200      # per session (100 A + 100 B)
CONFOUND_MEAN = 0.5   # mean A-vs-B arousal difference (a chance confound)
CONFOUND_SD   = 1.0   # varies session to session


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data"); os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    d = rng.normal(CONFOUND_MEAN, CONFOUND_SD, N_SESSIONS)   # per-session A-B arousal gap
    activity = np.zeros((N_SESSIONS, N_NEURONS, N_TRIALS))
    condition = np.zeros((N_SESSIONS, N_TRIALS))
    arousal = np.zeros((N_SESSIONS, N_TRIALS))
    loadings = np.zeros((N_SESSIONS, N_NEURONS))
    for s in range(N_SESSIONS):
        cond = np.array([1] * (N_TRIALS // 2) + [-1] * (N_TRIALS // 2))
        a = rng.normal(0, 1, N_TRIALS) + (cond == 1) * d[s]     # A trials shifted up
        load = rng.uniform(0.5, 1.5, N_NEURONS)                 # all neurons track arousal
        # firing = baseline + loading*arousal + noise  (NO condition term)
        fr = 10 + load[:, None] * a[None, :] + rng.normal(0, 1, (N_NEURONS, N_TRIALS))
        activity[s], condition[s], arousal[s], loadings[s] = fr, cond, a, load

    np.savez(os.path.join(data_dir, "data.npz"),
             activity=activity, condition=condition, arousal=arousal)
    np.savez(os.path.join(data_dir, "ground_truth.npz"),
             loadings=loadings, arousal_gap=d, seed=SEED,
             note=np.array("no per-neuron condition coding; firing = baseline + "
                           "loading*arousal + noise. A>B is a shared arousal "
                           "confound, so neurons within a session are correlated."))
    print(f"{N_SESSIONS} sessions x {N_NEURONS} neurons = {N_SESSIONS*N_NEURONS} "
          f"neurons, {N_TRIALS} trials/session")


if __name__ == "__main__":
    main()
