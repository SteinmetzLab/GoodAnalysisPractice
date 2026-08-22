"""
Generate the dataset for Notebook 7: "Correlated observations" (pseudo-
replication / the wrong unit of analysis).

The null is exactly true
------------------------
Condition labels are a **random balanced permutation** within each session (10 A
and 10 B among 20 trials), drawn independently of everything else. Firing is

    activity = BASE_RATE + loading * pupil_diameter + noise

with **no condition term**, and pupil diameter does not depend on condition. So
the marginal A-vs-B firing difference is exactly zero in expectation: over a
million sessions there is no effect of any size, and P(A > B) = 1/2.

Why a real experiment still finds one
-------------------------------------
Pupil diameter declines monotonically across each session, as an animal getting
sated or drowsy would. Because the A/B labels are shuffled, in any one session the
A trials land, by chance, slightly earlier or slightly later than the B trials. If
they land earlier, A trials caught a larger pupil, and since every neuron in the
session loads positively on pupil diameter, **every neuron in that session shows
A > B.** The chance gap has mean 0 and SD ~0.31 across sessions, so its sign is a
coin flip, but its effect is shared by all 100 neurons recorded together.

Test 400 neurons as if they were 400 independent observations and this chance
session-level gap is significant about 58% of the time at p < 0.001, with a
random sign. Test 4 sessions as 4 observations and the false positive rate is the
nominal 5%. Adding sessions does not help the naive test: the inflation grows
exactly as fast as the mean shrinks.

data/data.npz stores:
  activity        (sessions, neurons, trials)  firing rate, spikes/s
  condition       (sessions, trials)           +1 = A, -1 = B, shuffled per session
  pupil_diameter  (sessions, trials)           z-scored pupil diameter per trial

Run:  python generate_data.py
"""

import os
import numpy as np

SEED        = 75
N_SESSIONS  = 4
N_NEURONS   = 100     # per session
N_TRIALS    = 20      # per session, 10 A + 10 B in random order
BASE_RATE   = 10.0    # spikes/s
SLOPE_LO    = 0.7     # per-session amplitude of the monotonic pupil decline
SLOPE_HI    = 1.3
PUPIL_SD    = 0.3     # trial-to-trial pupil jitter on top of the decline
LOAD_LO     = 0.5     # every neuron loads POSITIVELY on pupil diameter
LOAD_HI     = 1.5
SPIKE_SD    = 1.0     # independent per-neuron, per-trial noise, spikes/s


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # Pupil declines from +slope to -slope over the session (drowsier over time)
    ramp = 1.0 - 2.0 * np.arange(N_TRIALS) / (N_TRIALS - 1)

    activity  = np.zeros((N_SESSIONS, N_NEURONS, N_TRIALS))
    condition = np.zeros((N_SESSIONS, N_TRIALS), dtype=np.int64)
    pupil     = np.zeros((N_SESSIONS, N_TRIALS))
    loadings  = np.zeros((N_SESSIONS, N_NEURONS))

    for s in range(N_SESSIONS):
        p = rng.uniform(SLOPE_LO, SLOPE_HI) * ramp
        p = p + rng.normal(0.0, PUPIL_SD, N_TRIALS)

        # Balanced random assignment: independent of pupil, of trial order,
        # and of firing. This is what makes the null exactly true.
        cond = np.repeat([1, -1], N_TRIALS // 2)
        rng.shuffle(cond)

        load = rng.uniform(LOAD_LO, LOAD_HI, N_NEURONS)     # all positive
        fr = (BASE_RATE + load[:, None] * p[None, :]
              + rng.normal(0.0, SPIKE_SD, (N_NEURONS, N_TRIALS)))

        activity[s], condition[s], pupil[s], loadings[s] = fr, cond, p, load

    gap = np.array([pupil[s][condition[s] == 1].mean()
                    - pupil[s][condition[s] == -1].mean()
                    for s in range(N_SESSIONS)])

    np.savez(os.path.join(data_dir, "data.npz"),
             activity=activity, condition=condition, pupil_diameter=pupil)
    np.savez(os.path.join(data_dir, "ground_truth.npz"),
             loadings=loadings, pupil_gap=gap, seed=SEED,
             note=np.array("There is NO condition term in the generative model: "
                           "activity = base + loading*pupil_diameter + noise, and "
                           "the A/B labels are a random balanced permutation "
                           "independent of pupil diameter. The marginal A-vs-B "
                           "difference is exactly zero in expectation. In each "
                           "session the A trials happened to land earlier or later "
                           "than the B trials, which under a declining pupil trend "
                           "gives that session a chance pupil gap that every neuron "
                           "in it inherits."))

    print(f"{N_SESSIONS} sessions x {N_NEURONS} neurons = {N_SESSIONS*N_NEURONS} "
          f"neurons, {N_TRIALS} trials/session")
    print(f"realized chance pupil gap per session: {np.round(gap, 3)}")
    print(f"  (mean 0 by construction; sign is a coin flip)")


if __name__ == "__main__":
    main()
