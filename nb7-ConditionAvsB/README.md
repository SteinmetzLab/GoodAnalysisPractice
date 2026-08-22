# Notebook 7 — Correlated observations (pseudo-replication)

**Pitfall:** p-values assume independent observations. Neurons recorded together
share brain state, so counting each as an independent sample understates the
standard error and overstates the evidence. The unit of analysis has to be the
unit you could have independently repeated.

400 neurons (4 sessions x 100) appear to fire more on condition A than condition
B at p = 2e-12. **There is no condition effect of any size in the generative
model**, and the A-vs-B difference is exactly zero in expectation.

## How the null is exactly true

Firing is `base + loading * arousal + noise`, with no condition term. Arousal
declines monotonically across each session (a drowsier animal). The A/B labels are
a **random balanced permutation** within each session, independent of arousal, of
trial order, and of firing.

So in any one session the A trials land by chance a little earlier or a little
later than the B trials. Earlier means higher arousal. That chance gap has mean 0
and SD ~0.31, so its sign is a coin flip, but because every neuron in the session
loads positively on arousal, **all 100 neurons inherit the same gap with the same
sign**. The 400 numbers entering the t-test are 4 coin flips copied 100 times
each.

This is what the earlier version of this notebook got wrong: it used
`CONFOUND_MEAN = 0.5`, so A trials really did have higher arousal in expectation,
and with all-positive loadings the population really did fire more on A. The claim
was true and only the inference was unjustified, which is a confounding lesson,
not a pseudo-replication one.

## Two results that are easy to guess wrong

**More sessions never help.** The naive false positive rate is flat from 2 to 32
sessions. The `sqrt(S)` cancels: the numerator's SD shrinks as `1/sqrt(S)` while
the wrong standard error shrinks as `1/sqrt(S*N)`, and the across-neuron SD does
not shrink with `S` at all. What survives is

    SD of the naive t under the null = sqrt(N) * sigma_shared / sigma_total

which here is `sqrt(100) * 0.309 / 0.550 = 5.6`, matching the measured
permutation null SD of ~6.0. It depends only on neurons **per session**. At one
neuron per session there is no pseudo-replication and the test is exactly
calibrated.

**Nothing rests on the seed.** The observed t = 7.30 is a 74th-percentile draw
from this model, with 25% of experiments at least as extreme, and the 3-to-1
session split is the modal outcome at about half of all experiments. Seed 75 was
chosen for legibility, not for significance. The argument has to rest on the rate,
since a hunted seed could produce a striking figure under a valid test too.

## Files
- `generate_data.py` -> `data/data.npz` (`activity` [sessions x neurons x
  trials], `condition`, `arousal`) and `data/ground_truth.npz`.
- `notebook7_student.ipynb` -> the naive across-neuron test (p = 2e-12).
- `notebook7_solution.ipynb` -> the 4 clusters and the ICC / design effect; the
  chance-arousal-gap mechanism per session; three correct analyses (session as
  unit, session-wise label permutation, regressing out arousal); the measured
  false positive rate with sweeps over neurons and sessions plus a positive
  control; and the demonstration that this dataset is a typical draw.
- `hints.md`.

## Reproduce
```bash
conda activate goodanalysis
python generate_data.py
python build_notebooks.py
```

## Realized numbers (seed 75)

Naive across 400 neurons t = 7.30, p = 1.5e-12, 66.2% of neurons show A > B.
Per-session mean A-B = [+0.384, -0.226, +0.196, +0.398] spikes/s (a 3-1 split),
from chance arousal gaps [+0.346, -0.215, +0.155, +0.465], which in turn come from
mean A-trial indices of 7.8, 10.4, 8.6, 7.4 against B-trial indices of 11.2, 8.6,
10.4, 11.6. Session level (n = 4) t = 1.29, p = 0.286. Session-wise label
permutation p = 0.231 with null SD 6.01 rather than 1. After regressing out
arousal t = -0.23, p = 0.819. ICC = 0.289, design effect 29.6, SE understated by
5.44x, effective sample size 13.5 rather than 400.

Across 1000 replications with no effect: naive p < 0.05 in 73.4%, p < 0.001 in
58.3%, with 49% of the hits favoring A (a coin flip); session level p < 0.05 in
5.9%.

Naive false positive rate at p < 0.05 by neurons per session (4 sessions): 5.2% at
1, 22.5% at 5, 57.0% at 25, 72.2% at 100. By session count (100 neurons each):
78.3%, 73.7%, 72.0%, 73.7%, 72.7% at 2, 4, 8, 16, 32 sessions. Flat.

Our observed t = 7.30 sits at the 74th percentile of the null distribution, with
25.5% of experiments at least as extreme; the 3-1 session split occurs in 49% of
experiments.

Positive control for the session-level test: a real condition effect is detected
in 5.7%, 56.7%, 98.7% and 100% of experiments at 0.0, 0.5, 1.0 and 1.5 spikes/s.
So the correct test is not merely conservative, though at n = 4 it does need the
effect to be around 1 spike/s to be reliable.
