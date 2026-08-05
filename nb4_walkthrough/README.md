# nb4 walkthrough — Nonsense correlations from slow fluctuations

A **guided demo to run together with a class**, not a mystery to solve. It is the
same pitfall as [`nb4/`](../nb4) (slow drift + a slowly-varying target defeats
ordinary statistics), extended to cover the pupil-correlation version of the
problem and to contrast controls that work with controls that only look like
they work.

Everything is simulated in the notebook: 100 neurons whose firing rates fluctuate
slowly (25 s timescale) around their own mean rate, spike trains drawn as an
inhomogeneous Poisson process, 1 s trials, a pupil trace with the same kind of
slow structure, and a block variable that flips every 30-70 trials. **The neurons
are generated independently of both the pupil and the block.**

## Files
- `notebook4_walkthrough.ipynb` — the whole thing (data generation included).
- `build_notebook.py` — rebuilds and re-executes it.

## Sections
1. Simulate a session (all parameters at the top, including `TP_NOISE_SD`)
2. Spike rasters
3. Spike counts binned by trial
4. The same matrix sorted by Rastermap — apparent structure, no cross-validation
   — then two ways to cross-validate the sort: an **interleaved** odd/even split
   (the structure replicates, because neighbouring trials share the same slow
   fluctuation) versus a **contiguous** half split (it does not)
5. **Test 1** — correlate each neuron with pupil diameter; histogram of *r* with
   the significant bars in red; histogram of *p*; matrix sorted by *r*; a
   clickable panel that overlays any neuron on the pupil trace
6. A control that **fails**: shuffle each neuron's timecourse
7. Controls that **work**: circular shift; another session's pupil trace
8. **How many independent samples do you actually have?** — Bartlett's effective
   *n* from the two autocorrelations, cross-checked against the width of the
   circular-shift null, and the honest correlation threshold that follows
9. **Test 2** — decode the block variable with ordinary k-fold CV; one fold shown
   with 90% training trials in black and the held-out 10% in green/red
10. The same failing control: shuffle each neuron's timecourse
11. Controls that **work**: pseudosessions; leave-one-block-out
12. **Do these controls still find real effects?** — 20 neurons given a genuine
    pupil drive (power vs. false positives for all three tests, as a function of
    effect size), and a population given genuine block coding
13. **How bad is it, as a function of the fluctuation timescale?** — both tests
    swept over timescales from 1 s to 50 s

## Realized numbers (seed 7, 511 trials)
| | result |
|---|---|
| Rastermap sort, mean correlation of adjacent neurons — held-out **even** trials | **+0.345** (fit +0.374; all pairs +0.004) |
| the same, held-out **second half** | **+0.088** (fit +0.379; all pairs +0.005) |
| critical \|r\| for p<0.05 at n=511 | 0.087 |
| neurons "significantly" correlated with pupil (p<0.05) | **79 / 100** |
| after shuffling trials | 7 / 100 |
| vs circular-shift null | 4 / 100 |
| vs session-permutation null | 8 / 100 |
| effective *n* (Bartlett, median over neurons) | **21** — and 15 from 1/var of the circular-shift null |
| honest \|r\| threshold at n=21 | **0.429** → 9 / 100 neurons significant |
| block decoding, trial-wise CV | **81.4%** |
| trial-wise CV on shuffled data | 50.0% ± 2.2% |
| pseudosession null | 80.4% ± 3.9% (p = 0.45) |
| leave-one-block-out CV | 10.3% (i.e. *worse* than chance) |

### Positive control (section 12)
With the 20 target neurons made fully pupil-locked, **all three tests find 20/20**
— but the parametric test also flags **71%** of the 80 untouched neurons, while the
circular-shift and session-permutation tests flag **5%**. Power ramps up with
effect size: the good controls reach ~50% detection when about 60% of a neuron's
slow modulation is pupil-locked, i.e. they are conservative, not blind.

For the block: leave-one-block-out climbs 9% → 21% → 78% → 97% as the real coding
strength goes 0 → 0.5 → 1.0 → 1.5 spikes/s, and the pseudosession test leaves its
own null band at 0.5 spikes/s — where leave-one-block-out is still reporting 21%,
i.e. worse than chance. Pseudosessions are the more sensitive of the two.

### Timescale sweep (section 13, 3 datasets per point)
| fluctuation timescale | 1 s | 4 s | 8 s | 16 s | 50 s |
|---|---|---|---|---|---|
| neurons "correlated with pupil" | 22% | 53% | 62% | 71% | **81%** |
| trial-wise CV | 65% | 82% | **88%** | 86% | 59% |
| leave-one-block-out | 42% | 32% | 23% | 7% | 2% |

The pupil false-positive rate climbs monotonically. The decoding artifact appears
as soon as the fluctuation outlasts a few trials and is **largest at intermediate
timescales**, so "our drift is very slow" is not a defence. Leave-one-block-out
never claims coding at any timescale.

## The point of the shuffle control
Permuting trials removes the slow autocorrelation that generated the effect, so
it tests a hypothesis nobody proposed. It comes out at chance, which makes the
original result look validated. A useful null has to preserve the temporal
structure of the data and only break its *alignment* with the variable of
interest — that is what the circular shift, the session permutation, the
pseudosessions and the block hold-out all do.

## Requirements
Beyond the base environment: `rastermap` (section 4) and `ipympl` (the clickable
figure in section 5). Both are in `../environment.yml`.

```bash
conda activate goodanalysis
python build_notebook.py       # regenerate + re-execute
```

The clickable figure needs a live kernel and uses `%matplotlib widget`. In the
committed notebook it appears as a static PNG; re-run the cell to make it
interactive. The cell right after it switches back to `%matplotlib inline`.

Running the whole notebook top to bottom takes about **4 minutes**. Almost all of
it is LDA refits: section 11a (100 pseudosessions), 12b (5 coding strengths x 40
pseudosessions) and 13 (7 timescales x 3 datasets). Lower `N_PSEUDO`,
`N_PSEUDO_SWEEP` and `N_REP` if you need it faster in front of a class.

## Note for later
Section 5 uses a **simulated** pupil trace. Swapping in a real pupil trace from
an Allen Institute dataset (Visual Coding / Visual Behavior Neuropixels) would
make the point harder to dismiss — a real behavioral signal has the same slow
structure and the same nonsense correlations follow. Also tracked in
[`../TODO.md`](../TODO.md).
