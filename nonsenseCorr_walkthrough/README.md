# Walkthrough: Nonsense correlations from slow fluctuations

A **guided demo to run together with a class**, not a mystery to solve. It is the
same pitfall as [`nb4-BlockDecoding/`](../nb4-BlockDecoding) (slow drift plus a
slowly varying target defeats ordinary statistics), extended to cover the
pupil-correlation version of the problem and to contrast controls that work with
controls that only look like they work.

Everything is simulated in the notebook: 100 neurons whose firing rates fluctuate
slowly (25 s timescale) around their own mean rate, spike trains drawn as an
inhomogeneous Poisson process, 1 s trials, a pupil trace with the same kind of
slow structure, and a block variable that flips every 30 to 70 trials. **The
neurons are generated independently of both the pupil and the block.**

This version was developed for the Summer Workshop on the Dynamic Brain 2026 by
Nick Steinmetz and Carrie Stine. The SWDB copy lives in
[AllenSWDB/SWDB_2026_Good_Analysis](https://github.com/AllenSWDB/SWDB_2026_Good_Analysis/tree/main/code/Workshop-2_NonsenseCorrelations).
The earlier version it replaces (`notebook4_walkthrough.ipynb`, built by
`build_notebook.py`, which also had a Rastermap section) is in this repo's git
history at commit `50828b4`.

## Files
- `nonsenseCorr_walkthrough.ipynb`: the notebook (data generation included). It
  contains short exercises with `_____` blanks; their cells are committed without
  outputs.
- `workshop2_utils.py`: helper functions the notebook imports.
- `resources/`: two diagrams of the circular-shift control.

## Parts
1. **Simulating null data**: simulate a session, spike rasters, spike counts
   binned by trial.
2. **Correlation with pupil diameter**: per-neuron correlation, a clickable
   matrix that overlays any neuron on the pupil trace, the shuffle control (fails),
   circular shift and another session's pupil trace (work), and the effective
   number of independent samples.
3. **Decoding the block variable**: ordinary k-fold cross-validation, the shuffle
   control (fails), pseudosessions and leave-one-block-out (work).
4. **Detection of true effects**: do the good controls still find real pupil and
   block effects?
5. **Severity across timescales**: both tests swept over fluctuation timescales
   from 1 s to 50 s.

## Realized numbers (seed 7, 511 trials)
| | result |
|---|---|
| critical \|r\| for p<0.05 at n=511 | 0.087 |
| neurons "significantly" correlated with pupil (p<0.05) | **79 / 100** |
| after shuffling trials | 7 / 100 |
| vs circular-shift null | 4 / 100 |
| vs session-permutation null | 8 / 100 |
| effective *n* (Bartlett, median over neurons) | **21**, and 15 from 1/var of the circular-shift null |
| honest \|r\| threshold at n=21 | **0.429**, giving 9 / 100 neurons significant |
| block decoding, trial-wise CV | **82.8%** |
| trial-wise CV on shuffled data | 50.4% ± 2.6% |
| pseudosession null | 81.9% ± 3.7% (p = 0.60) |
| leave-one-block-out CV | 7.3% (i.e. *worse* than chance) |

### Real block coding (Part 4)
| block coding strength (spikes/s) | 0 | 0.25 | 0.5 | 1.0 | 1.5 |
|---|---|---|---|---|---|
| trial-wise CV | 82.6% | 83.0% | 89.8% | 99.4% | 100% |
| leave-one-block-out | 6.4% | 7.0% | 15.2% | 75.2% | 96.3% |
| pseudosession p | 0.32 | 0.37 | 0.024 | 0.024 | 0.024 |

With 40 pseudosessions the smallest possible p is 1/41 = 0.024, so the
pseudosession test is at its floor from 0.5 spikes/s onward, where
leave-one-block-out is still reporting worse than chance.

### Timescale sweep (Part 5, 3 datasets per point)
| fluctuation timescale | 1 s | 4 s | 8 s | 16 s | 50 s |
|---|---|---|---|---|---|
| neurons "correlated with pupil" | 22% | 53% | 62% | 71% | **81%** |
| trial-wise CV | 67% | 84% | **90%** | 88% | 58% |
| leave-one-block-out | 40% | 31% | 22% | 5% | 1% |

The pupil false-positive rate climbs monotonically. The decoding artifact appears
as soon as the fluctuation outlasts a few trials and is **largest at intermediate
timescales**, so "our drift is very slow" is not a defense.

## The point of the shuffle control
Permuting trials removes the slow autocorrelation that generated the effect, so
it tests a hypothesis nobody proposed. It comes out at chance, which makes the
original result look validated. A useful null has to preserve the temporal
structure of the data and only break its *alignment* with the variable of
interest. That is what the circular shift, the session permutation, the
pseudosessions and the block hold-out all do.

## Requirements
The base environment plus `ipympl` for the clickable figure in Part 2. Both are
covered by `../environment.yml` (`jupyter` brings in `ipywidgets`).

The clickable figure needs a live kernel and uses `%matplotlib widget`. In the
committed notebook it appears as a static PNG; re-run the cell to make it
interactive. The cell right after it switches back to `%matplotlib inline`.

Running the whole notebook top to bottom takes about 25 s on an Apple M3 Max.

## License note
The notebook, `workshop2_utils.py` and `resources/` come from
AllenSWDB/SWDB_2026_Good_Analysis, which is released under the MIT License,
Copyright (c) 2023 Allen Institute for Neural Dynamics. The full license text is in `LICENSE_AllenSWDB`.
