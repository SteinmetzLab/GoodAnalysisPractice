# Notebook 4 — Decoding a slow variable under drift

**Pitfall:** ordinary (trial-wise) cross-validation is invalid for decoding a
*slowly-varying* task variable when the neural data have slow **drift**. You can
"decode" a variable the neurons do not encode.

A binary **block** variable stays constant for 30-70 trials then flips
(-1/+1). `dataset_1` is 100 neurons of pure slow drift (temporally smoothed
noise), independent of the block. Standard k-fold CV nonetheless decodes the
block at ~95%, because each held-out trial sits next to same-block training
trials in a similar drift state -- the decoder reads the drift and interpolates.

## Controls (and their limits)

- **Leave-one-block-out CV** (hold out a contiguous block) -> back to chance on
  `dataset_1`; recovers real coding on `dataset_2`.
- **Pseudosession null**: decode many surrogate block sequences with the same
  statistics but independent of the neurons; the drift decodes those about as
  well, so the real accuracy is unremarkable. Real coding (`dataset_2`) beats
  the null.
- **Advanced:** under strong drift these controls become conservative and can
  miss weak-but-real coding (false negatives); and pseudosessions need
  **randomized** block lengths -- fixed lengths give a degenerate null.

## Files

- `generate_data.py` -> `data/dataset_1.npz` (drift, no coding), `data/dataset_2.npz`
  (real coding), `data/ground_truth.npz`. Each stores `activity` (neurons x
  trials), `block_values`, `block_ids`.
- `notebook4_student.ipynb` -> decodes `dataset_1` at ~95% and claims coding.
- `notebook4_solution.ipynb` -> leave-block-out CV, mechanism (drift), a
  pseudosession null, the `dataset_2` positive control, and the advanced section
  on conservative controls + block-design requirements.
- `hints.md`.

Decoder: `sklearn` LinearDiscriminantAnalysis.

## Reproduce

```bash
conda activate goodanalysis   # numpy scipy scikit-learn matplotlib jupyter
python generate_data.py
python build_notebooks.py
```

## Realized numbers (seed 6)

`dataset_1` (drift): trial-wise CV 97.7%, leave-block-out 58.1% (chance),
pseudosession null ~95% (real not above it). `dataset_2` (coding): trial-wise
100%, block-out 100%, pseudosession p = 0.007. Regime 3: leave-block-out holds
until the block signal is ~1% of drift, then collapses below chance while the
oracle axis still reads ~73%.
