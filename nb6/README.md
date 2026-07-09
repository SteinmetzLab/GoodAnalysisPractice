# Notebook 6 — Baseline normalization can invent an increase

**Pitfall:** averaging per-unit ratios (fold-changes) is biased upward when the
denominator (the baseline) is noisily estimated -- Jensen's inequality,
E[post/pre] > E[post]/E[pre].

Baseline and response are drawn from the *same* per-neuron rate (no change), but
normalizing each neuron to its own noisy baseline and averaging gives a mean
fold-change of ~1.33 ("a 30% increase"). Neurons that happen to have a small
baseline get a huge ratio, creating a long right tail that drags the mean above 1
while the median stays at ~1.

## Files
- `generate_data.py` -> `data/data.npz` (`pre_counts`, `post_counts`,
  `duration`) and `data/ground_truth.npz` (true per-neuron rate).
- `notebook6_student.ipynb` -> the ~1.3x mean fold-change.
- `notebook6_solution.ipynb` -> fold-change vs baseline; right-skew (mean vs
  median); bias-free summaries (difference, ratio-of-means, log); bias vs number
  of baseline trials.
- `hints.md`.

## Reproduce
```bash
conda activate goodanalysis
python generate_data.py
python build_notebooks.py
```

## Realized numbers (seed 1)
Mean fold-change 1.33, median 1.08, ratio-of-means 1.07, mean difference +0.29
spikes/s, mean log2 fold-change +0.12 -- i.e. no real change.
