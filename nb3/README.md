# Notebook 3 — You must cross-validate sorted plots

**Pitfall:** sorting neurons by a feature of the data (here, peak time) and then
displaying that same data manufactures structure — a diagonal "sequence" — even
from pure noise. Circular analysis, applied to a whole figure.

The data are i.i.d. Gaussian noise (`neurons x time x trials`), no sequence
anywhere. The student notebook trial-averages, smooths, finds each neuron's peak
time, sorts by it, and gets a crisp diagonal sweep (made crisper by `jet`).

## The fix

Choose the sort order on one set of trials and display an **independent** set. A
real sequence survives; a sorting artifact collapses to noise. Equivalently,
check that the sorting feature (peak time) is reproducible across independent
data — here the split-half peak-time correlation is ~0.

## Files

- `generate_data.py` — writes `data/activity.npz` (activity: neuron x time x
  trial; + time, dt) and `data/ground_truth.npz`.
- `notebook3_student.ipynb` — reproduces the sorted "sequence".
- `notebook3_solution.ipynb` — cross-validate the sort (diagonal vanishes);
  peak times don't replicate across halves; sorting fresh noise gives the same
  diagonal; a colormap aside (jet oversells it); ground truth + lesson.
- `hints.md`.

## Reproduce

```bash
conda activate swdb2026
python generate_data.py
python build_notebooks.py
```

## Note

This is the flip side of Notebook 1: there, sorting a heatmap by peak time
revealed a *real* continuous structure that survives cross-validation. Sorting is
a fine visualization — the error is treating the sorted picture as evidence
without checking it holds out.
