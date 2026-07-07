# Notebook 1 — Clusters from nothing

**Pitfall:** apparent cluster structure where there is none. k-means with
`k = 3` is *forced* to return three groups, so it slices a continuous variable
(each neuron's response peak time, drawn uniformly at random) into three
contiguous bins and presents them as "early / middle / late" functional
classes. Per-cluster heatmaps and mean +/- SEM traces make the arbitrary slices
look real.

## Files

- `generate_data.py` — builds the dataset. Writes:
  - `data/activity.npz` — student-facing: `activity` (300 neurons x 100 time
    bins) and `time` (s).
  - `data/ground_truth.npz` — hidden generative params (continuous peak times);
    used only by the solution.
- `notebook1_student.ipynb` — the exercise (loads data, clusters, plots the
  false claim).
- `notebook1_solution.ipynb` — the reveal: peak-time histogram (uniform, not
  trimodal), full sorted heatmap (one continuum), PCA arch, silhouette vs k
  (no optimum at 3), and ground-truth confirmation.
- `build_notebooks.py` — regenerates both notebooks from source and executes
  them so outputs are embedded.

## Reproduce

```bash
conda activate swdb2026
python generate_data.py
python build_notebooks.py
```

## The reveal (spoiler)

There are no clusters. Peak time is a smooth continuum; k-means partitions it
into k contiguous cells regardless of whether gaps exist. PC1 correlates ~0.85
with peak time (the data lie on a 1-D arch) and the true peak times are uniform.
The silhouette score does peak at k=3, but the value is low (~0.23, at the
"no substantial structure" floor) and the same peak is reproduced by simulated
no-cluster data — so the peak is not evidence of clusters. The student notebook
runs silhouette and reports "selects k=3" without showing the plot or the low
value; the solution reveals both. The lesson: clustering output (and a "best k")
is never itself evidence that discrete classes exist — test it against a matched
continuous null before interpreting.
