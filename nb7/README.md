# Notebook 7 — Correlated observations (pseudo-replication)

**Pitfall:** p-values assume independent observations. Neurons recorded together
share brain state, so they are correlated; counting each as an independent sample
inflates significance and overstates the evidence.

600 neurons (6 sessions x 100) appear to distinguish conditions A and B at
p ~ 1e-13. But the neurons carry no condition information: firing is
`baseline + loading*arousal + noise`, and in this dataset arousal was slightly
higher on A trials. Every neuron in a session shares that arousal, so the "600
observations" are really ~6 correlated clusters. At the session level (n = 6) the
effect is not significant (p ~ 0.5), and regressing out arousal removes it.

## Files
- `generate_data.py` -> `data/data.npz` (`activity` [sessions x neurons x
  trials], `condition`, `arousal`) and `data/ground_truth.npz`.
- `notebook7_student.ipynb` -> the naive across-neuron test (p ~ 1e-13).
- `notebook7_solution.ipynb` -> neurons cluster by session (mean within-session
  correlation ~0.5); the session-level test (n = 6, n.s.); regressing out
  arousal.
- `hints.md`.

## Reproduce
```bash
conda activate goodanalysis
python generate_data.py
python build_notebooks.py
```

## Realized numbers (seed 2)
Naive across 600 neurons p = 4e-13; per-session (n = 6) p = 0.51; residual
condition effect after regressing out arousal p = 0.68.
