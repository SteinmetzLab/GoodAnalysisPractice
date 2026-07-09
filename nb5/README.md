# Notebook 5 — Simpson's paradox

**Pitfall:** a relationship pooled across groups can reverse the relationship
that holds within every group.

Firing rate vs. pupil (arousal), pooled across 7 sessions, slopes clearly
*positive* ("arousal increases firing"). But within every session the slope is
*negative* -- the sessions differ in both mean pupil and baseline rate, so the
between-session trend dominates the pool and flips the sign.

## Files
- `generate_data.py` -> `data/data.npz` (`pupil`, `firing_rate`, `session_id`)
  and `data/ground_truth.npz` (true within-session slope).
- `notebook5_student.ipynb` -> the pooled positive fit.
- `notebook5_solution.ipynb` -> color by session; within-session slopes (all
  negative); the fix (within-session centering / per-session intercepts).
- `hints.md`.

## Reproduce
```bash
conda activate goodanalysis
python generate_data.py
python build_notebooks.py
```

## Realized numbers (seed 0)
Pooled slope +28.5, mean within-session slope -13.9 (true -12).
