# Notebook 2 — Selection bias in tuning curves (two acts)

**Pitfall:** circular analysis / double dipping + the winner's curse, shown as
an escalating pair.

Each neuron's preferred direction is chosen by its peak response and the
response is then read out *at that same direction*. Selecting on the data you
then measure biases the selected value upward (regression to the mean). Both
datasets are generated with **control and test identical** — there is no real
condition difference.

- **Act 1 — real tuning, fake difference** (`data/tuned.npz`): neurons have
  genuine von Mises tuning. The tuning is real and survives cross-validation,
  but the control>test difference at the preferred direction is entirely
  manufactured by selecting the preferred direction from the control data.
- **Act 2 — no tuning, fake everything** (`data/untuned.npz`): a flat Poisson
  population. The same pipeline invents the whole tuning curve *and* the
  difference from pure noise; cross-validation flattens everything.

Narrative: *selection can fake the difference between conditions even for
genuinely tuned neurons — and, pushed further, it can fake the appearance of
tuning itself.*

## Files

- `generate_data.py` — writes `data/tuned.npz`, `data/untuned.npz` (counts:
  neuron x ori x condition x trial; + duration/orientations/condition_names) and
  `data/ground_truth.npz`.
- `notebook2_student.ipynb` — reproduces the tuning + reduction from `tuned.npz`.
- `notebook2_solution.ipynb` — Act 1 (cross-validation keeps the tuning, kills
  the difference; the difference follows whichever condition you select on;
  spurious gap vs tuning breadth) and Act 2 (raw data untuned; cross-validation
  flattens everything; total-counting-time aside); ground truth + lesson.
- `hints.md`.

## Design notes

- 200 neurons, 12 directions (30 deg), 10 trials x 0.1 s window. Total counting
  time per direction = 1 s (~paper-magnitude winner's curse). More neurons keep
  the cross-validated curves clean at that small counting time.
- Preferred direction uses a **random tie-break** (discrete counts produce many
  exact ties; `np.argmax`'s first-index rule would bias the preferred direction
  toward 0 deg).
- Cross-validation is averaged over 60 random half-splits.

## Reproduce

```bash
conda activate swdb2026
python generate_data.py
python build_notebooks.py
```

## Realized numbers (seed 20260706)

Act 1 (tuned): naive control 6.46 vs test 3.79 sp/s (gap 2.67, p~1e-25);
cross-validated gap 0.22 with the tuning peak intact (~3.9 vs flank ~2.1).
Act 2 (untuned): naive peak 2.83 vs 0.98 (p~1e-30); cross-validated peak ~1.0
(flat). Spurious gap grows with tuning breadth (kappa 8->0.5: 1.5 -> 3.1 sp/s).
