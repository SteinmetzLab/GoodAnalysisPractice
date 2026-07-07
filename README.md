# Good Analysis Practice

A small collection of hands-on exercises about **data-analysis pitfalls** in
systems neuroscience, built for a summer course (grad students / postdocs).

Each exercise is a Jupyter notebook that loads data, runs a perfectly ordinary
analysis, and produces a clean figure with a confident conclusion attached — **and
the conclusion is false.** The apparent result is an artifact of the data not
meeting the analysis's assumptions, of the analysis itself, or both. The
student's job is to figure out *why* before looking at the answer.

The recurring lesson: a compelling figure is not evidence. Structure that a
method *imposes* (clusters, tuning, sequences) has to be checked against a null
or held-out data before it is believed.

## The exercises

| Folder | Apparent result | The pitfall |
|--------|-----------------|-------------|
| [`nb1/`](nb1) | Three functional cell classes ("early/middle/late") | k-means imposes discrete clusters on a smooth continuum |
| [`nb2/`](nb2) | Orientation tuning reduced by a manipulation | Circular analysis / double dipping — selecting the preferred stimulus on the same data (fake difference for real tuning; fake tuning from noise) |
| [`nb3/`](nb3) | A reliable temporal sequence of firing | Sorting neurons by peak time manufactures a diagonal — you must cross-validate sorted plots |
| [`nb4/`](nb4) | A population that decodes a task "block" variable | Ordinary cross-validation is fooled by slow drift when decoding a slowly-varying variable — use held-out blocks / pseudosessions |

## How each exercise is organized

- `generate_data.py` — creates the synthetic dataset(s) in `data/`.
- `notebookN_student.ipynb` — what students receive: the analysis and the false
  claim. **Start here.**
- `notebookN_solution.ipynb` — the reveal: several independent diagnostics and
  the lesson.
- `hints.md` — guiding questions if you get stuck (try without them first).
- `README.md` — a short description of the pitfall.

The notebooks are committed with their outputs, so you can read the figures on
GitHub without running anything.

## Setup

```bash
conda create -n goodanalysis python=3.11 numpy scipy scikit-learn matplotlib jupyter
conda activate goodanalysis
jupyter lab            # open notebookN_student.ipynb in each folder
```

(To regenerate data or rebuild the notebooks from their `build_notebooks.py`,
also `pip install nbformat nbclient`.)

## Contributing / notes

These are teaching materials; the data are synthetic and generated with fixed
seeds so results are reproducible. Suggestions and new pitfalls are welcome.

## Development

These training materials were developed by Nick Steinmetz using Claude Code
(Opus 4.8).
