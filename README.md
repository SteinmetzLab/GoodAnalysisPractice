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
| [`nb1-CellTypes/`](nb1-CellTypes) | Three functional cell classes ("early/middle/late") | k-means imposes discrete clusters on a smooth continuum |
| [`nb2-TuningChange/`](nb2-TuningChange) | Orientation tuning reduced by a manipulation | Circular analysis / double dipping — selecting the preferred stimulus on the same data (fake difference for real tuning; fake tuning from noise) |
| [`nb3-Sequences/`](nb3-Sequences) | A reliable temporal sequence of firing | Sorting neurons by peak time manufactures a diagonal — you must cross-validate sorted plots |
| [`nb4-BlockDecoding/`](nb4-BlockDecoding) | A population that decodes a task "block" variable | Ordinary cross-validation is fooled by slow drift when decoding a slowly-varying variable — use held-out blocks / pseudosessions |
| [`nb5-ArousalAndRate/`](nb5-ArousalAndRate) | Arousal increases firing rate | Simpson's paradox — pooling across sessions reverses the within-session relationship |
| [`nb6-EventResponse/`](nb6-EventResponse) | A population PSTH response to an event: +9%, onset in the first post-event bin, p≈9e-19 | Averaging per-unit ratios is biased upward (Jensen), *and* a baseline-normalized PSTH pins its own baseline to exactly 1, so the bias arrives disguised as an onset |
| [`nb7-ConditionAvsB/`](nb7-ConditionAvsB) | 400 neurons fire more on condition A (p≈2e-12) | Pseudo-replication: 4 session-level coin flips copied 100 times each. The unit of analysis is the session, not the neuron, and more sessions never fix it |
| [`nb8-RegionRates/`](nb8-RegionRates) | Region A fires at half the rate of region B (10 vs 140 units, p=0.009) | A t-test on strongly skewed data with very unequal group sizes. A small sample misses the heavy tail, so its mean *and* its SD come out low together, and Welch's t-test reads the small SD as precision |
| [`nb9-PairedChange/`](nb9-PairedChange) | A manipulation has no effect on firing (paired t-test, p=0.63) | The opposite failure: a real 20% suppression in 32 of 40 units, missed. The effect is multiplicative and rates are lognormal, so the mean difference in spikes/s is a rate-weighted average with an effective n of 10 |

## Walkthroughs

Not every topic works as a mystery. These are guided demos meant to be run
**together with the class**, section by section.

| Folder | Topic |
|--------|-------|
| [`nonsenseCorr_walkthrough/`](nonsenseCorr_walkthrough) | Nonsense correlations from slow fluctuations: 79/100 neurons "significantly" correlated with pupil and 83% block decoding, from neurons generated independently of both — which controls catch it (circular shift, session permutation, pseudosessions, block hold-out) and which only appear to (trial shuffling); the effective sample size that explains it; a positive control showing the good tests still have power; and a sweep over the fluctuation timescale |

## Interactive demos

Things you run and click rather than read.

| Folder | What it does |
|--------|--------------|
| [`plotting_drilldown_demo/`](plotting_drilldown_demo) | Starts from the figure everyone publishes — three average PSTHs, mean ± SEM across neurons — and lets you click down through the neurons × time matrix, one neuron's trials, and one trial, to the raw voltage on the probe with a dot on every spike. Makes the distance between the published trace and the recorded data clickable. Comes in three forms: a standalone script, a Jupyter notebook, and **[a live web page](https://drilldown-psth-to-voltage.netlify.app)** that needs no install. Synthetic data. |
| [`allen_drilldown/`](allen_drilldown) | The same drill-down over **real Allen Institute Visual Coding – Neuropixels data**: 110 V1 units, drifting gratings at 8 directions. The population average is nearly flat, yet 74 of the 110 units are strongly orientation-tuned — they just prefer different orientations, so averaging cancels it. **[Live page](https://allen-v1-orientation-drilldown.netlify.app)**. |

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
conda env create -f environment.yml   # note: `conda env create`, not `conda create`
conda activate goodanalysis
jupyter lab                          # open notebookN_student.ipynb in each folder
```

`environment.yml` already includes everything, including `nbformat` and
`nbclient` (needed only to rebuild notebooks from their `build_notebooks.py`) and
`rastermap` and `ipympl` (needed only for `nonsenseCorr_walkthrough`). If you would rather
install by hand, the core is `python=3.11 numpy scipy scikit-learn matplotlib
jupyter`.

## Contributing / notes

These are teaching materials; the data are synthetic and generated with fixed
seeds so results are reproducible. Suggestions and new pitfalls are welcome.

## Development

These training materials were developed by Nick Steinmetz using Claude Code
(Opus 4.8).
