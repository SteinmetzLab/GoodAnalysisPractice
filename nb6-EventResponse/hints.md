# Notebook 6 — hints

- The figure's baseline is perfectly flat at 1. Ask what the denominator was. If
  you divide the baseline bins by *their own mean*, what must their average come
  out to, for every unit, before you look at any spikes?

- Try changing one operation and nothing else: subtract each unit's baseline
  (in spikes/s) instead of dividing by it. Does the step survive?

- Plot the population firing rate with no normalization at all. Is there anything
  at t = 0?

- The baseline is estimated from 300 ms x 20 trials per unit. For a unit firing at
  1 spike/s, how many spikes is that? What does dividing by that number do to a
  ratio when the count happens to come out low?

- Widen the plotting window to -0.8 s while keeping the baseline defined as
  [-0.3, 0). The bins before -0.3 s are not in the denominator any more. Where do
  they sit?

- Estimate the baseline from odd trials and build the PSTH from even trials. Does
  the step survive? Does the *elevation* survive? Those are two different
  questions, and the answers are not the same.

- Look at the distribution of per-unit fold-changes rather than its mean, and plot
  each unit's fold-change against its baseline rate. Which units are pulling the
  mean up?

- Compare summaries that do not average a ratio: the mean difference in spikes/s,
  the ratio of the population means, the geometric mean. Do they agree with the
  8.8%? For the geometric mean, does it matter that the post window is 0.7 s and
  the baseline window is 0.3 s?

- Re-estimate the baseline from progressively more data (there is plenty of
  held-out inter-trial time in the recording). If the elevation were a response,
  should it care?
