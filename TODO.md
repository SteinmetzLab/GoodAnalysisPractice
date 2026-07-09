# Ideas for future notebooks

Candidate pitfalls not yet built. Each would follow the same format (student
notebook with a false claim + solution notebook with the reveal). Grouped so we
pick conceptually distinct ones rather than variants of the same idea.

## From the original list

- **High-dimensional angles.** Claim: two coding dimensions are "(significantly)
  orthogonal" (or aligned). Reveal: random high-D vectors are nearly orthogonal
  by default, and angle estimates are biased/noisy with few trials — an observed
  angle means nothing without the null distribution from shuffled data and
  cross-validated axes. *(Confirm this is the intended reading.)*

- **Smoothing shifts apparent timing.** Claim: area/condition A responds earlier
  than B. Reveal: two signals with identical onset but different amplitudes cross
  a fixed threshold at different times after smoothing; symmetric/acausal filters
  leak signal backward (onset looks pre-stimulus). Fix: causal filters,
  amplitude-independent onset measures.

- **Significant vs. non-significant.** Claim: A is modulated (p<0.01), B is not
  (p=0.2), so A > B. Reveal: the A-vs-B difference is itself not significant
  (Gelman & Stern). Fix: test the interaction/difference directly.

- **Outlier / one-point-driven effects.** Claim: X correlates with Y (or a clear
  PSTH response). Reveal: one influential point / one artifact trial carries the
  whole effect; leave-one-out or plotting the raw data collapses it. Fix: robust
  stats, influence diagnostics, look at your data.

## From brainstorming

- **Multiple comparisons / garden of forking paths.** Testing many
  neurons/timepoints/regions (and flexible pipeline choices) yields "significant"
  results by chance. Fix: permutation/cluster or FDR correction; report the whole
  distribution.

- **Base rates & the wrong decoding metric.** "80% decoding accuracy" when 80% of
  trials are one class. Fix: proper chance level, balanced accuracy / d',
  permutation null.

- **Regression to the mean.** Select the most extreme units on a noisy
  measurement, remeasure, and they "recover" toward the mean with no real change.
  Fix: control group, model the selection.

- **Collider / selection-into-the-sample (Berkson).** Two independent traits look
  anticorrelated because units are included only when a *combination* of them
  crosses a detection/inclusion threshold. Fix: understand selection into the
  sample.

## Considered and dropped

- **Signal vs. noise correlations.** A signal correlation is a legitimate measure
  and can exist without a noise correlation (and vice versa); reporting one is not
  an error. Not a clean pitfall.
