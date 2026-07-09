# Notebook 6 — hints

- The fold-change divides each neuron's response by its own baseline. That
  baseline is estimated from only a few trials. What happens to the ratio when a
  neuron's baseline is, by chance, estimated to be very small?

- Look at the *distribution* of fold-changes, not just its mean. Compare the mean
  to the median. Is it symmetric?

- Plot each neuron's fold-change against its baseline rate. Where do the large
  fold-changes come from?

- Try summaries that don't average per-neuron ratios: the mean *difference*
  (post - pre), the ratio of the population means, the mean of the *log*
  fold-change. Do they agree with the ~1.3x?

- Re-estimate the baseline from more trials. Does the mean fold-change change?
