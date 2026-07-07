# Notebook 2 — hints

Try to answer these on your own before opening `notebook2_solution.ipynb`.

- The analysis does two things with the data: it *chooses* each neuron's
  preferred direction, and it *reports* the firing rate at that direction. Are
  those two steps using independent data? What happens if they aren't?

- Take the maximum of 12 noisy numbers whose true means are equal (or nearly
  equal). On average, is that maximum equal to the true mean, or larger? Now
  measure a *second, independent* condition at that same location — is it
  inflated too?

- Why is the test (gray) curve lower than control *only near the peak*, and why
  do they match in the flanks? Which condition was used to pick the preferred
  direction?

- Predict: if you defined each neuron's preferred direction from the *test*
  condition instead, what happens to the two curves?

- Separate the two questions. (a) Are these neurons really tuned? (b) Is the
  reduction real? What analysis could answer each — and would each survive if
  you selected the preferred direction on one set of trials and measured on
  another?

- Push it: what would this pipeline do to neurons with *no* tuning at all
  (shuffle the direction labels, or simulate a flat Poisson rate)?
