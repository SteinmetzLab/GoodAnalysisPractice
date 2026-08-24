# Notebook 9 — hints

- The claim is a *negative* result. Before believing "no effect," ask what effect
  size this test could have detected if one were there.

- Count. How many of the 40 units went down and how many went up? Does that
  number look like a coin flip?

- Look at the paired plot again. How many of the 40 lines can you actually see?
  What happened to the rest of them, and why?

- Try a log axis. Plot each unit's rate with the laser on against its rate with
  the laser off, and draw the unity line.

- For each unit, compute the *ratio* on/off rather than the difference. What is
  the median ratio? Now compare that to the mean difference in spikes/s that the
  t-test was given.

- If the manipulation scales every unit's rate by the same factor, how big is the
  change in spikes/s for a unit firing at 80 spikes/s compared with one firing at
  1 spikes/s? Which of those two units dominates the mean difference?

- The mean paired difference is effectively a weighted average, with each unit
  weighted by its firing rate. For weights `r`, the effective sample size is
  `(sum r)^2 / sum(r^2)`. Compute it. How many of your 40 units are really
  contributing?

- Try the Wilcoxon signed-rank test, the sign test, and a paired t-test on
  `log(rate)`. Do they agree with each other? Do they agree with the paired
  t-test on raw rates?

- Simulate: generate data where nothing changes at all and check that the paired
  t-test rejects 5% of the time. Then generate data with a real multiplicative
  effect and see how often each test finds it. Is this a calibration problem or a
  power problem?

- Finally, the trap in the other direction. The paired t-test, Wilcoxon and the
  sign test have three different null hypotheses. Write down what each one
  actually assumes about the distribution of the differences. Construct
  differences that are skewed but have a mean of exactly zero, and see which
  tests reject. Is "use the non-parametric test" a safe general rule?
