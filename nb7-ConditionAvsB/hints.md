# Notebook 7 — hints

- The test counts each of the 400 neurons as an independent observation. Were
  neurons recorded *at the same time*, in the same brain, independent of one
  another?

- Plot each neuron's A-minus-B difference grouped by the session it came from.
  How many distinct numbers are there really: 400, or closer to the number of
  sessions?

- Average A-B within each session first, then test across sessions. What happens
  to the p-value, and what is the honest sample size?

- The A/B labels were shuffled independently in each session. In a given session,
  did the A trials happen to fall earlier or later than the B trials? Plot arousal
  against trial number and mark which trials were A.

- What do all the neurons in one session have in common that varies from trial to
  trial? What happens to the condition effect if you regress arousal out within
  each session?

- Shuffle the A/B labels within each session, recompute the *naive* t-statistic,
  and repeat a few thousand times. If the naive t were a valid t-statistic, what
  SD should that null distribution have? What SD does it actually have?

- Suppose you ran this experiment again from scratch. Would the effect come out in
  the same direction? Try simulating the whole thing many times and look at both
  the p-value distribution and the sign.

- If the naive test is fooled here, would collecting more sessions rescue it? Work
  out how the numerator and the denominator of the t-statistic each scale with the
  number of sessions before you guess.
