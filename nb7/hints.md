# Notebook 7 — hints

- The test counts each of the 600 neurons as an independent observation. Are
  neurons that were recorded *at the same time* independent of one another?

- What might all the neurons in a session have in common that varies from trial
  to trial (brain state, arousal, movement)?

- Plot each neuron's A-minus-B difference grouped by session. How many distinct
  values are there really -- 600, or closer to the number of sessions?

- Compute the effect at the level of the session: average A-B within each
  session, then test across sessions. What happens to the p-value?

- Was anything *else* different between A and B trials? If you have a measure of
  arousal, what happens to the condition effect after you regress it out?
