# Notebook 4 — hints

Try to answer these on your own before opening `notebook4_solution.ipynb`.

- The block variable is *slow*: it stays constant for 30-70 trials at a time.
  Ordinary k-fold CV scatters trials randomly into folds. For a held-out trial
  in the middle of a block, what is true of its temporal neighbours (which are
  in the training set)?

- Is the neural activity stationary across the session? Plot a few neurons'
  activity as a function of trial number and look for slow changes (drift).

- If drift and the block are both slow, a decoder could "predict" a held-out
  trial's block from its drift state (shared with nearby training trials)
  without encoding the block at all. How would you break that shortcut?

- Try holding out a *whole block* at a time (all its trials together) instead of
  scattered trials. What happens to accuracy? Why?

- Build a null that respects the slow structure: generate surrogate block
  sequences with the same statistics but unrelated to the neurons (you already
  have `make_blocks`), decode each, and see whether the real accuracy stands out.

- Would that null still work if every block were exactly the same length?
