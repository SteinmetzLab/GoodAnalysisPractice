# Notebook 3 — hints

Try to answer these on your own before opening `notebook3_solution.ipynb`.

- The sort order was computed from the data being displayed. If you sorted a
  matrix of pure random numbers by each row's peak column, what would the sorted
  image look like?

- A "sequence" is a claim that each neuron fires at a *characteristic* time. How
  could you check whether a neuron's peak time is reproducible — i.e. the same on
  data that was not used to sort?

- Split the trials in two. Sort the neurons by their peak time on one half, then
  display the *other* half in that order. What do you predict for a real
  sequence? For sorted noise?

- Plot each neuron's peak time estimated from one half of trials against its peak
  time from the other half. Where would the points fall if the sequence were
  real? Where do they actually fall?

- Does the colormap change how "real" the diagonal looks? Why might a rainbow
  map (jet) exaggerate apparent structure?
