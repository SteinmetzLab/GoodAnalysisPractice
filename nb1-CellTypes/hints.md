# Notebook 1 — hints

Try to answer these on your own before opening `notebook1_solution.ipynb`.

- What assumption does k-means make about the structure of the data? Given
  a value of `k`, will it ever *decline* to return that many groups?

- Is "three classes" a property of the **data**, or a property of the
  **analysis**? Change `k = 3` to `k = 4` or `k = 5` and re-run — what do you
  get, and what does that tell you?

- The mean ± SEM traces summarize each cluster with a single curve. What does
  the distribution of *individual* neurons within a cluster look like? When `n`
  is large, what is the SEM actually measuring — and is a small SEM evidence
  that a group is homogeneous?

- Can you find a visualization that does **not** presuppose the number of
  groups? A few ideas: estimate each neuron's response peak time and plot the
  distribution; sort *all* neurons by peak time in one heatmap; project the
  neurons onto their first two principal components; compute a cluster-validity
  score (e.g. silhouette) across a range of `k`.

- If the data really contained three discrete types, which of the checks above
  would show it — and do they?
