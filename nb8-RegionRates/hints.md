# Notebook 8 — hints

- Plot the distribution of firing rates, not just the group means. Try a linear
  axis and then a log axis. Which one makes the distribution look normal?

- Compare the mean and the median in each region. Do they even agree on which
  region is higher?

- Look at the two group SDs. Region A's is much smaller than region B's. Is
  region A genuinely less variable, or did something about having only 10 units
  make it *look* less variable?

- Welch's t-test divides by each group's own standard deviation. What happens to
  the standard error if a group's SD comes out too small by chance? And for a
  right-skewed distribution, is a sample's SD too small at the same times its
  mean is too small, or at different times?

- Draw 10 values many times from a strongly right-skewed distribution. How often
  is the sample mean below the true mean? How often does the sample contain one
  of the very large values?

- Run the whole comparison many times on data where both groups come from the
  same distribution, and count how often each test gives p < 0.05. Try Welch's
  t-test, Student's pooled t-test, Mann-Whitney, and a t-test on log rates. Which
  ones land on 5%?

- Repeat that check with equal group sizes, say 30 and 30. Does the skew still
  cause a problem? What does that tell you about which ingredient is doing the
  damage?

- Everyone is taught to prefer Welch's t-test over Student's. Here Student's is
  the safer of the two. What does pooling the variance do that protects it, and
  why would you still not want to rely on that?

- If the false positives all have the same sign, what does that imply for a
  literature comparing rare, hard-to-sample cell populations against abundant
  ones?
