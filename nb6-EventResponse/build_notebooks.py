"""Build student and solution notebooks for Notebook 6 (baseline-ratio bias)."""
import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))

PREAMBLE = """\
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import ttest_1samp

plt.rcParams['font.family']       = 'Arial'
plt.rcParams['font.sans-serif']   = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['pdf.fonttype']      = 42
plt.rcParams['ps.fonttype']       = 42


def band(ax, x, m, s, color, label=None):
    \"\"\"Mean with a shaded +/- SEM band.\"\"\"
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.25, lw=0, zorder=2)
    ax.plot(x, m, '-', color=color, lw=1.8, zorder=3, label=label)
"""

LOAD = """\
d = np.load('data/data.npz')
spike_times    = d['spike_times'].astype(float)    # (n_spikes,) s, ascending
spike_clusters = d['spike_clusters'].astype(int)   # (n_spikes,) unit index per spike
event_times    = d['event_times']                  # (n_events,) s
n_neurons      = int(d['n_neurons'])
duration       = float(d['recording_duration'])    # s
n_events       = len(event_times)
print(f'{n_neurons} units, {n_events} events, {len(spike_times):,} spikes, '
      f'{duration:.1f} s recording')

session_rate = np.bincount(spike_clusters, minlength=n_neurons) / duration
print(f'session-wide firing rate: median {np.median(session_rate):.2f}, '
      f'range {session_rate.min():.2f} to {session_rate.max():.0f} spikes/s')
"""

RASTER = """\
# Six example units spanning the rate range, five consecutive events each.
order    = np.argsort(session_rate)
ex_units = order[[int(f*(n_neurons-1)) for f in (0.10, 0.30, 0.50, 0.70, 0.88, 0.98)]]
ex_events = np.arange(5)
WIN_LOOK = (-0.30, 0.70)

fig, ax = plt.subplots(figsize=(8.5, 5.0))
row = 0
yticks, ylabels = [], []
for u in ex_units:
    st_u = spike_times[spike_clusters == u]
    rows_here = []
    for j in ex_events:
        rel = st_u - event_times[j]
        rel = rel[(rel >= WIN_LOOK[0]) & (rel < WIN_LOOK[1])]
        ax.plot(rel, np.full(rel.size, row), '|', color='k', ms=6, mew=1.0)
        rows_here.append(row)
        row += 1
    yticks.append(np.mean(rows_here))
    ylabels.append(f'Unit {u}\\n{session_rate[u]:.1f} spikes/s')
    ax.axhline(row - 0.5, color='0.85', lw=0.8)
    row += 1

ax.axvspan(WIN_LOOK[0], 0.0, color='0.92', zorder=0)
ax.axvline(0.0, color='tab:red', ls='--', lw=1.2)
ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
ax.set_ylim(row - 0.5, -1.0)
ax.set_xlim(*WIN_LOOK)
ax.set_xlabel('Time from event (s)')
ax.set_ylabel('Example unit (five events each)')
ax.set_title('Raw data: spikes around the event\\n'
             'Red line is the event, shaded region is the pre-event baseline')
plt.tight_layout(); plt.show()
"""

PSTH_FUNC = """\
WIN = (-0.30, 0.70)      # s relative to the event
BIN = 0.05               # s, PSTH bin width

def psth_counts(win, bin_w, trials=None):
    \"\"\"Spike counts in bins relative to each event -> (units, trials, bins).\"\"\"
    edges  = np.arange(win[0], win[1] + bin_w / 2, bin_w)
    events = event_times if trials is None else event_times[trials]
    counts = np.zeros((n_neurons, len(events), len(edges) - 1))
    for j, t_ev in enumerate(events):
        rel = spike_times - t_ev                       # spike times re: this event
        sel = (rel >= win[0]) & (rel < win[1])
        b   = np.floor((rel[sel] - win[0]) / bin_w).astype(int)
        # np.add.at, NOT `counts[...] += 1`: fancy-index += drops repeated
        # indices, so a unit with two spikes in one bin on one trial would
        # be counted once. That silently undercounts high-rate units.
        np.add.at(counts[:, j, :], (spike_clusters[sel], b), 1)
    return counts, edges

counts, edges = psth_counts(WIN, BIN)
t_bin       = edges[:-1] + BIN / 2       # bin centers, s
rate        = counts.mean(1) / BIN       # (units, bins) firing rate, spikes/s
is_baseline = t_bin < 0                  # pre-event bins define the baseline
print(f'{rate.shape[1]} bins of {BIN*1000:.0f} ms: '
      f'{is_baseline.sum()} baseline, {(~is_baseline).sum()} post-event')
"""

MATRIX = """\
srt = np.argsort(rate.mean(1))           # sort units by their mean rate

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6),
                         gridspec_kw={'width_ratios': [1.25, 1]})

im = axes[0].imshow(rate[srt], aspect='auto', cmap='magma',
                    norm=LogNorm(vmin=0.5, vmax=100),
                    extent=[WIN[0], WIN[1], n_neurons, 0], interpolation='nearest')
axes[0].axvline(0.0, color='w', ls='--', lw=1.0)
axes[0].set_xlabel('Time from event (s)')
axes[0].set_ylabel('Unit (sorted by mean rate)')
axes[0].set_title('Units x time firing rate matrix')
cb = fig.colorbar(im, ax=axes[0]); cb.set_label('Firing rate (spikes/s)')

axes[1].hist(np.log10(rate.mean(1)), bins=50, color='0.6')
axes[1].set_xlabel('log10 mean firing rate (spikes/s)')
axes[1].set_ylabel('Number of units')
axes[1].set_title('Rates span more than two orders of magnitude,\\n'
                  'so a raw population average is dominated by a few units')
plt.tight_layout(); plt.show()
"""

# ---------------- STUDENT ----------------
student = []
student.append(new_markdown_cell("""\
# Notebook 6 — Did the event increase firing?

**Summer course: data-analysis pitfalls**

One session, 2000 sorted units, 20 repeats of an event. We have spike times, unit
labels, and the event times. We want the population response to the event.

Units differ enormously in firing rate, from under 1 to over 100 spikes/s, so a
raw population average would be dominated by a handful of fast units. To account
for this, a colleague built each unit's PSTH, divided it by that unit's own
pre-event baseline, and averaged the normalized PSTHs across units.

> ### The claim
> *"The event drives a sustained ~9% increase in firing across the population
> (n = 2000 units, p = 9e-19). The baseline is flat and the increase begins in
> the first bin after the event."*

Run it and you'll see it: a clean flat baseline, then a step.

**The claim is false. Nothing in this dataset responds to the event.** Your job is
to work out how the figure came to look like that. `hints.md` has guiding
questions and `notebook6_solution.ipynb` has the answer.
"""))
student.append(new_code_cell(PREAMBLE))
student.append(new_code_cell(LOAD))
student.append(new_markdown_cell("""\
## Step 1 — Look at the raw data

Before any averaging, a raster of a few example units around the first five
events. Rows are trials, grouped by unit.
"""))
student.append(new_code_cell(RASTER))
student.append(new_markdown_cell("""\
## Step 2 — Build the PSTH

Count spikes in 50 ms bins from 300 ms before each event to 700 ms after, then
convert to a firing rate. The pre-event bins are the baseline.
"""))
student.append(new_code_cell(PSTH_FUNC))
student.append(new_markdown_cell("""\
That gives a units x time matrix. Plotting it shows why some kind of
normalization is tempting: the rate distribution is enormously wide, so a plain
average across units is really a report on the fastest few percent.
"""))
student.append(new_code_cell(MATRIX))
student.append(new_markdown_cell("""\
## Step 3 — Normalize each unit to its own baseline, then average

Each unit's PSTH is divided by that unit's mean baseline rate, so every unit
contributes on a common scale where 1 means "at baseline". Then average across
units and plot mean with a shaded SEM band.
"""))
student.append(new_code_cell("""\
baseline = rate[:, is_baseline].mean(1)          # (units,) baseline rate, spikes/s
norm     = rate / baseline[:, None]              # each unit relative to itself

mean_norm = norm.mean(0)
sem_norm  = norm.std(0, ddof=1) / np.sqrt(n_neurons)

# One number per unit: its mean normalized rate after the event
post_per_unit = norm[:, ~is_baseline].mean(1)
t_stat, p_val = ttest_1samp(post_per_unit, 1.0)
pct           = (post_per_unit.mean() - 1) * 100

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4),
                         gridspec_kw={'width_ratios': [1.3, 1]})

ax = axes[0]
ax.axvspan(WIN[0], 0.0, color='0.92', zorder=0)
ax.axhline(1.0, color='0.55', ls=':', lw=1.0, zorder=1)
ax.axvline(0.0, color='k', ls='--', lw=1.0, zorder=1)
band(ax, t_bin, mean_norm, sem_norm, 'tab:red')
ax.text(WIN[0] + 0.01, ax.get_ylim()[1], 'Baseline\\nwindow', va='top',
        ha='left', fontsize=8, color='0.35')
ax.set_xlabel('Time from event (s)')
ax.set_ylabel('Firing rate (normalized to baseline)')
ax.set_title(f'Event-evoked increase: {pct:.1f}% over baseline\\n'
             f'Mean +/- SEM across {n_neurons} units, p = {p_val:.1e}')

ax = axes[1]
ax.hist(post_per_unit, bins=60, color='0.6')
ax.axvline(1.0, color='k', ls='--', lw=1.0, label='No change')
ax.axvline(post_per_unit.mean(), color='tab:red', lw=2,
           label=f'Mean = {post_per_unit.mean():.3f}')
ax.set_xlabel('Post-event rate (normalized to baseline)')
ax.set_ylabel('Number of units')
ax.set_title('Not driven by a few outliers:\\nthe whole distribution is shifted')
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

print(f'mean normalized rate after the event : {post_per_unit.mean():.4f}'
      f'  ({pct:+.1f}%)')
print(f'mean normalized rate in baseline     : {norm[:, is_baseline].mean():.4f}')
print(f'units above 1 after the event        : {np.mean(post_per_unit > 1)*100:.1f}%')
print(f't-test vs 1 across units             : t = {t_stat:.2f}, p = {p_val:.2e}')
"""))
student.append(new_markdown_cell("""\
A sustained ~9% increase, onset in the very first bin after the event, an
immaculate flat baseline, and p = 9e-19 across 2000 units. And it is not an
outlier effect: the whole per-unit distribution is shifted, with a clear majority
of units above 1.

Every single unit in this dataset fires at a constant rate for the whole
recording. **How did this figure happen?**
"""))

# ---------------- SOLUTION ----------------
solution = []
solution.append(new_markdown_cell("""\
# Notebook 6 — SOLUTION

## Short version

**Nothing changed.** Every unit is a homogeneous Poisson process with a constant
rate. The figure is the product of two things happening at once, and it is worth
separating them because only one of them is the famous bias:

1. **Averaging ratios is biased upward when the denominator is noisy.** Each
   post-event bin gets divided by a baseline estimated from a handful of spikes.
   Units that happen to have a low baseline estimate get a large ratio, and the
   right tail drags the mean above 1. The inflation is roughly
   `1 + 1/(number of baseline spikes)`. This is Jensen's inequality:
   E[X/Y] > E[X]/E[Y] when Y is variable.

2. **The baseline bins cannot show that bias, because they are pinned to exactly
   1 by construction.** The denominator is the mean of those same bins, so their
   mean ratio is 1 for every unit, exactly, with zero noise. So the flat,
   trustworthy-looking baseline is not evidence that the method is unbiased. It
   is an algebraic identity.

Together they manufacture the signature of a real response: a perfect baseline
and a sharp step at t = 0. The step is the boundary between the bins that are
pinned and the bins that are free to be inflated, not a physiological onset.

Note that the shifted histogram in the student notebook is not a defense. A
population-wide bias shifts the whole distribution, exactly as a real effect
would. Ruling out outliers does not rule out bias.
"""))
solution.append(new_code_cell(PREAMBLE))
solution.append(new_code_cell(LOAD))
solution.append(new_code_cell(PSTH_FUNC))

solution.append(new_markdown_cell("""\
## Evidence 1 — Subtract instead of divide and the response disappears

The cleanest test. Keep everything the same, including correcting each unit by
its own baseline, and change one operation: subtract the baseline (spikes/s)
instead of dividing by it.

Both versions have an immaculate baseline for the same algebraic reason. Only the
ratio has a step.

The middle panel is the raw population rate with no normalization at all. Its
error band is wide because units differ hugely in rate, which is exactly why
people normalize in the first place. The right panel is the sharp version of the
same test.
"""))
solution.append(new_code_cell("""\
baseline = rate[:, is_baseline].mean(1)
norm = rate / baseline[:, None]          # divide
diff = rate - baseline[:, None]          # subtract (spikes/s)

def panel(ax, X, ref, color):
    m = X.mean(0); s = X.std(0, ddof=1) / np.sqrt(X.shape[0])
    ax.axvspan(WIN[0], 0.0, color='0.92', zorder=0)
    ax.axhline(ref, color='0.55', ls=':', lw=1.0, zorder=1)
    ax.axvline(0.0, color='k', ls='--', lw=1.0, zorder=1)
    band(ax, t_bin, m, s, color)
    ax.set_xlabel('Time from event (s)')
    return m, s

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
panel(axes[0], norm, 1.0, 'tab:red')
axes[0].set_ylabel('Firing rate (normalized to baseline)')
axes[0].set_title('Divide by baseline\\n"9% increase"')

panel(axes[1], rate, rate.mean(), 'tab:blue')
axes[1].set_ylabel('Firing rate (spikes/s)')
axes[1].set_title('No normalization\\nFlat')

panel(axes[2], diff, 0.0, 'tab:green')
axes[2].set_ylabel('Firing rate change (spikes/s)')
axes[2].set_title('Subtract baseline\\nFlat')
plt.tight_layout(); plt.show()

pre_r, post_r = rate[:, is_baseline].mean(1), rate[:, ~is_baseline].mean(1)
t_d, p_d = ttest_1samp(post_r - pre_r, 0.0)
print(f'raw population rate      : {pre_r.mean():.4f} -> {post_r.mean():.4f} spikes/s '
      f'({(post_r.mean()/pre_r.mean() - 1)*100:+.2f}%)')
print(f'baseline-subtracted step : {(post_r - pre_r).mean():+.4f} spikes/s, '
      f't = {t_d:.2f}, p = {p_d:.2f}   <- no change')
print(f'ratio step               : {norm[:, ~is_baseline].mean() - 1:+.4f} '
      f'(= {(norm[:, ~is_baseline].mean() - 1)*100:+.1f}%)         <- the artifact')
"""))

solution.append(new_markdown_cell("""\
## Evidence 2 — The flat baseline is an identity, not a result

If the denominator is the mean of the baseline bins, then for **every** unit the
mean of its normalized baseline bins is exactly 1, whatever the spikes did. Sum
the ratios over the K baseline bins:

$$\\sum_{k=1}^{K} \\frac{r_k}{\\frac{1}{K}\\sum_j r_j} = \\frac{\\sum_k r_k}{\\frac{1}{K}\\sum_j r_j} = K$$

so the mean over baseline bins is 1, identically. There is no noise in that
number and no information in it. The same holds for the subtracted version, where
it is exactly 0.

This matters because the flat baseline is what makes the figure persuasive. It is
also the reason the step looks so sharp: t = 0 is the boundary between the bins
that are pinned and the bins that are free to be inflated.
"""))
solution.append(new_code_cell("""\
pinned_ratio = norm[:, is_baseline].mean(1)      # one per unit
pinned_diff  = diff[:, is_baseline].mean(1)
print('Per-unit mean over the baseline bins, across all '
      f'{n_neurons} units:')
print(f'  divide   : min = {pinned_ratio.min():.12f}, max = {pinned_ratio.max():.12f}')
print(f'  subtract : min = {pinned_diff.min():+.2e}, max = {pinned_diff.max():+.2e}')
print(f'\\n  spread across units (SD): {pinned_ratio.std():.2e}  <- exactly zero')
"""))

solution.append(new_markdown_cell("""\
## Evidence 3 — Two ways to unpin the baseline, and the step vanishes

If the step were a response, it would survive breaking the algebraic link between
numerator and denominator. It does not.

**Left: plot more pre-event time than the baseline window covers.** Keep the
baseline defined as [-0.3, 0) but plot back to -0.8 s. The bins before -0.3 s are
not in the denominator, so they are free to be inflated too, and they are, by the
same amount as the post-event bins. The "response" is present *before* the event.
Only the shaded window is flat.

**Right: estimate the baseline from held-out trials.** Odd trials give the
denominator, even trials give the PSTH. Now no bin is pinned, and the entire
trace, pre-event bins included, sits uniformly above 1. There is no step at all,
just a constant multiplicative bias.

Note that holding out trials does not remove the inflation. It only stops the
inflation from being hidden in the baseline. The bias comes from noise in the
denominator, which held-out data has just as much of.
"""))
solution.append(new_code_cell("""\
# --- Left: wider plotting window, same baseline definition ---
WIDE = (-0.80, 0.70)
counts_w, edges_w = psth_counts(WIDE, BIN)
rate_w  = counts_w.mean(1) / BIN
t_w     = edges_w[:-1] + BIN / 2
in_base = (t_w >= WIN[0]) & (t_w < 0.0)          # the original baseline window
norm_w  = rate_w / rate_w[:, in_base].mean(1)[:, None]

# --- Right: baseline from held-out (odd) trials, PSTH from even trials ---
odd, even = np.arange(1, n_events, 2), np.arange(0, n_events, 2)
c_base, _ = psth_counts((WIN[0], 0.0), BIN, trials=odd)
base_ho   = c_base.mean(1).mean(1) / BIN                 # (units,) spikes/s
c_psth, _ = psth_counts(WIN, BIN, trials=even)
rate_ho   = c_psth.mean(1) / BIN
ok        = base_ho > 0                                  # a few units get 0 spikes
norm_ho   = rate_ho[ok] / base_ho[ok, None]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

m = norm_w.mean(0); s = norm_w.std(0, ddof=1) / np.sqrt(n_neurons)
axes[0].axvspan(WIN[0], 0.0, color='0.92', zorder=0)
axes[0].axhline(1.0, color='0.55', ls=':', lw=1.0)
axes[0].axvline(0.0, color='k', ls='--', lw=1.0)
band(axes[0], t_w, m, s, 'tab:red')
axes[0].set_title('Same baseline window, wider plot\\n'
                  'The "response" precedes the event')
axes[0].text(WIN[0] + 0.01, axes[0].get_ylim()[1], 'Baseline\\nwindow',
             va='top', ha='left', fontsize=8, color='0.35')

m2 = norm_ho.mean(0); s2 = norm_ho.std(0, ddof=1) / np.sqrt(ok.sum())
axes[1].axvspan(WIN[0], 0.0, color='0.92', zorder=0)
axes[1].axhline(1.0, color='0.55', ls=':', lw=1.0)
axes[1].axvline(0.0, color='k', ls='--', lw=1.0)
band(axes[1], t_bin, m2, s2, 'tab:purple')
axes[1].set_title('Baseline from held-out trials\\n'
                  'Uniformly inflated, no step')

for ax in axes:
    ax.set_xlabel('Time from event (s)')
    ax.set_ylabel('Firing rate (normalized to baseline)')
plt.tight_layout(); plt.show()

print('Wider plotting window, one baseline definition:')
print(f'  before the baseline window (t < -0.3 s) : {m[t_w < WIN[0]].mean():.4f}')
print(f'  inside the baseline window             : {m[in_base].mean():.6f}  <- pinned')
print(f'  after the event (t >= 0)               : {m[t_w >= 0].mean():.4f}')
print(f'\\nHeld-out baseline ({ok.sum()} of {n_neurons} units had >= 1 baseline spike):')
print(f'  pre-event bins  : {m2[is_baseline].mean():.4f}')
print(f'  post-event bins : {m2[~is_baseline].mean():.4f}')
print(f'  step            : {(m2[~is_baseline].mean() - m2[is_baseline].mean())*100:+.2f} '
      f'percentage points  <- gone')
"""))

solution.append(new_markdown_cell("""\
## Evidence 4 — Where the inflation lives, and summaries that avoid it

The units driving the mean up are the ones with small, poorly estimated
baselines. The fold-change distribution is right-skewed, so the mean sits above
the median, which is at 1.

The table below compares summaries. A caveat on the log transform: averaging
log fold-changes is often recommended as the fix, and it is much better, but it
is only unbiased when numerator and denominator are estimated from **equal
amounts of data**. Here the post window is 0.7 s and the baseline window is
0.3 s, so the noisier denominator still leaves a small positive bias. Match the
window lengths and it goes away.
"""))
solution.append(new_code_cell("""\
post_r = rate[:, ~is_baseline].mean(1)
fold   = post_r / pre_r
ok_f   = np.isfinite(fold) & (fold > 0)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
axes[0].scatter(pre_r, fold, s=8, color='0.4', alpha=0.5, edgecolors='none')
axes[0].axhline(1.0, color='k', ls='--', lw=1.0)
axes[0].set_xscale('log')
axes[0].set_xlabel('Baseline rate (spikes/s)')
axes[0].set_ylabel('Fold-change (post / baseline)')
axes[0].set_title('The inflation comes from the low-baseline units')

axes[1].hist(fold[ok_f], bins=60, color='0.6')
axes[1].axvline(1.0, color='k', ls='--', lw=1.0, label='No change')
axes[1].axvline(np.median(fold[ok_f]), color='tab:blue', lw=2,
                label=f'Median = {np.median(fold[ok_f]):.3f}')
axes[1].axvline(fold[ok_f].mean(), color='tab:red', lw=2,
                label=f'Mean = {fold[ok_f].mean():.3f}')
axes[1].set_xlabel('Fold-change (post / baseline)')
axes[1].set_ylabel('Number of units')
axes[1].set_title('Right-skewed, so mean > median')
axes[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

# Matched windows: 0.3 s post vs 0.3 s baseline
n_pre = int(is_baseline.sum())
post_matched = rate[:, n_pre:n_pre + n_pre].mean(1)
f_matched    = post_matched / pre_r
ok_m         = np.isfinite(f_matched) & (f_matched > 0)

print(f'{"summary":38s} {"0.7 s post":>12s} {"0.3 s post":>12s}')
print('-' * 64)
rows = [
    ('mean fold-change (biased)',
     fold[ok_f].mean(), f_matched[ok_m].mean()),
    ('median fold-change',
     np.median(fold[ok_f]), np.median(f_matched[ok_m])),
    ('ratio of population means',
     post_r.mean() / pre_r.mean(), post_matched.mean() / pre_r.mean()),
    ('geometric mean (2 ** mean log2)',
     2 ** np.log2(fold[ok_f]).mean(), 2 ** np.log2(f_matched[ok_m]).mean()),
]
for name, a, b in rows:
    print(f'{name:38s} {a:12.4f} {b:12.4f}')
print(f'{"mean difference (spikes/s)":38s} '
      f'{(post_r - pre_r).mean():12.4f} {(post_matched - pre_r).mean():12.4f}')
print('\\nOnly the mean fold-change claims an increase. Everything that avoids')
print('averaging a ratio with a noisy denominator says no change.')
"""))

solution.append(new_markdown_cell("""\
## Evidence 5 — The bias is estimation noise: more baseline data shrinks it

If the inflation were a real response it would not care how much data went into
the denominator. It cares a great deal.

Here the baseline is re-estimated from held-out inter-trial time (recording time
outside every plotted window, so numerator and denominator are independent), and
the amount of it is swept. The dashed line is the prediction
`1 + mean(1 / baseline spike count)`, the second-order approximation to the ratio
bias. It converges onto the data as the baseline grows, and overshoots badly when
the baseline holds only a handful of spikes, which is where a two-term Taylor
expansion has no business being trusted. What matters is that both curves head to
1: the elevation is a property of how much data went into the denominator, not of
the event.

The sweep runs on a fixed set of units, those with at least one baseline spike at
the shortest duration, so the population does not change along the curve.
"""))
solution.append(new_code_cell("""\
# Held-out time: everything not inside a plotted window
gaps = []
for i, t_ev in enumerate(event_times):
    start = t_ev + WIN[1]
    stop  = event_times[i + 1] + WIN[0] if i + 1 < n_events else duration
    if stop > start:
        gaps.append((start, stop))
print(f'{len(gaps)} inter-trial gaps, {sum(b - a for a, b in gaps):.1f} s of held-out time')

def baseline_from_heldout(total_s):
    \"\"\"Spike counts in the first `total_s` s of held-out time -> (counts, seconds).\"\"\"
    cnt, used, need = np.zeros(n_neurons), 0.0, total_s
    for a, b in gaps:
        if need <= 1e-9:
            break
        take = min(b - a, need)
        sel  = (spike_times >= a) & (spike_times < a + take)
        np.add.at(cnt, spike_clusters[sel], 1)
        used += take
        need -= take
    return cnt, used

durations = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 30.0])
swept     = [baseline_from_heldout(D) for D in durations]
keep      = np.all([c > 0 for c, _ in swept], axis=0)     # fixed population

obs, theory, nspk = [], [], []
for cnt, used in swept:
    b = cnt / used
    obs.append((rate[keep] / b[keep, None]).mean())
    theory.append(1 + np.mean(1.0 / cnt[keep]))
    nspk.append(cnt[keep].mean())

fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.axhline(1.0, color='0.55', ls=':', lw=1.0, label='No change')
ax.plot(nspk, theory, 's--', ms=5, color='0.55', mfc='white',
        label='Predicted: 1 + mean(1 / baseline spikes)')
ax.plot(nspk, obs, 'o-', ms=6, color='k', label='Observed')
ax.set_xscale('log')
ax.set_xlabel('Spikes in the baseline estimate (per unit)')
ax.set_ylabel('Mean normalized rate after the event')
ax.set_title(f'The apparent response is denominator noise\\n'
             f'{keep.sum()} units, baseline held out from the PSTH')
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

print(f'{"held-out s":>11s} {"baseline spikes":>16s} {"observed":>10s} {"predicted":>10s}')
for D, ns, o, th in zip(durations, nspk, obs, theory):
    print(f'{D:11.0f} {ns:16.1f} {o:10.4f} {th:10.4f}')
"""))

solution.append(new_markdown_cell("""\
## The lesson

- **Averaging per-unit ratios is biased upward when the denominator is noisy.**
  A baseline-normalized PSTH divides by a baseline estimated from a few spikes.
  Small baselines blow up, the distribution is right-skewed, and the mean exceeds
  1 with no real change. The inflation is about `1 + 1/(baseline spike count)`,
  so it is worst for exactly the low-rate units that normalization was introduced
  to rescue.

- **A flat baseline in a baseline-normalized plot is not a control.** When the
  denominator is the mean of the baseline bins, those bins average to exactly 1
  by algebra, for every unit, no matter what. The immaculate baseline is the most
  persuasive part of the figure and it carries no information. Worse, it hides the
  bias in precisely the region a reader would use to judge it, which is what
  turns a uniform offset into an apparent onset at t = 0.

- **A shifted distribution does not distinguish a bias from an effect.** Checking
  that the result is not outlier-driven is good practice, and it rules out one
  alternative, but a population-wide bias moves the whole distribution just as a
  real effect would.

- **Diagnostics that work.** Subtract the baseline instead of dividing. Plot more
  pre-event time than the baseline window covers, so some pre-event bins are
  unpinned. Estimate the baseline from held-out trials. Report the mean
  difference in spikes/s, the ratio of population means, or a geometric mean over
  matched-length windows. If you must show fold-changes, show the distribution,
  not just its mean.

- **The bias is not specific to PSTHs.** Percent-change, normalized indices like
  (a-b)/(a+b), selectivity and modulation indices, dF/F with a short baseline,
  and any per-unit normalization that divides by a small noisy number all carry
  it. The rule of thumb: never let a quantity you are averaging have a noisy
  estimate in its denominator.
"""))

# ------------------------------------------------------------------
def build(cells, path):
    nb = new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'display_name': 'Python 3 (goodanalysis)',
                                 'language': 'python', 'name': 'python3'}
    NotebookClient(nb, timeout=900, kernel_name='python3',
                   resources={'metadata': {'path': HERE}}).execute()
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print('wrote', path)

build(student, os.path.join(HERE, 'notebook6_student.ipynb'))
build(solution, os.path.join(HERE, 'notebook6_solution.ipynb'))
