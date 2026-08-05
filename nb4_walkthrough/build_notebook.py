"""
Build the nb4 walkthrough notebook: "Nonsense correlations from slow fluctuations".

Unlike nb1-nb7 this is not a mystery to solve -- it is a guided demo to run
together with a class. The data are generated inside the notebook.

Run:  python build_notebook.py
Writes notebook4_walkthrough.ipynb (executed, with outputs).
"""

import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

# ===========================================================================
md("""\
# Nonsense correlations from slow fluctuations — a walkthrough

Neurons fluctuate slowly. So does pupil diameter, and so does a task variable
that stays constant for a run of trials. Any two slow signals correlate, and
ordinary statistics — p-values, cross-validation — will call that correlation
real.

We simulate a population with **no** relationship to pupil and **no** encoding of
the task variable, watch several standard analyses declare otherwise, then work
through which controls catch the problem and which do not.
""")

co("""\
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from rastermap import Rastermap

logging.getLogger('rastermap').setLevel(logging.WARNING)   # quiet its progress messages

plt.rcParams['font.family']       = 'Arial'
plt.rcParams['font.sans-serif']   = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['pdf.fonttype']      = 42
plt.rcParams['ps.fonttype']       = 42
""")

# ---------------------------------------------------------------- 1. simulate
md("""\
## 1. Simulate a session
""")

co("""\
SEED        = 7            # master random seed

N_NEURONS   = 100
N_BLOCKS    = 10           # task blocks in the session
BLOCK_LEN   = (30, 70)     # trials per block, drawn uniformly (inclusive)
TRIAL_DUR   = 1.0          # s, duration of one trial
DT          = 0.005        # s, resolution at which spikes are generated

FLUCT_TAU   = 25.0         # s, timescale of each neuron's slow rate fluctuation
PUPIL_TAU   = 20.0         # s, timescale of the pupil fluctuation
RATE_BASE   = (2.0, 9.0)   # spikes/s, each neuron's mean rate (drawn uniformly)
RATE_AMP    = 2.5          # spikes/s, sd of the slow fluctuation around that mean
TP_NOISE_SD = 0.0          # spikes/s, extra independent noise added to the rate at
                           # every timepoint, on top of the Poisson spiking noise
""")

co("""\
def slow_traces(rng, n_rows, n_pts, tau_pts):
    \"\"\"n_rows independent smooth random traces of length n_pts: white noise
    low-pass filtered with a Gaussian of sd tau_pts, then z-scored so each row
    has mean 0 and sd 1. mode='wrap' makes each trace periodic, so a circular
    shift of a trace is still a valid sample from the same process.\"\"\"
    z = gaussian_filter1d(rng.normal(0, 1, (n_rows, n_pts)), tau_pts, axis=1, mode='wrap')
    return z / z.std(axis=1, keepdims=True)


def make_blocks(rng, n_blocks=N_BLOCKS, lo=BLOCK_LEN[0], hi=BLOCK_LEN[1], n_trials=None):
    \"\"\"A sequence of alternating +1 / -1 blocks: each block's length is drawn
    uniformly from lo..hi trials and the first block's sign is random. Returns
    (value on each trial, block id on each trial). Pass n_trials to get exactly
    that many trials -- which is how you draw a pseudosession later on.\"\"\"
    sign = rng.choice([-1, 1])
    values, ids, i = [], [], 0
    while (len(values) < n_trials) if n_trials else (i < n_blocks):
        L = int(rng.integers(lo, hi + 1))
        values += [sign * (-1) ** i] * L      # this block's value, repeated
        ids    += [i] * L                     # this block's index, repeated
        i += 1
    values, ids = np.array(values), np.array(ids)
    return (values[:n_trials], ids[:n_trials]) if n_trials else (values, ids)
""")

co("""\
rng = np.random.default_rng(SEED)

# --- the task: a sequence of +/-1 blocks
block_values, block_ids = make_blocks(rng)
n_trials  = len(block_values)
per_trial = int(round(TRIAL_DUR / DT))        # timepoints in one trial
n_pts     = n_trials * per_trial              # timepoints in the session

# --- each neuron's firing rate: its own mean rate plus a slow fluctuation.
# Nothing here refers to the block variable or to the pupil: the neurons are
# generated completely independently of both.
rate_trial = (rng.uniform(*RATE_BASE, size=N_NEURONS)[:, None]
              + RATE_AMP * slow_traces(rng, N_NEURONS, n_trials, FLUCT_TAU / TRIAL_DUR))
rate_trial = rate_trial + rng.normal(0, TP_NOISE_SD, rate_trial.shape)   # 0 by default
rate_trial = np.clip(rate_trial, 0, None)     # a firing rate cannot be negative

# --- interpolate the per-trial rate onto the fine time grid, then draw spikes
# from an inhomogeneous Poisson process: independently in each 5 ms bin, the
# number of spikes is Poisson with mean rate*DT.
t_trial = (np.arange(n_trials) + 0.5) * TRIAL_DUR      # trial centres, s
t_pts   = np.arange(n_pts) * DT                        # fine time grid, s
rate    = np.array([np.interp(t_pts, t_trial, r) for r in rate_trial])   # spikes/s
spikes  = rng.poisson(rate * DT)                       # (neurons, timepoints)

# --- pupil diameter: the same kind of slow trace, also independent of everything
pupil = 0.5 + 0.15 * slow_traces(rng, 1, n_trials, PUPIL_TAU / TRIAL_DUR)[0]

print(f'{N_NEURONS} neurons, {n_trials} trials of {TRIAL_DUR:.0f} s '
      f'({n_trials*TRIAL_DUR/60:.1f} min), {N_BLOCKS} blocks')
print(f'firing rates span {rate.min():.1f} - {rate.max():.1f} spikes/s '
      f'(mean {rate.mean():.1f}); {spikes.sum():,} spikes total')
""")

# ---------------------------------------------------------------- 2. rasters
md("""\
## 2. Spike rasters
""")

co("""\
N_SHOW, T_SHOW = 12, 60      # neurons and seconds to display

fig, ax = plt.subplots(figsize=(9.5, 4))
# spike times of neuron i = the times of its non-empty bins, repeated by count
times = [np.repeat(t_pts, spikes[i]) for i in range(N_SHOW)]
ax.eventplot([t[t <= T_SHOW] for t in times], colors='k', linewidths=0.7,
             lineoffsets=np.arange(N_SHOW), linelengths=0.7)
ax.set_xlim(0, T_SHOW); ax.set_ylim(N_SHOW - 0.5, -0.5)
ax.set_yticks(np.arange(0, N_SHOW, 2))
ax.set_xlabel('time (s)'); ax.set_ylabel('neuron')
ax.set_title(f'Spike rasters: {N_SHOW} neurons, first {T_SHOW} s ({T_SHOW} trials)')
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 3. binning
md("""\
## 3. Spike counts binned by trial
""")

co("""\
# one bin per trial: sum the fine-grained counts within each trial
counts = spikes.reshape(N_NEURONS, n_trials, per_trial).sum(axis=2)   # spikes per trial
fr     = counts / TRIAL_DUR                                          # spikes/s
# z-score each neuron across trials, so neurons with different mean rates are
# on the same colour scale
frz = (fr - fr.mean(axis=1, keepdims=True)) / fr.std(axis=1, keepdims=True)


def show_matrix(ax, M, cmap, vmin, vmax, title, cbar_label, ylabel='neuron'):
    \"\"\"Draw a neurons x trials matrix. interpolation='nearest' is important:
    it stops matplotlib from blurring the image across neurons or across trials,
    which would invent structure that is not in the data.\"\"\"
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', extent=[0, M.shape[1], M.shape[0], 0])
    ax.set_ylabel(ylabel); ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.025, pad=0.01)
    return im


fig, axes = plt.subplots(2, 1, figsize=(9.5, 6), sharex=True)
show_matrix(axes[0], fr, 'magma', 0, np.percentile(fr, 99.5),
            'Spike counts binned by trial', 'firing rate (spikes/s)')
show_matrix(axes[1], frz, 'RdBu_r', -3, 3,
            'Same matrix, z-scored per neuron', 'z-score')
axes[1].set_xlabel('trial')
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 4. rastermap
md("""\
## 4. The same matrix, rows sorted by Rastermap
""")

co("""\
def rastermap_order(M):
    \"\"\"The neuron ordering Rastermap finds for a neurons x trials matrix: it
    puts neurons with similar activity next to each other.\"\"\"
    return Rastermap(n_clusters=None, n_PCs=32, locality=0.5, time_lag_window=0).fit(M).isort


isort = rastermap_order(frz)   # fit on the whole matrix: no held-out data anywhere

fig, ax = plt.subplots(figsize=(9.5, 3.6))
show_matrix(ax, frz[isort], 'RdBu_r', -3, 3,
            'Rastermap-sorted activity (sort and display use the same data)',
            'z-score', ylabel='neuron (Rastermap order)')
ax.set_xlabel('trial')
plt.tight_layout(); plt.show()
""")

md("""\
### Cross-validating the sort: two ways to hold out half the trials
""")

co("""\
def neighbour_corr(order, M):
    \"\"\"Mean correlation between neurons that `order` placed next to each other,
    measured on the matrix M. If the sort captured shared fluctuation that is
    really there, this stays high on trials the sort never saw.\"\"\"
    C = np.corrcoef(M)
    return np.mean(C[order[:-1], order[1:]])


def mean_pair_corr(M):
    \"\"\"Mean correlation over all neuron pairs: what neighbour_corr would give
    for an ordering that captured nothing.\"\"\"
    C = np.corrcoef(M)
    return np.mean(C[np.triu_indices(len(C), 1)])


odd, even     = np.arange(1, n_trials, 2), np.arange(0, n_trials, 2)          # interleaved
first, second = np.arange(n_trials // 2), np.arange(n_trials // 2, n_trials)  # contiguous

order_odd   = rastermap_order(frz[:, odd])     # sort using only the odd trials
order_first = rastermap_order(frz[:, first])   # sort using only the first half

fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
show_matrix(axes[0, 0], frz[order_odd][:, odd], 'RdBu_r', -3, 3,
            'Sorted on odd trials, showing odd trials', 'z-score',
            ylabel='neuron (odd-trial order)')
show_matrix(axes[0, 1], frz[order_odd][:, even], 'RdBu_r', -3, 3,
            'Same order, showing held-out EVEN trials', 'z-score',
            ylabel='neuron (odd-trial order)')
show_matrix(axes[1, 0], frz[order_first][:, first], 'RdBu_r', -3, 3,
            'Sorted on the 1st half, showing the 1st half', 'z-score',
            ylabel='neuron (1st-half order)')
show_matrix(axes[1, 1], frz[order_first][:, second], 'RdBu_r', -3, 3,
            'Same order, showing the held-out 2nd half', 'z-score',
            ylabel='neuron (1st-half order)')
for ax, lab in zip(axes.ravel(), ['odd trial', 'even trial', 'trial', 'trial']):
    ax.set_xlabel(lab)
plt.tight_layout(); plt.show()

print('mean correlation between neurons that ended up adjacent in the sort:')
print(f'  interleaved split: fit (odd) {neighbour_corr(order_odd, frz[:, odd]):+.3f}   '
      f'held out (even) {neighbour_corr(order_odd, frz[:, even]):+.3f}   '
      f'[all pairs {mean_pair_corr(frz[:, even]):+.3f}]')
print(f'  half split       : fit (1st) {neighbour_corr(order_first, frz[:, first]):+.3f}   '
      f'held out (2nd)  {neighbour_corr(order_first, frz[:, second]):+.3f}   '
      f'[all pairs {mean_pair_corr(frz[:, second]):+.3f}]')
""")

md("""\
Interleaving the trials is not a test of the sort. Neighbouring trials share the
same slow fluctuation, so the held-out half carries the same chance correlations
the sort was fit to, and the structure replicates. Splitting the session into
contiguous halves *is* a test, and the ordering does not transfer.

(In `nb3` the trials really were independent, so there an odd/even split was a
valid test. The same split fails here for exactly the reason ordinary k-fold
cross-validation will fail in section 8.)
""")

# ---------------------------------------------------------------- 5. pupil
md("""\
## 5. Test 1 — correlation with pupil diameter
""")

co("""\
fig, ax = plt.subplots(figsize=(9.5, 2))
ax.plot(pupil, color='tab:purple', lw=1.2)
ax.set_xlim(0, n_trials)
ax.set_xlabel('trial'); ax.set_ylabel('pupil diameter (a.u.)')
ax.set_title('Pupil diameter across the session')
plt.tight_layout(); plt.show()
""")

co("""\
# Pearson correlation of every neuron's trial-by-trial firing rate with the pupil
# trace, plus the p-value scipy reports for it (a two-sided t-test on r, n-2 df).
res     = [stats.pearsonr(row, pupil) for row in fr]
r_pupil = np.array([x.statistic for x in res])
p_pupil = np.array([x.pvalue for x in res])
n_sig   = np.sum(p_pupil < 0.05)

# With this many trials, |r| barely has to leave zero to be called significant.
# Inverting the same t-test gives the threshold value of |r| explicitly:
t_crit = stats.t.ppf(1 - 0.05 / 2, n_trials - 2)              # two-sided, n-2 df
r_crit = t_crit / np.sqrt(t_crit ** 2 + n_trials - 2)
print(f'n = {n_trials} trials  ->  any |r| > {r_crit:.3f} gives p < 0.05')

# One bin grid for every correlation histogram in this notebook, with +/-r_crit
# exactly on bin edges so that no bar mixes significant with non-significant.
_edges = np.arange(0, 0.8 + r_crit, r_crit)
R_BINS = np.concatenate([-_edges[1:][::-1], _edges])
""")

co("""\
def hist_by_significance(ax, r, title):
    \"\"\"Histogram of correlations with the significant bars in red: away from zero
    in either direction is significant, near zero is not.\"\"\"
    _, _, patches = ax.hist(r, bins=R_BINS)
    for patch, lo, hi in zip(patches, R_BINS[:-1], R_BINS[1:]):
        patch.set_facecolor('tab:red' if abs(lo + hi) / 2 > r_crit else '0.6')
    for side in (-1, 1):
        ax.axvline(side * r_crit, color='k', ls='--', lw=0.8)
    ax.set_xlabel("correlation with pupil (Pearson's r)"); ax.set_ylabel('neurons')
    ax.set_title(title)
    ax.legend(handles=[Patch(facecolor='tab:red', label='p < 0.05'),
                       Patch(facecolor='0.6', label='n.s.')],
              frameon=False, fontsize=8, loc='upper left')


fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
hist_by_significance(axes[0], r_pupil, 'Correlation with pupil diameter')
axes[1].hist(p_pupil, bins=np.linspace(0, 1, 21), color='0.5')
axes[1].axvline(0.05, color='tab:red', lw=1.2, ls='--', label='p = 0.05')
axes[1].set_xlabel('p-value'); axes[1].set_ylabel('neurons')
axes[1].set_title('p-values')
axes[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

print(f'{n_sig} / {N_NEURONS} neurons significantly correlated with pupil at p < 0.05 '
      f'({n_sig/N_NEURONS*100:.0f}%; 5% expected by chance)')
print(f'median |r| = {np.median(np.abs(r_pupil)):.3f}, '
      f'largest |r| = {np.max(np.abs(r_pupil)):.3f}')
""")

co("""\
order_pupil = np.argsort(r_pupil)[::-1]        # most positively correlated first

fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.8), sharex=True,
                         gridspec_kw=dict(height_ratios=[1, 3]))
axes[0].plot(pupil, color='tab:purple', lw=1.2)
axes[0].set_ylabel('pupil (a.u.)'); axes[0].set_title('Pupil diameter')
show_matrix(axes[1], frz[order_pupil], 'RdBu_r', -3, 3,
            'Activity, neurons sorted by their correlation with pupil',
            'z-score', ylabel='neuron (sorted by r)')
axes[1].set_xlabel('trial'); axes[1].set_xlim(0, n_trials)
plt.tight_layout(); plt.show()
""")

md("""\
### Click a row to overlay that neuron on the pupil trace
""")

co("""\
%matplotlib widget

pupil_z = (pupil - pupil.mean()) / pupil.std()
M_sorted = frz[order_pupil]

fig, (ax_im, ax_tr) = plt.subplots(2, 1, figsize=(9.5, 6.5),
                                   gridspec_kw=dict(height_ratios=[2, 1.4]))
fig.canvas.header_visible = False
show_matrix(ax_im, M_sorted, 'RdBu_r', -3, 3,
            'Click a row to plot that neuron below', 'z-score',
            ylabel='neuron (sorted by r)')
ax_im.set_xlabel('trial')
row_marker = ax_im.axhline(0.5, color='lime', lw=1.5)

line_neuron, = ax_tr.plot(M_sorted[0], color='0.55', lw=0.8, label='neuron (z)')
ax_tr.plot(pupil_z, color='tab:purple', lw=1.8, label='pupil (z)')
ax_tr.set_xlim(0, n_trials); ax_tr.set_xlabel('trial'); ax_tr.set_ylabel('z-score')
ax_tr.legend(frameon=False, fontsize=8, ncol=2, loc='upper right')


def show_row(row):
    \"\"\"Draw the neuron in display row `row` of the sorted matrix.\"\"\"
    i = order_pupil[row]
    line_neuron.set_ydata(frz[i])
    row_marker.set_ydata([row + 0.5, row + 0.5])
    ax_tr.set_title(f'neuron {i}:   r = {r_pupil[i]:+.2f},   p = {p_pupil[i]:.1e}')
    ax_tr.relim(); ax_tr.autoscale_view(scalex=False)
    fig.canvas.draw_idle()


def on_click(event):
    if event.inaxes is ax_im and event.ydata is not None:
        show_row(int(np.clip(event.ydata, 0, N_NEURONS - 1)))


fig.canvas.mpl_connect('button_press_event', on_click)
show_row(0)
plt.tight_layout()
""")

co("""\
%matplotlib inline
""")

# ---------------------------------------------------------------- 6. shuffle
md("""\
## 6. A control that fails: shuffle each neuron's timecourse
""")

co("""\
# independently permute the trials of each neuron: this keeps every neuron's
# distribution of firing rates but destroys its slow temporal structure
fr_shuffled = np.array([rng.permutation(row) for row in fr])
frz_shuffled = ((fr_shuffled - fr_shuffled.mean(axis=1, keepdims=True))
                / fr_shuffled.std(axis=1, keepdims=True))

res_sh     = [stats.pearsonr(row, pupil) for row in fr_shuffled]   # vs the real pupil
r_shuffled = np.array([x.statistic for x in res_sh])
p_shuffled = np.array([x.pvalue for x in res_sh])
n_sig_shuffled = np.sum(p_shuffled < 0.05)

fig, ax = plt.subplots(figsize=(9.5, 3.4))
show_matrix(ax, frz_shuffled, 'RdBu_r', -3, 3,
            'Shuffled activity (trials permuted independently per neuron)', 'z-score')
ax.set_xlabel('trial')
plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.2))
hist_by_significance(axes[0], r_pupil, 'Real data')
hist_by_significance(axes[1], r_shuffled, 'After shuffling trials')
axes[2].hist(p_shuffled, bins=np.linspace(0, 1, 21), color='0.5')
axes[2].axvline(0.05, color='tab:red', lw=1.2, ls='--', label='p = 0.05')
axes[2].set_xlabel('p-value'); axes[2].set_ylabel('neurons')
axes[2].set_title('p-values after shuffling')
axes[2].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

print(f'real     : {n_sig:3d} / {N_NEURONS} neurons significant (p < 0.05)')
print(f'shuffled : {n_sig_shuffled:3d} / {N_NEURONS} neurons significant (p < 0.05)')
""")

md("""\
The shuffle sits at chance, so the real result looks like a real effect. But the
shuffle destroyed the slow structure that produced it — so it was never a test
of the claim.
""")

# ---------------------------------------------------------------- 7. controls
md("""\
## 7. Two controls that work
""")

md("""\
### 7a. Circularly shift each neuron
""")

co("""\
def corr_rows(M, v):
    \"\"\"Pearson correlation between every row of M and the vector v: the same
    number scipy.stats.pearsonr returns, written out as a dot product of z-scored
    variables (which is all a Pearson correlation is). Each null below needs
    20,000 correlations; looping over scipy.stats.pearsonr takes ~15 s for that,
    this takes ~0.1 s.\"\"\"
    Mz = (M - M.mean(axis=1, keepdims=True)) / M.std(axis=1, keepdims=True)
    vz = (v - v.mean()) / v.std()
    return (Mz @ vz) / v.size


print('corr_rows matches scipy.stats.pearsonr:',
      np.allclose(corr_rows(fr, pupil), r_pupil))
""")

co("""\
N_SHIFT = 200
shift_amounts = rng.integers(1, n_trials, N_SHIFT)

# A circular shift slides a neuron's whole timecourse in time, keeping its slow
# structure intact but breaking any true alignment with the pupil trace.
r_circ = np.array([corr_rows(np.roll(fr, s, axis=1), pupil) for s in shift_amounts])
# per-neuron p-value: how often does a shifted version reach |r| as large as the
# unshifted one?
p_circ = (np.sum(np.abs(r_circ) >= np.abs(r_pupil), axis=0) + 1) / (N_SHIFT + 1)
n_sig_circ = np.sum(p_circ < 0.05)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
axes[0].hist(r_circ.ravel(), bins=R_BINS, density=True,
             color='0.6', label='circular shifts (null)')
axes[0].hist(r_pupil, bins=R_BINS, density=True,
             color='tab:blue', alpha=0.6, label='real')
axes[0].set_xlabel("correlation with pupil (Pearson's r)"); axes[0].set_ylabel('density')
axes[0].set_title('Circular-shift null covers the real correlations')
axes[0].legend(frameon=False, fontsize=8)
axes[1].hist(p_circ, bins=np.linspace(0, 1, 21), color='0.6')
axes[1].axvline(0.05, color='tab:red', lw=1.2, ls='--', label='p = 0.05')
axes[1].set_xlabel('p-value vs circular-shift null'); axes[1].set_ylabel('neurons')
axes[1].set_title('p-values')
axes[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

print(f'parametric p < 0.05        : {n_sig:3d} / {N_NEURONS} neurons')
print(f'vs circular-shift null     : {n_sig_circ:3d} / {N_NEURONS} neurons')
""")

md("""\
### 7b. Correlate with a different session's pupil trace
""")

co("""\
N_SESSIONS = 200
# pupil traces from "other sessions": drawn from the same generative process,
# so they have the same slow structure but no relation to these neurons
other_pupil = 0.5 + 0.15 * slow_traces(rng, N_SESSIONS, n_trials, PUPIL_TAU / TRIAL_DUR)

r_other = np.array([corr_rows(fr, p) for p in other_pupil])
p_other = (np.sum(np.abs(r_other) >= np.abs(r_pupil), axis=0) + 1) / (N_SESSIONS + 1)
n_sig_other = np.sum(p_other < 0.05)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
axes[0].hist(r_other.ravel(), bins=R_BINS, density=True,
             color='0.6', label="other sessions' pupil (null)")
axes[0].hist(r_pupil, bins=R_BINS, density=True,
             color='tab:blue', alpha=0.6, label='this session')
axes[0].set_xlabel("correlation with pupil (Pearson's r)"); axes[0].set_ylabel('density')
axes[0].set_title('Session-permutation null covers the real correlations')
axes[0].legend(frameon=False, fontsize=8)
axes[1].hist(p_other, bins=np.linspace(0, 1, 21), color='0.6')
axes[1].axvline(0.05, color='tab:red', lw=1.2, ls='--', label='p = 0.05')
axes[1].set_xlabel('p-value vs session-permutation null'); axes[1].set_ylabel('neurons')
axes[1].set_title('p-values')
axes[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

print(f'parametric p < 0.05        : {n_sig:3d} / {N_NEURONS} neurons')
print(f'vs session-permutation null: {n_sig_other:3d} / {N_NEURONS} neurons')
""")

# ---------------------------------------------------------------- 8. decoding
md("""\
## 8. Test 2 — decoding the block variable
""")

co("""\
fig, ax = plt.subplots(figsize=(9.5, 1.8))
ax.plot(block_values, '.-', ms=3, lw=0.8, color='k')
ax.set_xlim(0, n_trials); ax.set_yticks([-1, 1])
ax.set_xlabel('trial'); ax.set_ylabel('block value')
ax.set_title(f'The block variable ({N_BLOCKS} blocks, {BLOCK_LEN[0]}-{BLOCK_LEN[1]} trials each)')
plt.tight_layout(); plt.show()
""")

co("""\
X = fr.T                  # sklearn wants (samples, features) = (trials, neurons)
y = block_values


def trialwise_cv(X, y, folds=10, seed=0):
    \"\"\"Ordinary k-fold cross-validation: trials are shuffled and split into
    folds, each fold is predicted by a decoder trained on the other folds.\"\"\"
    kf = KFold(folds, shuffle=True, random_state=seed)
    return np.mean([LinearDiscriminantAnalysis().fit(X[tr], y[tr]).score(X[te], y[te])
                    for tr, te in kf.split(X)])


acc = trialwise_cv(X, y)
print(f'Trial-wise cross-validated accuracy: {acc*100:.1f}%   (chance = 50%)')
""")

co("""\
def plot_fold(ax, X, y, train, test, title):
    \"\"\"Train a decoder on `train` and plot its continuous output for every
    trial: training trials in black, held-out trials green if the sign of the
    output matches the true block and red if it does not.\"\"\"
    model = LinearDiscriminantAnalysis().fit(X[train], y[train])
    out = model.decision_function(X)              # continuous decoder output
    ok = np.sign(out[test]) == y[test]
    for b in np.unique(block_ids):               # shade the +1 blocks
        idx = np.where(block_ids == b)[0]
        if block_values[idx[0]] > 0:
            ax.axvspan(idx[0], idx[-1] + 1, color='0.9', lw=0, zorder=0)
    ax.plot(train, out[train], '.', color='k', ms=2.5,
            label=f'training trials ({len(train)/len(y)*100:.0f}%)')
    ax.plot(test[ok], out[test][ok], 'o', color='tab:green', ms=5, label='held-out, correct')
    ax.plot(test[~ok], out[test][~ok], 'o', color='tab:red', ms=5, label='held-out, wrong')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlim(0, n_trials); ax.set_xlabel('trial'); ax.set_ylabel('decoder output')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.22))
    return np.mean(ok)


train, test = next(iter(KFold(10, shuffle=True, random_state=0).split(X)))
fig, ax = plt.subplots(figsize=(9.5, 3.8))
fold_acc = plot_fold(ax, X, y, train, test, 'One fold of trial-wise CV (shaded = +1 blocks)')
ax.set_title(f'One fold of trial-wise CV (shaded = +1 blocks): '
             f'{fold_acc*100:.0f}% of held-out trials correct')
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 9. shuffle
md("""\
## 9. The same failing control: shuffle each neuron's timecourse
""")

co("""\
N_SHUFFLE = 20
acc_shuffled = np.array([
    trialwise_cv(np.array([rng.permutation(row) for row in fr]).T, y)
    for _ in range(N_SHUFFLE)])

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(acc_shuffled * 100, bins=12, color='0.6', label=f'shuffled ({N_SHUFFLE} draws)')
ax.axvline(acc * 100, color='tab:red', lw=2, label=f'real = {acc*100:.1f}%')
ax.axvline(50, color='0.3', ls='--', lw=1, label='chance')
ax.set_xlabel('trial-wise CV accuracy (%)'); ax.set_ylabel('count')
ax.set_title('Shuffling trials destroys the decoding')
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

print(f'real     : {acc*100:.1f}%')
print(f'shuffled : {acc_shuffled.mean()*100:.1f}% +/- {acc_shuffled.std()*100:.1f}% '
      f'(max {acc_shuffled.max()*100:.1f}%)')
""")

# ---------------------------------------------------------------- 10. controls
md("""\
## 10. Two controls that work
""")

md("""\
### 10a. Pseudosessions
""")

co("""\
N_PSEUDO = 100
# A pseudosession is a block sequence drawn from the same generative process as
# the real one -- same number of blocks, same length distribution -- but it is
# not the sequence this session actually had. Decode each one the same way.
acc_pseudo = np.array([
    trialwise_cv(X, make_blocks(rng, n_trials=n_trials)[0], folds=5)
    for _ in range(N_PSEUDO)])
acc_real5 = trialwise_cv(X, y, folds=5)
p_pseudo = (np.sum(acc_pseudo >= acc_real5) + 1) / (N_PSEUDO + 1)

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.hist(acc_pseudo * 100, bins=15, color='0.6', label=f'pseudosessions (n = {N_PSEUDO})')
ax.axvline(acc_real5 * 100, color='tab:red', lw=2, label=f'real = {acc_real5*100:.1f}%')
ax.axvline(50, color='0.3', ls='--', lw=1, label='chance')
ax.set_xlabel('trial-wise CV accuracy (%)'); ax.set_ylabel('count')
ax.set_title(f'The real block sequence decodes no better than a made-up one (p = {p_pseudo:.2f})')
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

print(f'pseudosessions: {acc_pseudo.mean()*100:.1f}% +/- {acc_pseudo.std()*100:.1f}%')
print(f'real          : {acc_real5*100:.1f}%    p = {p_pseudo:.2f}')
""")

md("""\
### 10b. Leave one block out
""")

co("""\
# Hold out every trial of one block at a time, so no trial from the held-out
# block -- and none of its slow fluctuation state -- is in the training set.
blocks = np.unique(block_ids)
acc_blockout = np.array([
    LinearDiscriminantAnalysis().fit(X[block_ids != b], y[block_ids != b])
    .score(X[block_ids == b], y[block_ids == b]) for b in blocks])

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.bar(blocks, acc_blockout * 100,
       color=['tab:red' if block_values[block_ids == b][0] > 0 else 'tab:blue'
              for b in blocks])
ax.axhline(np.mean(acc_blockout) * 100, color='k',
           label=f'mean = {np.mean(acc_blockout)*100:.0f}%')
ax.axhline(50, color='0.3', ls='--', lw=1, label='chance')
ax.set_xticks(blocks); ax.set_ylim(0, 100)
ax.set_xlabel('held-out block'); ax.set_ylabel('accuracy (%)')
ax.set_title('Leave-one-block-out: worse than chance (red = +1 block, blue = -1)')
ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()

print(f'trial-wise CV       : {acc*100:.1f}%')
print(f'leave-one-block-out : {np.mean(acc_blockout)*100:.1f}%')
""")

co("""\
# One held-out block in the style of the fold plot above: every trial of this
# block is in the test set, and the decoder mostly calls it the wrong sign --
# the neighbouring blocks, which the decoder did train on, had the opposite value.
b = int(blocks[np.argmin(acc_blockout)])
test_b = np.where(block_ids == b)[0]
train_b = np.where(block_ids != b)[0]

fig, ax = plt.subplots(figsize=(9.5, 3.8))
plot_fold(ax, X, y, train_b, test_b,
          f'Held-out block {b} (a {block_values[test_b[0]]:+d} block): '
          f'{acc_blockout[b]*100:.0f}% correct')
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'display_name': 'Python 3 (swdb2026)',
                             'language': 'python', 'name': 'python3'}
NotebookClient(nb, timeout=1800, kernel_name='python3',
               resources={'metadata': {'path': HERE}}).execute()
out = os.path.join(HERE, 'notebook4_walkthrough.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print('wrote', out)
