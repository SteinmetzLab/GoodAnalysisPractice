
# Standard setup
# standard imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold




# Global figure settings
plt.rcParams['font.family']       = 'sans-serif'
plt.rcParams['font.sans-serif']   = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['pdf.fonttype']      = 42
plt.rcParams['ps.fonttype']       = 42

# Part 1 helper functions
SEED        = 7            # master random seed — keep this fixed to reproduce figures

N_NEURONS   = 100
N_BLOCKS    = 10           # task blocks in the session
BLOCK_LEN   = (30, 70)     # trials per block, drawn uniformly (inclusive)
TRIAL_DUR   = 1.0          # s, duration of one trial
DT          = 0.005        # s, resolution at which spikes are generated (5 ms bins)

FLUCT_TAU   = 25.0         # s, timescale of each neuron's slow rate fluctuation
PUPIL_TAU   = 20.0         # s, timescale of the pupil fluctuation
RATE_BASE   = (2.0, 9.0)   # spikes/s, each neuron's mean rate (drawn uniformly)
RATE_AMP    = 2.5          # spikes/s, SD of the slow fluctuation around that mean
TP_NOISE_SD = 0.0          # extra independent noise (0 = Poisson noise only)

# cell 10
def slow_traces(rng, n_rows, n_pts, tau_pts):
    """Generate smooth random traces with a given timescale.

    White noise low-pass filtered with a Gaussian kernel, then z-scored
    row-wise. mode='wrap' makes each trace periodic so a circular shift
    is a valid sample from the same generative process.

    Input Parameters
    ----------
    rng     : np.random.Generator
    n_rows  : int   — number of independent traces (e.g. neurons)
    n_pts   : int   — length of each trace in samples (e.g. trials)
    tau_pts : float — timescale in samples (larger = slower fluctuations)

    Returns
    -------
    traces : array (n_rows, n_pts) — z-scored smooth random traces
    """
    z = gaussian_filter1d(rng.normal(0, 1, (n_rows, n_pts)), tau_pts, axis=1, mode='wrap')
    return z / z.std(axis=1, keepdims=True)

# cell 10
def make_blocks(rng, n_blocks=N_BLOCKS, lo=BLOCK_LEN[0], hi=BLOCK_LEN[1], n_trials=None):
    """Sequence of alternating +1/−1 blocks, random lengths in [lo, hi].
    Pass n_trials to draw exactly that many trials (used for pseudosessions).

    Input Parameters
    ----------
    rng      : np.random.Generator
    n_blocks : int          — number of blocks (used when n_trials is None)
    lo, hi   : int          — minimum and maximum block length in trials
    n_trials : int or None  — if given, draw exactly this many trials
                              (truncates the last block); used for pseudosessions

    Returns
    -------
    values : array (n_trials,) or (n_blocks * mean_len,) — +1 / -1 label per trial
    ids    : array (same shape)                          — block index per trial
    """
    sign = rng.choice([-1, 1])
    values, ids, i = [], [], 0
    while (len(values) < n_trials) if n_trials else (i < n_blocks):
        L = int(rng.integers(lo, hi + 1))
        values += [sign * (-1) ** i] * L
        ids    += [i] * L
        i += 1
    values, ids = np.array(values), np.array(ids)
    return (values[:n_trials], ids[:n_trials]) if n_trials else (values, ids)



# cell 21
def show_matrix(ax, M, cmap, vmin, vmax, title, cbar_label, ylabel='neuron'):
    """Display a neurons x trials matrix as a heatmap.

    Uses nearest-neighbour interpolation to avoid blurring across trials
    or neurons, which would invent structure that isn't in the data.

    Parameters
    ----------
    ax          : matplotlib Axes
    M           : array (n_neurons, n_trials) — matrix to display
    cmap        : str   — colormap name (e.g. 'RdBu_r')
    vmin, vmax  : float — colormap limits
    title       : str
    cbar_label  : str   — colorbar axis label
    ylabel      : str   — y-axis label (default 'neuron')

    Returns
    -------
    im : AxesImage — the imshow object (useful for adding a colorbar elsewhere)
    """
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', extent=[0, M.shape[1], M.shape[0], 0])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label, fraction=0.025, pad=0.01)
    return im

# cell 27
def crit_r(n):
    """Critical |r| for a two-sided Pearson correlation test at p < 0.05.

    Inverts the t-statistic for a t-distribution with n-2 degrees of freedom
    to find the correlation threshold at which p = 0.05.

    Parameters
    ----------
    n : int — number of observations (trials)

    Returns
    -------
    r_crit : float — minimum |r| for significance at p < 0.05
    """
    t = stats.t.ppf(1 - 0.05 / 2, n - 2)
    return t / np.sqrt(t ** 2 + n - 2)


# cell 29
def hist_by_significance(ax, r, title, bins, crit):
    """Histogram of correlations coloured by significance.

    Parameters
    ----------
    ax    : matplotlib Axes
    r     : array (n_neurons,) — per-neuron correlations
    title : str
    bins  : array             — bin edges
    crit  : float             — significance threshold
    """
    _, _, patches = ax.hist(r, bins=bins)
    for patch, lo, hi in zip(patches, bins[:-1], bins[1:]):
        patch.set_facecolor('tab:red' if abs(lo + hi) / 2 > crit else '0.6')
    for side in (-1, 1):
        ax.axvline(side * crit, color='k', ls='--', lw=0.8)
    ax.set_xlabel("correlation with pupil (Pearson's r)")
    ax.set_ylabel('neurons')
    ax.set_title(title)
    ax.legend(handles=[Patch(facecolor='tab:red', label='p < 0.05'),
                       Patch(facecolor='0.6', label='n.s.')],
              frameon=False, fontsize=8, loc='upper left')

# cell 45
def corr_rows(M, v):
    """Pearson correlation between every row of M and a vector v.

    Computed as a dot product of z-scored arrays — equivalent to
    scipy.stats.pearsonr but vectorised across all rows simultaneously.

    Parameters
    ----------
    M : array (n_rows, n_pts) — matrix of signals (e.g. firing rates)
    v : array (n_pts,)        — signal to correlate against (e.g. pupil trace)

    Returns
    -------
    r : array (n_rows,) — Pearson r for each row of M vs v
    """
    Mz = (M - M.mean(axis=1, keepdims=True)) / M.std(axis=1, keepdims=True)
    vz = (v - v.mean()) / v.std()
    return (Mz @ vz) / v.size    

# cell 55
def autocorr(x, max_lag):
    """Autocorrelation of x at lags 0 through max_lag, normalised to 1 at lag 0.

    Parameters
    ----------
    x       : array (n,) — input signal (e.g. one neuron's firing rate)
    max_lag : int        — maximum lag to compute

    Returns
    -------
    ac : array (max_lag + 1,) — autocorrelation at lags 0, 1, ..., max_lag
    """
    x = x - x.mean()
    full = np.correlate(x, x, 'full')[len(x) - 1:]
    return (full / full[0])[:max_lag + 1]

def effective_n(x, y):
    """Bartlett's effective sample size for two autocorrelated series.

    Corrects for the fact that autocorrelated signals contain fewer
    independent observations than their raw length suggests. When both
    signals are uncorrelated, returns n. When both wander slowly,
    returns n_eff << n.

    Parameters
    ----------
    x : array (n,) — first signal  (e.g. a neuron's firing rate)
    y : array (n,) — second signal (e.g. the pupil trace)

    Returns
    -------
    n_eff : float — equivalent number of independent paired observations
    """
    n = len(x)
    max_lag = n // 4
    k = np.arange(1, max_lag + 1)
    return n / (1 + 2 * np.sum((1 - k / n) * autocorr(x, max_lag)[1:]
                               * autocorr(y, max_lag)[1:]))


# cell 63
def plot_fold(ax, X, y, train, test, block_ids, block_values, n_trials, title):
    """Train an LDA decoder and plot its output across all trials.

    Trains on the provided training indices and plots the continuous
    decision function for all trials. Training trials are shown as black
    dots; held-out trials are coloured green (correct) or red (incorrect).
    +1 blocks are shaded gray. The decision boundary is drawn at 0.

    Parameters
    ----------
    ax           : matplotlib Axes
    X            : array (n_trials, n_neurons) — firing rates in sklearn format
    y            : array (n_trials,)           — block labels (+1 / -1)
    train        : array                       — indices of training trials
    test         : array                       — indices of held-out test trials
    block_ids    : array (n_trials,)           — block index for each trial
    block_values : array (n_trials,)           — block label (+1 / -1) for each trial
    n_trials     : int                         — total number of trials (sets x-axis limit)
    title        : str

    Returns
    -------
    accuracy : float — fraction of held-out trials correctly decoded
    """
    model = LinearDiscriminantAnalysis().fit(X[train], y[train])
    out   = model.decision_function(X)
    ok    = np.sign(out[test]) == y[test]
    for b in np.unique(block_ids):
        idx = np.where(block_ids == b)[0]
        if block_values[idx[0]] > 0:
            ax.axvspan(idx[0], idx[-1] + 1, color='0.9', lw=0, zorder=0)
    ax.plot(train, out[train], '.', color='k', ms=2.5,
            label=f'training trials ({len(train)/len(y)*100:.0f}%)')
    ax.plot(test[ok],  out[test][ok],  'o', color='tab:green', ms=5, label='held-out, correct')
    ax.plot(test[~ok], out[test][~ok], 'o', color='tab:red',   ms=5, label='held-out, wrong')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlim(0, n_trials)
    ax.set_xlabel('trial')
    ax.set_ylabel('decoder output')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc='upper center',
              bbox_to_anchor=(0.5, -0.22))
    return np.mean(ok)


# cell 73
def blockout_cv(X, y, ids):
    """Leave-one-block-out cross-validation for block decoding.

    For each block, trains an LDA decoder on all other blocks and tests
    on the held-out block. No trial from the held-out block is ever in
    the training set, preventing slow drift from leaking across the
    train/test boundary.

    Parameters
    ----------
    X   : array (n_trials, n_neurons) — firing rates in sklearn format
    y   : array (n_trials,)           — block labels (+1 / -1)
    ids : array (n_trials,)           — block index for each trial

    Returns
    -------
    acc : array (n_blocks,) — decoding accuracy for each held-out block
    """
    return np.array([
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        .fit(X[ids != b], y[ids != b])
        .score(X[ids == b], y[ids == b])
        for b in np.unique(ids)
    ])