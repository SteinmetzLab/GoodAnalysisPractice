"""
Shared guts of the drill-down demo: the session container and the drawing
functions, written so they render into axes that somebody else owns.

Two front ends use this:
  drilldown.py            -- standalone script, one window per level
  drilldown_notebook.ipynb -- one figure, six panels, %matplotlib widget

Each draw_* function fills in the axes it is given and returns whatever the
caller needs to turn a click into a selection. Nothing here creates figures,
shows windows, or connects events.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import generate_data as gd

# ----------------------------------------------------------------------
STIM_COLORS = ('#0072B2', '#D55E00', '#009E73')   # A, B, C

BIN         = 0.010      # s, PSTH bin width
SMOOTH_SD   = 0.010      # s, sd of the Gaussian smoothing kernel for PSTHs
VOLT_WIDTH  = 0.15       # s, width of the voltage window shown initially
VOLT_CENTRE = 0.055      # s, its centre relative to stimulus onset
VOLT_CLIM   = 55.0       # uV, colour limits of the voltage image

HIT_TOL = 0.20           # fraction of axes height within which a trace counts
                         # as clicked

# ----------------------------------------------------------------------
# Normalisation and aggregation options
#
# There are two places you can normalise, and they are not equivalent:
#   * per trial, BEFORE averaging trials  -- each trial is divided/centred by
#     its OWN pre-stimulus baseline, which is estimated from very few spikes
#     and is therefore extremely noisy
#   * per neuron, BEFORE averaging neurons -- each neuron is divided/centred
#     by a baseline pooled over all of its trials, which is far better
#     estimated
# The point of the selectors is to let you watch the population figure change
# shape depending on which you pick.
# ----------------------------------------------------------------------
TRIAL_NORMS  = ('none', 'subtract baseline', 'divide by baseline',
                'baseline z-score')
NEURON_NORMS = ('none', 'subtract baseline', 'divide by baseline',
                'peak = 1', 'baseline z-score')
STATS        = ('mean', 'median')

RATE_FLOOR = 1.0     # floor on any baseline used as a divisor (spikes/s)
SD_FLOOR   = 0.5     # floor on any baseline SD used as a divisor (spikes/s)
N_BOOT     = 200     # bootstrap resamples for the standard error of a median
BOOT_SEED  = 11      # fixed, so the error bands are reproducible


def _value_label(trial_norm, neuron_norm):
    """Axis label for whatever the numbers mean after normalisation."""
    last = neuron_norm if neuron_norm != 'none' else trial_norm
    return {
        'none':               'firing rate (spikes/s)',
        'subtract baseline':  '$\\Delta$ firing rate (spikes/s)',
        'divide by baseline': 'rate / baseline',
        'peak = 1':           'normalized rate (peak = 1)',
        'baseline z-score':   'baseline z-score (SD)',
    }[last]


def apply_style():
    plt.rcParams['font.family']       = 'Arial'
    plt.rcParams['font.sans-serif']   = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.spines.top']   = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['pdf.fonttype']      = 42
    plt.rcParams['ps.fonttype']       = 42
    plt.rcParams['figure.facecolor']  = 'w'


def gauss_smooth(x, sd_bins):
    """Gaussian smoothing along the last axis, edge-padded."""
    if sd_bins <= 0:
        return np.asarray(x, dtype=float)
    half = int(np.ceil(4 * sd_bins))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sd_bins) ** 2)
    k /= k.sum()
    x = np.asarray(x, dtype=float)
    xp = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(half, half)], mode='edge')
    flat = xp.reshape(-1, xp.shape[-1])
    out = np.empty((flat.shape[0], x.shape[-1]))
    for i in range(flat.shape[0]):
        out[i] = np.convolve(flat[i], k, mode='valid')
    return out.reshape(x.shape)


def neuron_colors(n, seed=7):
    """Distinct colour per neuron; shuffled so neighbouring depths differ."""
    base = plt.get_cmap('gist_rainbow')(np.linspace(0.0, 0.92, n))
    return base[np.random.default_rng(seed).permutation(n)]


def placeholder(ax, text):
    """Blank an axes and leave an instruction in the middle of it."""
    ax.clear()
    ax.set_axis_off()
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center',
            fontsize=11, color='0.55', style='italic')


# ----------------------------------------------------------------------
class Session:
    """The recording: spikes, trials, and lazily simulated raw voltage."""

    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.spike_neuron = d['spike_neuron']
        self.spike_time   = d['spike_time']
        self.onset        = d['trial_onset']
        self.stim         = d['trial_stim'].astype(int)
        self.depth        = d['neuron_depth']
        self.wf_amp       = d['neuron_wf_amp']
        self.stim_names   = [str(s) for s in d['stim_names']]
        self.pre          = float(d['pre'])
        self.post         = float(d['post'])
        self.n_channels   = int(d['n_channels'])
        self.ch_pitch     = float(d['ch_pitch'])
        self.fs_volt      = float(d['fs_volt'])

        self.n_neurons = int(self.spike_neuron.max()) + 1
        self.n_trials  = self.onset.size
        self.n_stim    = len(self.stim_names)
        self.trials_of = [np.nonzero(self.stim == s)[0]
                          for s in range(self.n_stim)]

        # Per-neuron spike times, so trial extraction is a pair of searchsorteds.
        order = np.argsort(self.spike_neuron, kind='stable')
        sn, st = self.spike_neuron[order], self.spike_time[order]
        bnd = np.searchsorted(sn, np.arange(self.n_neurons + 1))
        self.spikes = [np.sort(st[bnd[i]:bnd[i + 1]])
                       for i in range(self.n_neurons)]

        self.by_depth = np.argsort(self.depth)
        self.colors = neuron_colors(self.n_neurons)

        self._bin_spikes()
        self.configure()
        self._volt_cache = {}

    # -- spike extraction ------------------------------------------------
    def trial_spikes(self, neuron, trial):
        """Spike times of one neuron on one trial, relative to onset (s)."""
        t = self.spikes[neuron]
        t0 = self.onset[trial] - self.pre
        t1 = self.onset[trial] + self.post
        a, b = np.searchsorted(t, (t0, t1))
        return t[a:b] - self.onset[trial]

    # -- PSTHs -----------------------------------------------------------
    def _bin_spikes(self):
        edges = np.arange(-self.pre, self.post + BIN / 2, BIN)
        self.edges = edges
        self.t = 0.5 * (edges[:-1] + edges[1:])

        counts = np.zeros((self.n_neurons, self.n_trials, self.t.size))
        for i in range(self.n_neurons):
            for j in range(self.n_trials):
                counts[i, j] = np.histogram(self.trial_spikes(i, j), edges)[0]

        # rates[neuron, trial, bin], spikes/s, smoothed for display
        self.rates = gauss_smooth(counts / BIN, SMOOTH_SD / BIN)
        self.base = self.t < 0            # pre-stimulus bins

    # -- normalisation and aggregation -----------------------------------
    def configure(self, trial_norm='none', neuron_norm='none', stat='mean'):
        """
        Rebuild self.psth under a choice of normalisation and statistic.

        Both normalisations are applied; 'none'/'none'/'mean' reproduces the
        plain average. The neuron-level step is deliberately kept affine
        (offset and scale per neuron) so that level 2 can apply the identical
        transform to individual trials and stay consistent with level 1.
        """
        if trial_norm not in TRIAL_NORMS:
            raise ValueError(f'trial_norm must be one of {TRIAL_NORMS}')
        if neuron_norm not in NEURON_NORMS:
            raise ValueError(f'neuron_norm must be one of {NEURON_NORMS}')
        if stat not in STATS:
            raise ValueError(f'stat must be one of {STATS}')
        self.trial_norm, self.neuron_norm, self.stat = (
            trial_norm, neuron_norm, stat)

        R, base = self.rates, self.base
        b_mean = R[:, :, base].mean(axis=2)                 # [neuron, trial]
        self.n_floored = 0

        if trial_norm == 'none':
            Rn = R
        elif trial_norm == 'subtract baseline':
            Rn = R - b_mean[:, :, None]
        elif trial_norm == 'divide by baseline':
            self.n_floored = int((b_mean < RATE_FLOOR).sum())
            Rn = R / np.maximum(b_mean, RATE_FLOOR)[:, :, None]
        else:                                               # baseline z-score
            b_sd = R[:, :, base].std(axis=2, ddof=1)
            self.n_floored = int((b_sd < SD_FLOOR).sum())
            Rn = (R - b_mean[:, :, None]) / np.maximum(b_sd, SD_FLOOR)[:, :, None]
        self.rates_n = Rn

        psth = np.stack([self._central(Rn[:, tr, :], 1) for tr in self.trials_of],
                        axis=1)                             # [neuron, stim, bin]

        nb = psth[:, :, base].reshape(self.n_neurons, -1)   # pooled over stimuli
        off = np.zeros(self.n_neurons)
        sc = np.ones(self.n_neurons)
        self.n_floored_neurons = 0
        if neuron_norm == 'subtract baseline':
            off = nb.mean(axis=1)
        elif neuron_norm == 'divide by baseline':
            raw = nb.mean(axis=1)
            self.n_floored_neurons = int((raw < RATE_FLOOR).sum())
            sc = np.maximum(raw, RATE_FLOOR)
        elif neuron_norm == 'peak = 1':
            sc = np.maximum(np.abs(psth).reshape(self.n_neurons, -1).max(axis=1),
                            1e-6)
        elif neuron_norm == 'baseline z-score':
            off = nb.mean(axis=1)
            raw = nb.std(axis=1, ddof=1)
            self.n_floored_neurons = int((raw < SD_FLOOR).sum())
            sc = np.maximum(raw, SD_FLOOR)
        self._nn_off, self._nn_sc = off, sc

        self.psth = (psth - off[:, None, None]) / sc[:, None, None]

        self.value_label = _value_label(trial_norm, neuron_norm)
        self.stat_label = 'Average' if stat == 'mean' else 'Median'
        self.spread_label = 'SEM' if stat == 'mean' else 'bootstrap SE'
        return self

    def _central(self, x, axis):
        return (np.median(x, axis=axis) if self.stat == 'median'
                else np.mean(x, axis=axis))

    def _spread(self, x, axis):
        """Standard error of the chosen statistic; bootstrapped for a median."""
        n = x.shape[axis]
        if self.stat == 'mean':
            return x.std(axis=axis, ddof=1) / np.sqrt(n)
        rng = np.random.default_rng(BOOT_SEED)
        xm = np.moveaxis(x, axis, 0)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        return np.median(xm[idx], axis=1).std(axis=0, ddof=1)

    def population_stat(self, s):
        """Central tendency and its standard error across neurons, stim s."""
        x = self.psth[:, s, :]
        return self._central(x, 0), self._spread(x, 0)

    def neuron_trial_traces(self, i, s):
        """One neuron's single-trial traces for stim s, normalised as level 1."""
        return (self.rates_n[i][self.trials_of[s]] - self._nn_off[i]) / self._nn_sc[i]

    def neuron_stat(self, i, s):
        """Central tendency and its standard error across trials, neuron i."""
        x = self.neuron_trial_traces(i, s)
        return self._central(x, 0), self._spread(x, 0)

    # -- raw voltage -----------------------------------------------------
    def trial_voltage(self, trial):
        if trial in self._volt_cache:
            return self._volt_cache[trial]
        sp = [self.trial_spikes(i, trial) for i in range(self.n_neurons)]
        out = gd.simulate_trial_voltage(
            sp, self.depth, self.wf_amp, -self.pre, self.post,
            seed=gd.SEED_VOLT + int(trial), fs=self.fs_volt,
            n_channels=self.n_channels, ch_pitch=self.ch_pitch)
        if len(self._volt_cache) > 5:
            self._volt_cache.pop(next(iter(self._volt_cache)))
        self._volt_cache[trial] = out
        return out


# ----------------------------------------------------------------------
# Level 0 -- the summary everyone publishes
# ----------------------------------------------------------------------
def draw_summary(ax, S, title=None):
    ax.clear()
    ax.set_axis_on()
    lo = np.inf
    for s in range(S.n_stim):
        m, se = S.population_stat(s)
        lo = min(lo, (m - se).min())
        ax.fill_between(S.t, m - se, m + se, color=STIM_COLORS[s],
                        alpha=0.5, linewidth=0)
        ax.plot(S.t, m, color=STIM_COLORS[s], lw=2,
                label=f'stim {S.stim_names[s]}')
    ax.axvline(0, color='0.6', lw=0.8, zorder=0)
    if lo < 0:
        ax.axhline(0, color='0.8', lw=0.8, zorder=0)
    ax.set_xlabel('time from stimulus onset (s)')
    ax.set_ylabel(S.value_label)
    ax.set_xlim(S.t[0], S.t[-1])
    ax.set_title(title if title is not None else
                 f'{S.stat_label} across {S.n_neurons} neurons '
                 f'(shading: $\\pm${S.spread_label} across neurons)')
    ax.legend(frameon=False, loc='upper right', fontsize=9)


def hit_summary(ax, S, xdata, ydata):
    """Which trace was clicked, or None if the click was nowhere near one."""
    span = np.diff(ax.get_ylim())[0]
    dist = [abs(np.interp(xdata, S.t, S.population_stat(s)[0]) - ydata)
            for s in range(S.n_stim)]
    s = int(np.argmin(dist))
    return s if dist[s] <= HIT_TOL * span else None


# ----------------------------------------------------------------------
# Level 1 -- neurons x time for one stimulus
# ----------------------------------------------------------------------
def draw_neuron_matrix(ax, S, s, cax=None):
    ax.clear()
    ax.set_axis_on()
    M = S.psth[:, s, :]
    # Normalisation can send values below zero; a sequential map would then
    # hide the sign, so switch to a symmetric diverging one.
    if M.min() < -1e-9:
        lim = np.percentile(np.abs(M), 99.5)
        kw = dict(cmap='RdBu_r', vmin=-lim, vmax=lim)
    else:
        kw = dict(cmap='magma', vmin=0, vmax=np.percentile(M, 99.5))
    im = ax.imshow(M, aspect='auto', origin='upper',
                   extent=[S.t[0], S.t[-1], S.n_neurons - 0.5, -0.5],
                   interpolation='nearest', **kw)
    ax.axvline(0, color='w', lw=0.8, alpha=0.6)
    ax.set_xlabel('time from stimulus onset (s)')
    ax.set_ylabel('neuron #')
    ax.set_title(f'Stim {S.stim_names[s]}: every neuron behind that average\n'
                 f'({S.n_neurons} neurons, {len(S.trials_of[s])} trials each)',
                 color=STIM_COLORS[s])
    if cax is not None:
        cax.clear()
        cax.set_axis_on()
        cb = ax.figure.colorbar(im, cax=cax)
        cb.set_label(S.value_label)
    return im


def hit_neuron_matrix(S, ydata):
    i = int(round(ydata))
    return i if 0 <= i < S.n_neurons else None


# ----------------------------------------------------------------------
# Level 2 -- one neuron: all trials, all stimuli
# ----------------------------------------------------------------------
def draw_neuron(ax_raster, ax_psth, S, neuron):
    """Returns row_trial: the trial index shown on each raster row."""
    ax_raster.clear(); ax_raster.set_axis_on()
    ax_psth.clear();   ax_psth.set_axis_on()

    rows, colors, row_trial, boundaries = [], [], [], []
    for s in range(S.n_stim):
        for tr in S.trials_of[s]:
            rows.append(S.trial_spikes(neuron, tr))
            colors.append(STIM_COLORS[s])
            row_trial.append(tr)
        boundaries.append(len(rows))
    row_trial = np.array(row_trial)

    ax_raster.eventplot(rows, colors=colors,
                        lineoffsets=np.arange(len(rows)),
                        linelengths=0.85, linewidths=0.9)
    for b in boundaries[:-1]:
        ax_raster.axhline(b - 0.5, color='0.7', lw=0.8)
    for s in range(S.n_stim):
        lo = 0 if s == 0 else boundaries[s - 1]
        ax_raster.text(S.post * 1.01, 0.5 * (lo + boundaries[s] - 1),
                       S.stim_names[s], color=STIM_COLORS[s], fontsize=11,
                       fontweight='bold', va='center', ha='left')
    ax_raster.axvline(0, color='0.6', lw=0.8, zorder=0)
    ax_raster.set_xlim(-S.pre, S.post)
    ax_raster.set_ylim(len(rows) - 0.5, -0.5)
    ax_raster.set_xlabel('time from stimulus onset (s)')
    ax_raster.set_ylabel('trial (grouped by stimulus)')
    ax_raster.set_title(f'Neuron {neuron}: all {S.n_trials} trials')

    lo = np.inf
    for s in range(S.n_stim):
        m, se = S.neuron_stat(neuron, s)
        lo = min(lo, (m - se).min())
        ax_psth.fill_between(S.t, m - se, m + se, color=STIM_COLORS[s],
                             alpha=0.5, linewidth=0)
        ax_psth.plot(S.t, m, color=STIM_COLORS[s], lw=2,
                     label=f'stim {S.stim_names[s]}')
    ax_psth.axvline(0, color='0.6', lw=0.8, zorder=0)
    if lo < 0:
        ax_psth.axhline(0, color='0.8', lw=0.8, zorder=0)
    ax_psth.set_xlim(-S.pre, S.post)
    ax_psth.set_xlabel('time from stimulus onset (s)')
    ax_psth.set_ylabel(S.value_label)
    ax_psth.set_title(f'Neuron {neuron} {S.stat} PSTHs\n'
                      f'(shading: $\\pm${S.spread_label} across trials)')
    ax_psth.legend(frameon=False, loc='upper right', fontsize=9)
    return row_trial


def hit_neuron(row_trial, ydata):
    row = int(round(ydata))
    return int(row_trial[row]) if 0 <= row < len(row_trial) else None


# ----------------------------------------------------------------------
# Level 3 -- one trial: all neurons, and the raw voltage
# ----------------------------------------------------------------------
def draw_trial(ax_raster, ax_volt, S, trial, highlight_neuron):
    """
    Returns (band, dots): the Rectangle marking the voltage window on the
    raster, and the list of scatter artists so the caller can toggle them.
    """
    ax_raster.clear(); ax_raster.set_axis_on()
    ax_volt.clear();   ax_volt.set_axis_on()

    s = S.stim[trial]
    volt, tv, ch_depth = S.trial_voltage(trial)
    dlo = ch_depth[0] - S.ch_pitch / 2
    dhi = ch_depth[-1] + S.ch_pitch / 2

    spikes = [S.trial_spikes(i, trial) for i in range(S.n_neurons)]
    ax_raster.eventplot([spikes[i] for i in S.by_depth],
                        colors=[S.colors[i] for i in S.by_depth],
                        lineoffsets=[S.depth[i] for i in S.by_depth],
                        linelengths=13.0, linewidths=1.2)
    ax_raster.axvline(0, color='0.6', lw=0.8, zorder=0)
    ax_raster.axhline(S.depth[highlight_neuron], color='0.85', lw=8, zorder=-1)
    ax_raster.set_xlim(-S.pre, S.post)
    ax_raster.set_ylim(dlo, dhi)
    ax_raster.set_xlabel('time from stimulus onset (s)')
    ax_raster.set_ylabel('depth on probe (µm)')
    ax_raster.set_title(f'Trial {trial} (stim {S.stim_names[s]}): '
                        f'all {S.n_neurons} neurons\n'
                        f'grey band = neuron {highlight_neuron}',
                        color=STIM_COLORS[s])

    band = Rectangle((0, 0), 0, 1, transform=ax_raster.get_xaxis_transform(),
                     color='0.5', alpha=0.18, lw=0, zorder=-2)
    ax_raster.add_patch(band)

    ax_volt.imshow(volt, aspect='auto', origin='lower', cmap='gray',
                   extent=[tv[0], tv[-1], dlo, dhi],
                   vmin=-VOLT_CLIM, vmax=VOLT_CLIM,
                   interpolation='antialiased')
    dots = []
    for i in range(S.n_neurons):
        if spikes[i].size:
            dots.append(ax_volt.scatter(
                spikes[i], np.full(spikes[i].size, S.depth[i]),
                s=22, color=S.colors[i], zorder=3,
                edgecolor='w', linewidth=0.5))
    ax_volt.set_ylim(dlo, dhi)
    ax_volt.set_xlim(VOLT_CENTRE - VOLT_WIDTH / 2,
                     VOLT_CENTRE + VOLT_WIDTH / 2)
    ax_volt.set_xlabel('time from stimulus onset (s)')
    ax_volt.set_ylabel('depth on probe (µm)')
    ax_volt.set_title(f'Raw voltage, {S.n_channels} channels '
                      f'(±{VOLT_CLIM:.0f} µV grey scale)\n'
                      'dots = spikes, coloured by neuron')

    def sync_band(*_):
        x0, x1 = ax_volt.get_xlim()
        band.set_x(x0)
        band.set_width(x1 - x0)

    ax_volt.callbacks.connect('xlim_changed', sync_band)
    sync_band()
    return band, dots
