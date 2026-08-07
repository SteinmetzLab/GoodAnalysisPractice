"""
Synthetic dataset for the interactive drill-down demo.

A single "session": N_NEURONS neurons recorded on a 64-channel probe while
three stimuli (A, B, C) are presented in random order.

Generative model
----------------
* Each neuron has a baseline rate (lognormal, median ~5 spikes/s).
* Each neuron has a response amplitude to each stimulus, drawn from a Gaussian
  whose mean depends on the stimulus:  A > B ~ C.  Amplitudes can be negative
  (suppression); the resulting rate is clipped at zero.
* The response time course is a gamma-shaped kernel (fast rise, slower decay)
  that is essentially over within ~200 ms.
* Spikes are an inhomogeneous Poisson process with a 1.5 ms refractory period.
* Each neuron also has a depth on the probe and a waveform amplitude, which are
  used to synthesise raw voltage traces on demand (see simulate_trial_voltage).

Two things live here: the session generator (writes data/dataset.npz) and the
voltage simulator, which the viewer calls lazily because storing raw voltage
for every trial would be wasteful.

Run:  python generate_data.py
"""

import os
import numpy as np

# ----------------------------------------------------------------------
# Session parameters
# ----------------------------------------------------------------------
SEED              = 2026
N_NEURONS         = 50
N_TRIALS_PER_STIM = 40
STIM_NAMES        = ("A", "B", "C")

# Mean / sd (across neurons) of the peak response amplitude, in spikes/s.
# A is clearly larger than B and C -- this is the "result" the summary shows.
RESP_MEAN = (14.0, 6.0, 5.0)
RESP_SD   = (6.0, 5.0, 5.0)

BASE_RATE_MEDIAN = 5.0     # spikes/s
BASE_RATE_LOGSD  = 0.5

PRE     = 0.4              # s of trial window before stimulus onset
POST    = 0.8              # s of trial window after stimulus onset
ITI_MIN = 1.7              # s, minimum inter-trial interval (> PRE + POST)
ITI_JIT = 0.6              # s, uniform jitter added to the ITI

GAMMA_SHAPE = 2.2          # shape k of the gamma response kernel
GAMMA_SCALE = 0.035        # scale theta, s   -> peak at (k-1)*theta = 42 ms
KERNEL_DUR  = 0.30         # s, kernel truncated here

DT_GEN      = 0.001        # s, grid on which the rate function is evaluated
REFRACTORY  = 0.0015       # s

# ----------------------------------------------------------------------
# Probe / voltage parameters
# ----------------------------------------------------------------------
N_CHANNELS   = 64
CH_PITCH     = 20.0        # um between channels
DEPTH_MARGIN = 60.0        # um kept clear at each end of the probe
WF_AMP_RANGE = (40.0, 160.0)   # uV, trough amplitude of each neuron

FS_VOLT      = 20000.0     # Hz
NOISE_SD_UV  = 12.0        # uV, white noise

# Waveform = negative 2-D Gaussian, followed by a wider-in-time positive one.
WF_TROUGH_ST   = 0.00022   # s, temporal sd of the negative lobe
WF_PEAK_ST     = 0.00055   # s, temporal sd of the positive lobe (wider)
WF_PEAK_DELAY  = 0.00065   # s, positive lobe follows the trough by this much
WF_PEAK_FRAC   = 0.35      # positive lobe amplitude, as a fraction of trough
WF_SIGMA_D_TR  = 45.0      # um, spatial sd of the negative lobe
WF_SIGMA_D_PK  = 55.0      # um, spatial sd of the positive lobe

SEED_VOLT = 90000          # per-trial voltage seeds are SEED_VOLT + trial index


# ----------------------------------------------------------------------
# Response kernel
# ----------------------------------------------------------------------
def response_kernel(t):
    """Gamma-shaped response, normalised to a peak of 1. t in seconds."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = (t > 0) & (t < KERNEL_DUR)
    tt = t[m]
    out[m] = tt ** (GAMMA_SHAPE - 1) * np.exp(-tt / GAMMA_SCALE)
    mode = (GAMMA_SHAPE - 1) * GAMMA_SCALE
    peak = mode ** (GAMMA_SHAPE - 1) * np.exp(-mode / GAMMA_SCALE)
    return out / peak


# ----------------------------------------------------------------------
# Session generation
# ----------------------------------------------------------------------
def make_session(seed=SEED):
    rng = np.random.default_rng(seed)

    n_stim = len(STIM_NAMES)
    n_trials = N_TRIALS_PER_STIM * n_stim

    # --- trial schedule -------------------------------------------------
    stim_id = np.repeat(np.arange(n_stim), N_TRIALS_PER_STIM)
    rng.shuffle(stim_id)
    itis = ITI_MIN + rng.uniform(0.0, ITI_JIT, size=n_trials)
    onset = 2.0 + np.cumsum(itis) - itis[0]
    session_dur = onset[-1] + POST + 2.0

    # --- per-neuron properties -----------------------------------------
    baseline = np.exp(rng.normal(np.log(BASE_RATE_MEDIAN), BASE_RATE_LOGSD,
                                 size=N_NEURONS))
    resp_amp = np.empty((N_NEURONS, n_stim))
    for s in range(n_stim):
        resp_amp[:, s] = rng.normal(RESP_MEAN[s], RESP_SD[s], size=N_NEURONS)

    depth = rng.uniform(DEPTH_MARGIN,
                        (N_CHANNELS - 1) * CH_PITCH - DEPTH_MARGIN,
                        size=N_NEURONS)
    wf_amp = rng.uniform(*WF_AMP_RANGE, size=N_NEURONS)

    # --- spike trains ---------------------------------------------------
    n_bins = int(np.ceil(session_dur / DT_GEN))
    grid = np.arange(n_bins) * DT_GEN

    # Precompute the kernel once on the generation grid, then add a shifted
    # copy at each trial onset.
    k_len = int(np.ceil(KERNEL_DUR / DT_GEN))
    kern = response_kernel(np.arange(k_len) * DT_GEN)

    spike_neuron, spike_time = [], []
    for i in range(N_NEURONS):
        rate = np.full(n_bins, baseline[i])
        for tr in range(n_trials):
            b0 = int(round(onset[tr] / DT_GEN))
            b1 = min(b0 + k_len, n_bins)
            rate[b0:b1] += resp_amp[i, stim_id[tr]] * kern[:b1 - b0]
        np.clip(rate, 0.0, None, out=rate)

        counts = rng.poisson(rate * DT_GEN)
        idx = np.nonzero(counts)[0]
        # At these rates counts are ~always 0 or 1; repeat handles the rest.
        idx = np.repeat(idx, counts[idx])
        t = grid[idx] + rng.uniform(0.0, DT_GEN, size=idx.size)
        t.sort()
        if t.size:
            keep = np.concatenate(([True], np.diff(t) > REFRACTORY))
            t = t[keep]
        spike_neuron.append(np.full(t.size, i, dtype=np.int32))
        spike_time.append(t)

    spike_neuron = np.concatenate(spike_neuron)
    spike_time = np.concatenate(spike_time)
    order = np.argsort(spike_time, kind="stable")

    return dict(
        spike_neuron=spike_neuron[order],
        spike_time=spike_time[order],
        trial_onset=onset,
        trial_stim=stim_id.astype(np.int8),
        neuron_depth=depth,
        neuron_wf_amp=wf_amp,
        neuron_baseline=baseline,
        resp_amp=resp_amp,
        stim_names=np.array(STIM_NAMES),
        pre=PRE, post=POST,
        n_channels=N_CHANNELS, ch_pitch=CH_PITCH,
        fs_volt=FS_VOLT, session_dur=session_dur,
    )


# ----------------------------------------------------------------------
# Raw voltage, simulated on demand for one trial
# ----------------------------------------------------------------------
def simulate_trial_voltage(spikes_by_neuron, depth, wf_amp, t0, t1, seed,
                           fs=FS_VOLT, n_channels=N_CHANNELS,
                           ch_pitch=CH_PITCH, noise_sd=NOISE_SD_UV):
    """
    White noise plus one waveform per spike.

    spikes_by_neuron : list of arrays of spike times (s, relative to onset),
                       one entry per neuron
    returns (volt [n_channels x n_samples] in uV, t [s], ch_depth [um])
    """
    rng = np.random.default_rng(seed)
    n_samp = int(round((t1 - t0) * fs))
    t = t0 + np.arange(n_samp) / fs
    ch_depth = np.arange(n_channels) * ch_pitch

    volt = rng.normal(0.0, noise_sd, size=(n_channels, n_samp)).astype(np.float32)

    nhw = int(round(0.003 * fs))                 # +/- 3 ms of waveform support
    tw = np.arange(-nhw, nhw + 1) / fs
    trough_t = np.exp(-tw ** 2 / (2 * WF_TROUGH_ST ** 2))
    peak_t = np.exp(-(tw - WF_PEAK_DELAY) ** 2 / (2 * WF_PEAK_ST ** 2))

    for i, st in enumerate(spikes_by_neuron):
        if len(st) == 0:
            continue
        ci = np.nonzero(np.abs(ch_depth - depth[i]) <= 3 * WF_SIGMA_D_PK)[0]
        if ci.size == 0:
            continue
        dd = ch_depth[ci] - depth[i]
        sp_tr = np.exp(-dd ** 2 / (2 * WF_SIGMA_D_TR ** 2))[:, None]
        sp_pk = np.exp(-dd ** 2 / (2 * WF_SIGMA_D_PK ** 2))[:, None]
        patch = wf_amp[i] * (-sp_tr * trough_t[None, :]
                             + WF_PEAK_FRAC * sp_pk * peak_t[None, :])

        for ts in np.atleast_1d(st):
            c = int(round((ts - t0) * fs))
            a, b = c - nhw, c + nhw + 1
            pa, pb = 0, patch.shape[1]
            if a < 0:
                pa, a = -a, 0
            if b > n_samp:
                pb, b = pb - (b - n_samp), n_samp
            if b <= a:
                continue
            volt[ci[0]:ci[-1] + 1, a:b] += patch[:, pa:pb]

    return volt, t, ch_depth


# ----------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)
    out = os.path.join(data_dir, "dataset.npz")

    d = make_session()
    np.savez_compressed(out, **d)

    n_sp = d["spike_time"].size
    print(f"wrote {out}")
    print(f"  {N_NEURONS} neurons, {d['trial_onset'].size} trials "
          f"({N_TRIALS_PER_STIM} per stimulus), {d['session_dur']:.0f} s")
    print(f"  {n_sp} spikes, mean rate "
          f"{n_sp / N_NEURONS / d['session_dur']:.1f} spikes/s per neuron")
    for s, name in enumerate(STIM_NAMES):
        print(f"  stim {name}: mean response amplitude "
              f"{d['resp_amp'][:, s].mean():.1f} spikes/s")


if __name__ == "__main__":
    main()
