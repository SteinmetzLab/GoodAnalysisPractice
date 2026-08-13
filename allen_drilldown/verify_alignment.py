"""
Confirm the probe-clock alignment by spike-triggered average.

Uses align.json (from align_probe_clock.py): session time <-> probe sample via
piecewise-linear interpolation through the matched barcode edges. If that map
is right, each unit's STA on its own peak channel shows a sharp negative
trough at lag 0. This is the check that decides whether the voltage panel is
showing the truth.

Writes verify_alignment.png.
"""
import json
import os

import h5py
import numpy as np
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

URL = ('https://allen-brain-observatory.s3.us-west-2.amazonaws.com'
       '/visual-coding-neuropixels/raw-data/732592105/733744649/spike_band.dat')
NWB = r'D:\temp\allen_drilldown\session_732592105.nwb'
CACHE = r'D:\temp\allen_drilldown'
HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_ID = 733744649
NCH = 384
UV_PER_BIT = 0.195
T0, DUR, PAD = 2000.0, 3.0, 0.01
WF_MS = 3.0

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['pdf.fonttype'] = 42

AL = json.load(open(os.path.join(HERE, 'align.json')))
KS = np.asarray(AL['knotSamples'], dtype=np.float64)
KT = np.asarray(AL['knotTimes'], dtype=np.float64)


def time_to_sample(t):
    """Session seconds -> probe sample index (piecewise linear on barcodes)."""
    return np.interp(t, KT, KS)


def fetch_cached(sample_start, n_samples, tag=''):
    path = os.path.join(CACHE, f'raw{tag}_{sample_start}_{n_samples}.npy')
    if os.path.exists(path):
        return np.load(path)
    b0, b1 = sample_start * NCH * 2, (sample_start + n_samples) * NCH * 2
    print(f'  ranged GET {(b1-b0)/1e6:.1f} MB ...', flush=True)
    r = requests.get(URL, headers={'Range': f'bytes={b0}-{b1-1}'}, timeout=1800)
    r.raise_for_status()
    a = np.frombuffer(r.content, dtype='<i2').reshape(-1, NCH).T.copy()
    np.save(path, a)
    return a


def main():
    print(f'alignment: offset {AL["B"]:+.6f} s, rate {AL["samplingRate"]:.6f} Hz, '
          f'{AL["nEdges"]} knots')
    s0 = int(time_to_sample(T0 - PAD))
    s1 = int(time_to_sample(T0 + DUR + PAD))
    V = fetch_cached(s0, s1 - s0, tag='b').astype(np.float32)
    V -= np.median(V, axis=0, keepdims=True)
    V *= UV_PER_BIT
    rms = V.std(axis=1)
    bad = rms > 3 * np.median(rms)
    print(f'block {V.shape}, median RMS {np.median(rms):.1f} uV, '
          f'{bad.sum()} bad channels excluded')

    with h5py.File(NWB, 'r') as f:
        el = f['general/extracellular_ephys/electrodes']
        el_id, el_probe = el['id'][:], el['probe_id'][:]
        el_local = el['local_index'][:]
        by_el = {int(i): k for k, i in enumerate(el_id)}
        u = f['units']
        peak_ch, u_id = u['peak_channel_id'][:], u['id'][:]
        ends = u['spike_times_index'][:]
        starts = np.concatenate(([0], ends[:-1]))
        st = f['units/spike_times']
        cand = []
        for k in range(len(u_id)):
            e = by_el.get(int(peak_ch[k]))
            if e is None or int(el_probe[e]) != PROBE_ID:
                continue
            t = st[starts[k]:ends[k]]
            sel = t[(t >= T0) & (t < T0 + DUR)]
            if sel.size >= 30:
                cand.append(dict(n=sel.size, local=int(el_local[e]),
                                 t=sel, uid=int(u_id[k])))
    cand.sort(key=lambda d: -d['n'])
    cand = cand[:20]

    half = int(round(WF_MS / 1000 * AL['samplingRate']))
    rows = []
    for d in cand:
        idx = np.round(time_to_sample(d['t']) - s0).astype(int)
        idx = idx[(idx > half) & (idx < V.shape[1] - half - 1)]
        acc = np.zeros((NCH, 2 * half + 1))
        for i in idx:
            acc += V[:, i - half:i + half + 1]
        sta = acc / idx.size
        # z-score each channel's STA by its own expected noise, and ignore
        # the channels that are simply broken
        z = sta / (rms[:, None] / np.sqrt(idx.size))
        z[bad] = 0
        ch = int(np.argmin(z.min(axis=1)))
        lag = int(np.argmin(z[ch])) - half
        rows.append(dict(d, sta=sta, ch=ch, lag=lag, z=float(z[ch].min()),
                         uv=float(sta[ch].min()), nsp=idx.size))

    print(f'\n{"unit":>10} {"nsp":>4} {"local_index":>12} {"best col":>9} '
          f'{"lag (ms)":>9} {"trough uV":>10} {"z":>8}')
    for r in rows:
        print(f'{r["uid"]:>10} {r["nsp"]:>4} {r["local"]:>12} {r["ch"]:>9} '
              f'{r["lag"]/AL["samplingRate"]*1000:>9.3f} {r["uv"]:>10.1f} '
              f'{r["z"]:>8.1f}')
    same = np.array([r['ch'] == r['local'] for r in rows])
    lags = np.array([r['lag'] for r in rows])
    print(f'\nbest column == local_index for {same.sum()}/{len(rows)} units')
    print(f'lag: median {np.median(lags):.1f} samples '
          f'({np.median(lags)/AL["samplingRate"]*1000:+.3f} ms), '
          f'range [{lags.min()}, {lags.max()}]')
    print(f'trough z: median {np.median([r["z"] for r in rows]):.1f}')

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
    t_ms = (np.arange(2 * half + 1) - half) / AL['samplingRate'] * 1000
    for r in rows[:8]:
        axs[0].plot(t_ms, r['sta'][r['ch']], lw=1.2,
                    label=f'u{r["uid"]} ch{r["ch"]}')
    axs[0].axvline(0, color='#D55E00', lw=1)
    axs[0].set_xlabel('lag from spike time (ms)')
    axs[0].set_ylabel('spike-triggered average (µV)')
    axs[0].set_title('STA on each unit\'s peak channel')
    axs[0].legend(fontsize=6, frameon=False, ncol=2)

    r = rows[0]
    lo, hi = max(0, r['ch'] - 18), min(NCH, r['ch'] + 19)
    v = np.abs(r['sta'][lo:hi]).max()
    im = axs[1].imshow(r['sta'][lo:hi], aspect='auto', cmap='RdBu_r',
                       vmin=-v, vmax=v, extent=[t_ms[0], t_ms[-1], hi, lo])
    axs[1].axvline(0, color='0.3', lw=0.8)
    axs[1].axhline(r['local'] + 0.5, color='0.2', lw=0.8, ls='--')
    axs[1].set_xlabel('lag (ms)'); axs[1].set_ylabel('file column')
    axs[1].set_title(f'unit {r["uid"]}: STA across columns\n'
                     f'dashed = local_index {r["local"]}')
    fig.colorbar(im, ax=axs[1]).set_label('µV')

    axs[2].scatter([r['local'] for r in rows], [r['ch'] for r in rows],
                   c='#0072B2', zorder=3)
    axs[2].plot([0, NCH], [0, NCH], color='0.7', lw=1, ls='--')
    axs[2].set_xlabel('electrodes.local_index')
    axs[2].set_ylabel('best column in spike_band.dat')
    axs[2].set_title('Channel map: identity?')
    # Kilosort timestamps a spike at its template window start, so the
    # extracellular trough lands a constant ~0.77 ms later in the raw file.
    # Record it so the dots sit on the waveforms rather than beside them.
    AL['spikeLagSamples'] = int(np.median(lags))
    AL['spikeLagMs'] = float(np.median(lags) / AL['samplingRate'] * 1000)
    AL['staTroughZMedian'] = float(np.median([r['z'] for r in rows]))
    AL['channelMapIsIdentity'] = bool(same.sum() >= 0.6 * len(rows))
    with open(os.path.join(HERE, 'align.json'), 'w') as fh:
        json.dump(AL, fh, indent=1)
    print(f'updated align.json: spikeLagSamples={AL["spikeLagSamples"]}')

    fig.tight_layout()
    out = os.path.join(HERE, 'verify_alignment.png')
    fig.savefig(out, dpi=110)
    print('wrote', out)


if __name__ == '__main__':
    main()
