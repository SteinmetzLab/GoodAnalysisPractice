"""
Read web/data.bin back and sanity-check it: does this probe actually show
orientation tuning, and do different units prefer different directions?

Writes check_tuning.png.
"""
import json
import os
import struct

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['pdf.fonttype'] = 42


def load(path=None):
    path = path or os.path.join(HERE, 'web', 'data.bin')
    raw = open(path, 'rb').read()
    (hlen,) = struct.unpack('<I', raw[:4])
    h = json.loads(raw[4:4 + hlen])
    arr = {}
    for name, dt in h['dtypes'].items():
        arr[name] = np.frombuffer(raw, dtype=np.dtype(dt),
                                  count=h['lengths'][name],
                                  offset=h['offsets'][name])
        assert h['offsets'][name] % 8 == 0, name
    return h, arr


def main():
    h, a = load()
    NU, NT, NC, NB = h['nUnits'], h['nTrials'], h['nCond'], h['nBins']
    PRE, POST, BIN = h['pre'], h['post'], h['bin']
    print(json.dumps({k: h[k] for k in
                      ('sessionId', 'probeName', 'structure', 'nUnits',
                       'nTrials', 'nCond', 'condNames', 'stimDuration')},
                     indent=1))

    spT, spIdx, cond = a['spT'], a['spIdx'], a['cond']
    scale = (PRE + POST) / h['quant']

    def spikes(i, j):
        k = i * NT + j
        return spT[spIdx[k]:spIdx[k + 1]] * scale - PRE

    # counts in the 0-2 s stimulus window, per unit per trial
    resp = np.zeros((NU, NT))
    base = np.zeros((NU, NT))
    for i in range(NU):
        for j in range(NT):
            t = spikes(i, j)
            resp[i, j] = np.count_nonzero((t >= 0) & (t < h['stimDuration']))
            base[i, j] = np.count_nonzero(t < 0)
    resp /= h['stimDuration']
    base /= PRE

    tuning = np.stack([resp[:, cond == c].mean(axis=1) for c in range(NC)], 1)
    base_m = base.mean(axis=1)
    print(f'\nmean baseline rate {base_m.mean():.2f} spikes/s, '
          f'mean evoked {tuning.mean():.2f} spikes/s')

    # Orientation selectivity index on the ORIENTATION (mod 180) axis, and
    # direction selectivity on the full circle.
    ang = np.deg2rad(np.array(h['condValues']))
    r = tuning - base_m[:, None]
    r = np.clip(r, 0, None)
    tot = r.sum(axis=1) + 1e-9
    osi = np.abs((r * np.exp(2j * ang)).sum(axis=1)) / tot
    dsi = np.abs((r * np.exp(1j * ang)).sum(axis=1)) / tot
    pref = np.array(h['condValues'])[np.argmax(tuning, axis=1)]

    print(f'OSI: median {np.median(osi):.2f}, '
          f'{np.count_nonzero(osi > 0.3)} of {NU} units > 0.3')
    print(f'DSI: median {np.median(dsi):.2f}, '
          f'{np.count_nonzero(dsi > 0.3)} of {NU} units > 0.3')
    print('preferred direction histogram:',
          dict(zip(h['condNames'], np.bincount(np.argmax(tuning, 1),
                                               minlength=NC).tolist())))

    # ---- figure
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.35)

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(h['condValues'], tuning.mean(axis=0), 'o-', color='0.2')
    ax.set_xlabel('direction (deg)')
    ax.set_ylabel('firing rate (spikes/s)')
    ax.set_title('Population mean tuning (flat = tuning averages out)')

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(osi, bins=20, color='#0072B2')
    ax.set_xlabel('OSI'); ax.set_ylabel('units')
    ax.set_title('Orientation selectivity')

    ax = fig.add_subplot(gs[0, 3])
    ax.hist(pref, bins=np.arange(-22.5, 360, 45), color='#D55E00')
    ax.set_xlabel('preferred direction (deg)'); ax.set_ylabel('units')
    ax.set_title('Preferred directions')

    best = np.argsort(-osi)[:8]
    for k, i in enumerate(best):
        ax = fig.add_subplot(gs[1 + k // 4, k % 4])
        ax.plot(h['condValues'], tuning[i], 'o-', color='#009E73')
        ax.axhline(base_m[i], color='0.7', ls='--', lw=1)
        ax.set_title(f'unit {i} (OSI {osi[i]:.2f}, pref {pref[i]:.0f}°)',
                     fontsize=9)
        ax.set_xlabel('direction (deg)')
        if k % 4 == 0:
            ax.set_ylabel('firing rate (spikes/s)')
    fig.suptitle(f'Session {h["sessionId"]} {h["probeName"]} {h["structure"]}: '
                 f'{NU} units, drifting gratings', y=0.99)
    out = os.path.join(HERE, 'check_tuning.png')
    fig.savefig(out, dpi=100, bbox_inches='tight')
    print('\nwrote', out)


if __name__ == '__main__':
    main()
