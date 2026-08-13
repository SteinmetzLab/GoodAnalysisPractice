"""
Check web/volt.bin the same way verify_alignment.py checked the raw file:
does each unit's spike-triggered average show a trough on its own channel?

Runs the test twice -- once with exact NWB spike times and the exact
sample map, once the way the web page does it (times relative to onset,
uint16-quantised, constant sample rate) -- so a mismatch localises the bug.

Writes check_voltage.png.
"""
import json
import os
import struct

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NWB = r'D:\temp\allen_drilldown\session_732592105.nwb'
PROBE_ID = 733744649

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

AL = json.load(open(os.path.join(HERE, 'align.json')))
KS = np.asarray(AL['knotSamples'], float)
KT = np.asarray(AL['knotTimes'], float)


def load_volt():
    raw = open(os.path.join(HERE, 'web', 'volt.bin'), 'rb').read()
    (hlen,) = struct.unpack('<I', raw[:4])
    h = json.loads(raw[4:4 + hlen])
    return h, raw


def block(h, raw, j):
    off = h['offsets'][str(j)]
    a = np.frombuffer(raw, dtype=np.int8, count=h['blockBytes'], offset=off)
    return a.reshape(h['nChannels'], h['nSamples']).astype(np.float32) \
        * (h['uvFullScale'] / 127)


def main():
    h, raw = load_volt()
    fs, win, c0 = h['fs'], h['win'], h['firstColumn']
    print(f'volt.bin: {len(h["trials"])} trials, {h["nChannels"]} ch x '
          f'{h["nSamples"]} samples, win {win}, first column {c0}')
    print(f'  {h["note"]}')

    f = h5py.File(NWB, 'r')
    el = f['general/extracellular_ephys/electrodes']
    el_id, el_probe = el['id'][:], el['probe_id'][:]
    el_local, el_y = el['local_index'][:], el['probe_vertical_position'][:]
    el_loc = np.array([s.decode() if isinstance(s, bytes) else str(s)
                       for s in el['location'][:]])
    by_el = {int(i): k for k, i in enumerate(el_id)}
    u = f['units']
    u_id, peak_ch = u['id'][:], u['peak_channel_id'][:]
    quality = np.array([s.decode() if isinstance(s, bytes) else str(s)
                        for s in u['quality'][:]])
    ends = u['spike_times_index'][:]
    starts = np.concatenate(([0], ends[:-1]))
    st = f['units/spike_times']

    keep = []
    for k in range(len(u_id)):
        e = by_el.get(int(peak_ch[k]))
        if e is None or int(el_probe[e]) != PROBE_ID or el_loc[e] != STRUCT:
            continue
        if quality[k] != 'good':
            continue
        if not (u['amplitude_cutoff'][k] < 0.1 and u['presence_ratio'][k] > 0.95
                and u['isi_violations'][k] < 0.5):
            continue
        keep.append(k)
    keep = np.array(keep)
    e_idx = np.array([by_el[int(peak_ch[k])] for k in keep])
    depth = el_y[e_idx].astype(float)
    order = np.argsort(depth, kind='stable')
    keep, e_idx, depth = keep[order], e_idx[order], depth[order]
    ucol = el_local[e_idx].astype(int)

    dg = f['intervals/drifting_gratings_presentations']
    onset_all, ori_all = dg['start_time'][:], dg['orientation'][:]
    ok = np.isfinite(ori_all)
    onset = onset_all[ok]

    half = int(round(0.003 * fs))
    j = h['trials'][10]
    V = block(h, raw, j)
    base0 = np.interp(onset[j] + win[0], KT, KS)     # file sample of block[0]
    print(f'\ntrial {j}, onset {onset[j]:.4f} s, block starts at sample '
          f'{base0:.1f}')

    rows = []
    for n, k in enumerate(keep):
        t = st[starts[k]:ends[k]]
        sel = t[(t >= onset[j] + win[0] + 0.004)
                & (t < onset[j] + win[1] - 0.004)]
        if sel.size < 8:
            continue
        col = ucol[n] - c0
        if not (0 <= col < h['nChannels']):
            continue
        # (a) exact: map absolute spike time through the barcode interpolation
        ia = np.round(np.interp(sel, KT, KS) - base0).astype(int)
        # (b) as the page does it: relative time x constant rate
        ib = np.round((sel - onset[j] - win[0]) * fs).astype(int)
        out = {}
        for tag, idx in (('exact', ia), ('page', ib)):
            idx = idx[(idx > half) & (idx < h['nSamples'] - half)]
            if idx.size < 5:
                continue
            sta = np.mean([V[col, i - half:i + half + 1] for i in idx], axis=0)
            out[tag] = (sta, sta.min(), (np.argmin(sta) - half) / fs * 1000)
        if len(out) == 2:
            rows.append((n, sel.size, out))

    rows.sort(key=lambda r: r[2]['exact'][1])
    print(f'\n{"unit":>5} {"nsp":>4} | {"exact trough":>13} {"lag ms":>7} '
          f'| {"page trough":>12} {"lag ms":>7}')
    for n, cnt, o in rows[:12]:
        print(f'{n:>5} {cnt:>4} | {o["exact"][1]:>13.1f} {o["exact"][2]:>7.3f} '
              f'| {o["page"][1]:>12.1f} {o["page"][2]:>7.3f}')
    ex = np.array([o['exact'][2] for _, _, o in rows])
    pg = np.array([o['page'][2] for _, _, o in rows])
    print(f'\nexact lags: median {np.median(ex):+.3f} ms, '
          f'IQR {np.percentile(ex,25):+.3f}..{np.percentile(ex,75):+.3f}')
    print(f'page  lags: median {np.median(pg):+.3f} ms, '
          f'IQR {np.percentile(pg,25):+.3f}..{np.percentile(pg,75):+.3f}')

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    t_ms = (np.arange(2 * half + 1) - half) / fs * 1000
    for ax, tag in zip(axs, ('exact', 'page')):
        for n, cnt, o in rows[:8]:
            ax.plot(t_ms, o[tag][0], lw=1, label=f'u{n}')
        ax.axvline(0, color='#D55E00', lw=1)
        ax.set_xlabel('lag (ms)'); ax.set_title(f'{tag} mapping')
    axs[0].set_ylabel('STA within volt.bin (µV)')
    axs[0].legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'check_voltage.png'), dpi=110)
    print('wrote check_voltage.png')
    f.close()


STRUCT = 'VISp'

if __name__ == '__main__':
    main()
