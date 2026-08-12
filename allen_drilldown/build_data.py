"""
Extract one probe's V1 units and the drifting-grating trials from an Allen
Institute Visual Coding - Neuropixels session, and write the binary payload
the web drill-down loads.

Session 732592105, probe 733744649 (probeC) -- the brain_observatory_1.1
session/probe pair with the most good V1 units (110 under Allen's default QC).

The stimulus is `drifting_gratings`: 8 directions x 5 temporal frequencies x
15 repeats, plus blank sweeps. We pool across temporal frequency and use
**direction** as the condition, so each of the 8 conditions has 75 trials.

Output: web/data.bin -- a uint32 JSON-header length, the JSON header, then
raw little-endian arrays back to back. Only spikes inside a trial window are
included, stored relative to the window start as uint16, which is all the page
ever displays and much kinder than absolute session time.

Run:  <venv>/python build_data.py
"""

import json
import os
import struct

import h5py
import numpy as np

NWB = r'D:\temp\allen_drilldown\session_732592105.nwb'
SESSION_ID = 732592105
PROBE_ID = 733744649
PROBE_NAME = 'probeC'
STRUCTURE = 'VISp'

PRE, POST = 0.5, 2.5          # s around stimulus onset
BIN = 0.025                   # s, PSTH bin width used by the page
N_BINS = int(round((PRE + POST) / BIN))

# Allen's default unit QC
QC = dict(amplitude_cutoff=0.1, presence_ratio=0.95, isi_violations=0.5)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'web')


def ragged(f, base):
    """Read an NWB ragged column: values plus its VectorIndex end-offsets."""
    vals = f[f'units/{base}'][:]
    ends = f[f'units/{base}_index'][:]
    starts = np.concatenate(([0], ends[:-1]))
    return vals, starts, ends


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    f = h5py.File(NWB, 'r')

    # ---------------------------------------------------------------- units
    el = f['general/extracellular_ephys/electrodes']
    el_id = el['id'][:]
    el_probe = el['probe_id'][:]
    el_loc = np.array([s.decode() if isinstance(s, bytes) else str(s)
                       for s in el['location'][:]])
    el_y = el['probe_vertical_position'][:].astype(float)
    el_x = el['probe_horizontal_position'][:].astype(float)
    by_el = {int(i): k for k, i in enumerate(el_id)}

    u = f['units']
    u_id = u['id'][:]
    peak_ch = u['peak_channel_id'][:]
    quality = np.array([s.decode() if isinstance(s, bytes) else str(s)
                        for s in u['quality'][:]])
    amp_cut = u['amplitude_cutoff'][:]
    pres = u['presence_ratio'][:]
    isi_v = u['isi_violations'][:]
    fr = u['firing_rate'][:]
    snr = u['snr'][:]

    keep = []
    for k in range(len(u_id)):
        e = by_el.get(int(peak_ch[k]))
        if e is None or int(el_probe[e]) != PROBE_ID or el_loc[e] != STRUCTURE:
            continue
        if quality[k] != 'good':
            continue
        if not (amp_cut[k] < QC['amplitude_cutoff']
                and pres[k] > QC['presence_ratio']
                and isi_v[k] < QC['isi_violations']):
            continue
        keep.append(k)
    keep = np.array(keep)
    print(f'{len(keep)} units on {PROBE_NAME} in {STRUCTURE} passing QC')

    e_idx = np.array([by_el[int(peak_ch[k])] for k in keep])
    depth = el_y[e_idx]
    xpos = el_x[e_idx]
    # Sort units by depth so level 1 and level 3 share a sensible order.
    order = np.argsort(depth, kind='stable')
    keep, depth, xpos, e_idx = keep[order], depth[order], xpos[order], e_idx[order]
    n_units = len(keep)

    st_vals, st_start, st_end = ragged(f, 'spike_times')
    spikes = [st_vals[st_start[k]:st_end[k]] for k in keep]

    # -------------------------------------------------------------- trials
    dg = f['intervals/drifting_gratings_presentations']
    onset = dg['start_time'][:]
    ori = dg['orientation'][:]
    tf = dg['temporal_frequency'][:]
    ok = np.isfinite(ori)                       # drops the blank sweeps
    onset, ori, tf = onset[ok], ori[ok], tf[ok]
    ori_vals = np.unique(ori)
    print(f'{len(onset)} grating trials, {len(ori_vals)} directions '
          f'{ori_vals.astype(int).tolist()}, '
          f'temporal frequencies {np.unique(tf).tolist()}')
    cond = np.searchsorted(ori_vals, ori).astype(np.uint8)
    counts = np.bincount(cond)
    print('trials per direction:', counts.tolist())

    dur = float(np.median(dg['stop_time'][:] - dg['start_time'][:]))
    gap = float(np.median(np.diff(onset)))
    print(f'stimulus duration {dur:.2f} s, inter-onset {gap:.2f} s, '
          f'window [-{PRE}, {POST}] s')

    # ------------------------------------------- trial-windowed spike times
    n_trials = len(onset)
    times, bounds = [], [0]
    for i in range(n_units):
        t = spikes[i]
        lo = np.searchsorted(t, onset - PRE)
        hi = np.searchsorted(t, onset + POST)
        for j in range(n_trials):
            rel = t[lo[j]:hi[j]] - onset[j]
            times.append(rel)
            bounds.append(bounds[-1] + rel.size)
    times = np.concatenate(times) if times else np.zeros(0)
    bounds = np.asarray(bounds, dtype=np.uint32)
    print(f'{times.size} in-window spikes '
          f'({times.size / n_units / n_trials:.1f} per unit per trial)')

    # Quantise to uint16 over the window: 3 s / 65535 = 46 us, far finer than
    # the 25 ms display bin.
    q = np.clip(np.round((times + PRE) / (PRE + POST) * 65535), 0, 65535)
    q = q.astype(np.uint16)

    arrays = [
        ('spT',    q),
        ('spIdx',  bounds),
        ('depth',  depth.astype(np.float32)),
        ('xpos',   xpos.astype(np.float32)),
        ('unitId', u_id[keep].astype(np.uint32)),
        ('firing', fr[keep].astype(np.float32)),
        ('snr',    snr[keep].astype(np.float32)),
        ('cond',   cond),
        ('tf',     tf.astype(np.float32)),
        ('onset',  onset.astype(np.float32)),
    ]

    # Lay the arrays out with explicit byte offsets, each 8-byte aligned so
    # the browser can wrap them as typed-array views over the ArrayBuffer with
    # no copying (TypedArray views require a correctly aligned byteOffset).
    def align(n):
        return (n + 7) & ~7

    header = dict(
        sessionId=SESSION_ID, probeId=PROBE_ID, probeName=PROBE_NAME,
        structure=STRUCTURE,
        nUnits=n_units, nTrials=n_trials, nCond=len(ori_vals), nBins=N_BINS,
        condNames=[f'{int(o)}°' for o in ori_vals],
        condValues=ori_vals.tolist(),
        pre=PRE, post=POST, bin=BIN, quant=65535,
        stimDuration=round(dur, 3),
        qc=QC,
        dtypes={n: str(a.dtype) for n, a in arrays},
        lengths={n: int(a.size) for n, a in arrays},
        offsets={},
    )
    # The header holds the offsets, but padding it changes its own length, so
    # size it first with placeholders and then fill them in.
    def encode(h):
        return json.dumps(h, separators=(',', ':')).encode()

    # Size the header with 8-digit placeholder offsets so that substituting
    # the real ones cannot change its length (any file under 100 MB fits).
    for name, a in arrays:
        header['offsets'][name] = 99999999
    hlen = align(4 + len(encode(header))) - 4
    pos = align(4 + hlen)
    for name, a in arrays:
        assert pos < 100000000, 'payload too large for 8-digit offsets'
        header['offsets'][name] = pos
        pos = align(pos + a.nbytes)
    blob = encode(header)
    assert len(blob) <= hlen, (len(blob), hlen)
    blob += b' ' * (hlen - len(blob))

    out = os.path.join(OUT_DIR, 'data.bin')
    with open(out, 'wb') as fh:
        fh.write(struct.pack('<I', len(blob)))
        fh.write(blob)
        for name, a in arrays:
            assert fh.tell() == header['offsets'][name], name
            b = np.ascontiguousarray(a).tobytes()
            fh.write(b)
            fh.write(b'\0' * (align(len(b)) - len(b)))
    f.close()

    mb = os.path.getsize(out) / 1e6
    print(f'\nwrote {out}  ({mb:.2f} MB)')
    print(f'  {n_units} units x {n_trials} trials x {N_BINS} bins')
    print(f'  depth range {depth.min():.0f}-{depth.max():.0f} um on the probe')


if __name__ == '__main__':
    main()
