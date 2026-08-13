"""
Pre-extract raw-voltage snippets for a subset of trials and write web/volt.bin.

Allen's spike_band.dat is 217 GB per probe and their S3 bucket sends no CORS
header, so the browser can neither stream it nor read it cross-origin. But the
file is flat channel-interleaved int16, so a time window is contiguous bytes
and a ranged GET from Python fetches exactly what we want. We pull one window
per selected trial, crop to the channels spanning our V1 units, and host the
result next to the page.

Sample <-> session-time mapping comes from align.json (align_probe_clock.py,
confirmed by verify_alignment.py).

One snippet per (direction, temporal frequency) pair = 40 trials, so every
stimulus condition has a representative.

Run:  <venv>/python build_voltage.py
"""

import json
import os
import struct

import h5py
import numpy as np
import requests
from scipy.signal import butter, sosfiltfilt

URL = ('https://allen-brain-observatory.s3.us-west-2.amazonaws.com'
       '/visual-coding-neuropixels/raw-data/732592105/733744649/spike_band.dat')
NWB = r'D:\temp\allen_drilldown\session_732592105.nwb'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'web')
CACHE = r'D:\temp\allen_drilldown\snippets'

PROBE_ID = 733744649
NCH = 384
UV_PER_BIT = 0.195
STRUCTURE = 'VISp'

WIN = (-0.05, 0.15)        # s around stimulus onset
PAD = 0.05                 # s of extra signal each side, trimmed after filtering
HP_HZ = 300.0              # high-pass for display
CH_MARGIN = 8              # extra channels beyond the V1 unit span
CLIP_UV = 400.0            # int8 full scale after filtering
QC = dict(amplitude_cutoff=0.1, presence_ratio=0.95, isi_violations=0.5)

AL = json.load(open(os.path.join(HERE, 'align.json')))
KS = np.asarray(AL['knotSamples'], float)
KT = np.asarray(AL['knotTimes'], float)
FS = AL['samplingRate']


def time_to_sample(t):
    return np.interp(t, KT, KS)


def fetch(s0, n):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f'{s0}_{n}.npy')
    if os.path.exists(path):
        return np.load(path)
    b0, b1 = s0 * NCH * 2, (s0 + n) * NCH * 2
    r = requests.get(URL, headers={'Range': f'bytes={b0}-{b1-1}'}, timeout=1800)
    r.raise_for_status()
    assert r.status_code == 206, r.status_code
    a = np.frombuffer(r.content, dtype='<i2').reshape(-1, NCH).T.copy()
    np.save(path, a)
    return a


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
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
    keep = []
    for k in range(len(u_id)):
        e = by_el.get(int(peak_ch[k]))
        if e is None or int(el_probe[e]) != PROBE_ID or el_loc[e] != STRUCTURE:
            continue
        if quality[k] != 'good':
            continue
        if not (u['amplitude_cutoff'][k] < QC['amplitude_cutoff']
                and u['presence_ratio'][k] > QC['presence_ratio']
                and u['isi_violations'][k] < QC['isi_violations']):
            continue
        keep.append(k)
    keep = np.array(keep)
    e_idx = np.array([by_el[int(peak_ch[k])] for k in keep])
    depth = el_y[e_idx].astype(float)
    order = np.argsort(depth, kind='stable')
    keep, e_idx, depth = keep[order], e_idx[order], depth[order]
    unit_col = el_local[e_idx].astype(int)      # file column == local_index
    print(f'{len(keep)} V1 units, peak columns '
          f'{unit_col.min()}..{unit_col.max()}, depth {depth.min():.0f}'
          f'..{depth.max():.0f} um')

    c0 = max(0, unit_col.min() - CH_MARGIN)
    c1 = min(NCH - 1, unit_col.max() + CH_MARGIN)
    cols = np.arange(c0, c1 + 1)
    # depth of each kept column, for the image's y axis
    col_depth = np.full(cols.size, np.nan)
    for e in range(len(el_id)):
        if int(el_probe[e]) != PROBE_ID:
            continue
        li = int(el_local[e])
        if c0 <= li <= c1:
            col_depth[li - c0] = float(el_y[e])
    print(f'channels {c0}..{c1} ({cols.size}), depth '
          f'{np.nanmin(col_depth):.0f}..{np.nanmax(col_depth):.0f} um')

    ends = u['spike_times_index'][:]
    st_starts = np.concatenate(([0], ends[:-1]))
    st_all = f['units/spike_times']
    spikes_of = [st_all[st_starts[k]:ends[k]] for k in keep]

    dg = f['intervals/drifting_gratings_presentations']
    onset_all = dg['start_time'][:]
    ori_all, tf_all = dg['orientation'][:], dg['temporal_frequency'][:]
    ok = np.isfinite(ori_all)
    onset, ori, tf = onset_all[ok], ori_all[ok], tf_all[ok]
    ori_vals, tf_vals = np.unique(ori), np.unique(tf)

    # one representative trial per (direction, temporal frequency)
    chosen = []
    for oi, o in enumerate(ori_vals):
        for ti, t in enumerate(tf_vals):
            idx = np.nonzero((ori == o) & (tf == t))[0]
            if idx.size:
                chosen.append(int(idx[0]))
    chosen = sorted(chosen)
    print(f'{len(chosen)} trials selected '
          f'({len(ori_vals)} directions x {len(tf_vals)} temporal freqs)')

    n_samp = int(round((WIN[1] - WIN[0]) * FS))
    n_pad = int(round(PAD * FS))
    sos = butter(3, HP_HZ, btype='highpass', fs=FS, output='sos')
    per_mb = n_samp * cols.size / 1e6
    print(f'high-pass {HP_HZ:g} Hz, pad {PAD*1000:g} ms each side')
    print(f'window {WIN[0]}..{WIN[1]} s = {n_samp} samples; '
          f'{per_mb:.2f} MB per trial as int8, '
          f'{per_mb*len(chosen):.1f} MB total')

    blocks = []
    for n, j in enumerate(chosen):
        s0 = int(round(time_to_sample(onset[j] + WIN[0]))) - n_pad
        V = fetch(s0, n_samp + 2 * n_pad).astype(np.float32) * UV_PER_BIT
        # The AP band still carries a lot of low-frequency power. Averaging
        # kills it, but a single trial does not, and undreamt-of slow blobs
        # swamp the 1 ms waveforms on screen. High-pass first, then common
        # average reference, then trim the padding used to avoid edge ringing.
        V = sosfiltfilt(sos, V, axis=1)
        V -= np.median(V, axis=0, keepdims=True)
        V = V[:, n_pad:n_pad + n_samp]
        sub = V[c0:c1 + 1]
        q = np.clip(np.round(sub / CLIP_UV * 127), -127, 127).astype(np.int8)
        blocks.append(q)
        if n % 8 == 0 or n == len(chosen) - 1:
            print(f'  [{n+1}/{len(chosen)}] trial {j} '
                  f'({int(ori[j])} deg, {tf[j]:g} Hz)  '
                  f'rms {sub.std():.1f} uV  peak {np.abs(sub).max():.0f} uV',
                  flush=True)
    # Measure the template lag on the FILTERED snippets rather than trusting
    # the value measured on raw data: filtering and referencing can move it.
    lags = []
    for j, q in zip(chosen, blocks):
        for n2, k in enumerate(keep):
            col = int(unit_col[n2]) - c0
            if not (0 <= col < cols.size):
                continue
            t = spikes_of[n2]
            sel = t[(t >= onset[j] + WIN[0] + 0.004)
                    & (t < onset[j] + WIN[1] - 0.004)]
            if sel.size < 8:
                continue
            base0 = np.interp(onset[j] + WIN[0], KT, KS)
            idx = np.round(np.interp(sel, KT, KS) - base0).astype(int)
            hw = int(round(0.002 * FS))
            idx = idx[(idx > hw) & (idx < n_samp - hw)]
            if idx.size < 8:
                continue
            sta = np.mean([q[col, i - hw:i + hw + 1] for i in idx], axis=0)
            if sta.min() * CLIP_UV / 127 < -40:
                lags.append((np.argmin(sta) - hw) / FS * 1000)
    lag_ms = float(np.median(lags)) if lags else AL['spikeLagMs']
    print(f'\ntemplate lag measured on the snippets: {lag_ms:+.3f} ms '
          f'(n={len(lags)} unit-trials)')
    f.close()

    def align8(n):
        return (n + 7) & ~7

    header = dict(
        trials=[int(x) for x in chosen],
        nSamples=n_samp, nChannels=int(cols.size),
        firstColumn=int(c0),
        colDepth=[None if np.isnan(d) else float(d) for d in col_depth],
        win=list(WIN), fs=FS, uvFullScale=CLIP_UV,
        spikeLagMs=lag_ms,
        hpHz=HP_HZ,
        note=('int8, %g Hz high-pass then common average reference, '
              '+/- %g uV full scale' % (HP_HZ, CLIP_UV)),
        offsets={}, blockBytes=int(blocks[0].nbytes),
    )

    def enc(h):
        return json.dumps(h, separators=(',', ':')).encode()

    for j in chosen:
        header['offsets'][str(j)] = 999999999
    hlen = align8(4 + len(enc(header))) - 4
    pos = align8(4 + hlen)
    for j in chosen:
        header['offsets'][str(j)] = pos
        pos = align8(pos + blocks[0].nbytes)
    blob = enc(header)
    assert len(blob) <= hlen, (len(blob), hlen)
    blob += b' ' * (hlen - len(blob))

    out = os.path.join(OUT_DIR, 'volt.bin')
    with open(out, 'wb') as fh:
        fh.write(struct.pack('<I', len(blob)))
        fh.write(blob)
        for j, q in zip(chosen, blocks):
            assert fh.tell() == header['offsets'][str(j)]
            b = q.tobytes()
            fh.write(b)
            fh.write(b'\0' * (align8(len(b)) - len(b)))
    print(f'\nwrote {out}  ({os.path.getsize(out)/1e6:.1f} MB)')
    print(f'  {len(chosen)} trials x {cols.size} channels x {n_samp} samples')


if __name__ == '__main__':
    main()
