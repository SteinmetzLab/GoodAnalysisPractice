"""
Pre-extract raw-voltage snippets for EVERY drifting-grating trial and write
web/volt.json plus web/volt_NN.bin.

Allen's spike_band.dat is 217 GB per probe and their S3 bucket sends no CORS
header, so the browser can neither stream it nor read it cross-origin. But the
file is flat channel-interleaved int16, so a time window is contiguous bytes
and a ranged GET from Python fetches exactly what we want. We pull one window
per trial, crop to the channels spanning our V1 units, high-pass, common
average reference, quantise to int8, and host the result next to the page.

Sample <-> session-time mapping comes from align.json (align_probe_clock.py,
confirmed by verify_alignment.py and check_voltage.py).

Output is SHARDED. One 323 MB file would be awkward to host and impossible to
put in git; ~40 MB shards are neither. volt.json holds the per-trial (shard,
byte offset) table and is small enough to commit. The shards are gitignored --
this script regenerates them.

Run:  <venv>/python build_voltage.py
      <venv>/python build_voltage.py --trials 40     (quick subset for testing)
"""

import argparse
import concurrent.futures as cf
import json
import os
import struct
import time

import h5py
import numpy as np
import requests
from scipy.signal import butter, sosfiltfilt

URL = ('https://allen-brain-observatory.s3.us-west-2.amazonaws.com'
       '/visual-coding-neuropixels/raw-data/732592105/733744649/spike_band.dat')
NWB = r'D:\temp\allen_drilldown\session_732592105.nwb'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'web')

PROBE_ID = 733744649
NCH = 384
UV_PER_BIT = 0.195
STRUCTURE = 'VISp'

WIN = (-0.05, 0.15)        # s around stimulus onset
PAD = 0.02                 # s of extra signal each side, trimmed after filtering
HP_HZ = 300.0              # high-pass for display
CH_MARGIN = 8              # extra channels beyond the V1 unit span
CLIP_UV = 400.0            # int8 full scale after filtering
SHARD_MB = 40              # target shard size
WORKERS = 6                # concurrent ranged GETs
QC = dict(amplitude_cutoff=0.1, presence_ratio=0.95, isi_violations=0.5)

AL = json.load(open(os.path.join(HERE, 'align.json')))
KS = np.asarray(AL['knotSamples'], float)
KT = np.asarray(AL['knotTimes'], float)
FS = AL['samplingRate']

SESSION = requests.Session()
SESSION.mount('https://', requests.adapters.HTTPAdapter(
    pool_connections=WORKERS, pool_maxsize=WORKERS, max_retries=3))


def time_to_sample(t):
    return np.interp(t, KT, KS)


def fetch(s0, n):
    """Ranged GET of a contiguous sample block, all channels. int16."""
    b0, b1 = s0 * NCH * 2, (s0 + n) * NCH * 2
    for attempt in range(4):
        try:
            r = SESSION.get(URL, headers={'Range': f'bytes={b0}-{b1-1}'},
                            timeout=600)
            r.raise_for_status()
            if r.status_code != 206 or len(r.content) != (b1 - b0):
                raise IOError(f'status {r.status_code}, '
                              f'{len(r.content)} of {b1-b0} bytes')
            return np.frombuffer(r.content, dtype='<i2').reshape(-1, NCH).T
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=0,
                    help='only the first N trials (for a quick test)')
    args = ap.parse_args()

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
    unit_col = el_local[e_idx].astype(int)
    print(f'{len(keep)} V1 units, peak columns '
          f'{unit_col.min()}..{unit_col.max()}')

    c0 = int(max(0, unit_col.min() - CH_MARGIN))
    c1 = int(min(NCH - 1, unit_col.max() + CH_MARGIN))
    n_ch = int(c1 - c0 + 1)
    col_depth = np.full(n_ch, np.nan)
    for e in range(len(el_id)):
        if int(el_probe[e]) != PROBE_ID:
            continue
        li = int(el_local[e])
        if c0 <= li <= c1:
            col_depth[li - c0] = float(el_y[e])
    print(f'channels {c0}..{c1} ({n_ch})')

    ends = u['spike_times_index'][:]
    st_starts = np.concatenate(([0], ends[:-1]))
    st_all = f['units/spike_times']
    spikes_of = [st_all[st_starts[k]:ends[k]] for k in keep]

    dg = f['intervals/drifting_gratings_presentations']
    ori_all, tf_all = dg['orientation'][:], dg['temporal_frequency'][:]
    ok = np.isfinite(ori_all)
    onset, ori, tf = dg['start_time'][:][ok], ori_all[ok], tf_all[ok]
    f.close()

    chosen = list(range(len(onset)))
    if args.trials:
        chosen = chosen[:args.trials]

    n_samp = int(round((WIN[1] - WIN[0]) * FS))
    n_pad = int(round(PAD * FS))
    sos = butter(3, HP_HZ, btype='highpass', fs=FS, output='sos')
    block_bytes = n_samp * n_ch
    total_mb = block_bytes * len(chosen) / 1e6
    fetch_gb = (n_samp + 2 * n_pad) * NCH * 2 * len(chosen) / 1e9
    print(f'\n{len(chosen)} trials, window {WIN[0]}..{WIN[1]} s '
          f'({n_samp} samples), high-pass {HP_HZ:g} Hz')
    print(f'  {block_bytes/1e6:.2f} MB per trial as int8 -> '
          f'{total_mb:.0f} MB total')
    print(f'  will pull ~{fetch_gb:.1f} GB from S3 with {WORKERS} workers')

    def one(j):
        s0 = int(round(time_to_sample(onset[j] + WIN[0]))) - n_pad
        V = fetch(s0, n_samp + 2 * n_pad).astype(np.float32) * UV_PER_BIT
        # The AP band still carries plenty of low-frequency power. Averaging
        # cancels it but a single trial does not, and the slow blobs swamp the
        # 1 ms waveforms on screen. High-pass, then common average reference,
        # then trim the padding that kept filter ringing out of the window.
        V = sosfiltfilt(sos, V, axis=1)
        V -= np.median(V, axis=0, keepdims=True)
        sub = V[c0:c1 + 1, n_pad:n_pad + n_samp]
        return np.clip(np.round(sub / CLIP_UV * 127), -127, 127).astype(np.int8)

    blocks = [None] * len(chosen)
    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one, j): n for n, j in enumerate(chosen)}
        for fut in cf.as_completed(futs):
            n = futs[fut]
            blocks[n] = fut.result()
            done += 1
            if done % 25 == 0 or done == len(chosen):
                el_s = time.time() - t0
                rate = done / el_s
                print(f'  {done}/{len(chosen)}  {rate:.1f} trials/s  '
                      f'eta {(len(chosen)-done)/max(rate,1e-9)/60:.1f} min',
                      flush=True)

    # ---- measure the template lag on the filtered snippets -----------------
    lags = []
    for n, j in enumerate(chosen[:60]):
        q = blocks[n]
        base0 = np.interp(onset[j] + WIN[0], KT, KS)
        for n2 in range(len(keep)):
            col = int(unit_col[n2]) - c0
            if not (0 <= col < n_ch):
                continue
            t = spikes_of[n2]
            sel = t[(t >= onset[j] + WIN[0] + 0.004)
                    & (t < onset[j] + WIN[1] - 0.004)]
            if sel.size < 8:
                continue
            hw = int(round(0.002 * FS))
            idx = np.round(np.interp(sel, KT, KS) - base0).astype(int)
            idx = idx[(idx > hw) & (idx < n_samp - hw)]
            if idx.size < 8:
                continue
            sta = np.mean([q[col, i - hw:i + hw + 1] for i in idx], axis=0)
            if sta.min() * CLIP_UV / 127 < -40:
                lags.append((np.argmin(sta) - hw) / FS * 1000)
    lag_ms = float(np.median(lags)) if lags else AL['spikeLagMs']
    print(f'\ntemplate lag on the snippets: {lag_ms:+.3f} ms '
          f'(n={len(lags)} unit-trials)')

    # ---- shard ------------------------------------------------------------
    per_shard = max(1, int(SHARD_MB * 1e6 // block_bytes))
    n_shards = (len(chosen) + per_shard - 1) // per_shard
    print(f'\nwriting {n_shards} shards of up to {per_shard} trials '
          f'({per_shard*block_bytes/1e6:.0f} MB each)')

    for old in os.listdir(OUT_DIR):
        if old.startswith('volt') and old.endswith('.bin'):
            os.remove(os.path.join(OUT_DIR, old))

    offsets = {}
    shard_names = []
    for sh in range(n_shards):
        lo, hi = sh * per_shard, min((sh + 1) * per_shard, len(chosen))
        name = f'volt_{sh:02d}.bin'
        shard_names.append(name)
        with open(os.path.join(OUT_DIR, name), 'wb') as fh:
            for n in range(lo, hi):
                offsets[str(chosen[n])] = [sh, fh.tell()]
                fh.write(blocks[n].tobytes())

    header = dict(
        trials=[int(x) for x in chosen],
        shards=shard_names,
        nSamples=n_samp, nChannels=n_ch, firstColumn=int(c0),
        colDepth=[None if np.isnan(d) else float(d) for d in col_depth],
        win=list(WIN), fs=FS, uvFullScale=CLIP_UV, hpHz=HP_HZ,
        spikeLagMs=lag_ms, blockBytes=int(block_bytes),
        note=('int8, %g Hz high-pass then common average reference, '
              '+/- %g uV full scale' % (HP_HZ, CLIP_UV)),
        offsets=offsets,
    )
    with open(os.path.join(OUT_DIR, 'volt.json'), 'w') as fh:
        json.dump(header, fh, separators=(',', ':'))

    tot = sum(os.path.getsize(os.path.join(OUT_DIR, s)) for s in shard_names)
    print(f'\nwrote volt.json ('
          f'{os.path.getsize(os.path.join(OUT_DIR, "volt.json"))/1e3:.0f} KB) '
          f'+ {n_shards} shards ({tot/1e6:.0f} MB)')
    print(f'  every one of the {len(chosen)} trials has raw voltage')


if __name__ == '__main__':
    main()
