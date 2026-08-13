"""
Recover the map from spike_band.dat sample index to NWB session time.

Allen records the same barcode pulse train on two clocks: the probe's own
acquisition clock (`event_timestamps.npy`, in probe samples) and the master
NI-DAQ sync line (`sync.h5`, 100 kHz, line 0 = "barcodes"). The session NWB is
written in the master clock, so aligning the two gives us the map.

We do NOT decode the barcode values. Each barcode burst begins with a leading
pulse after a multi-second gap, and those burst-start times are a distinctive
enough fingerprint to match the two lists directly, then fit

    t_session = A * sample_index + B

by least squares. If the fit is right the residuals are microseconds, which is
a far stronger check than any single decoded value.

Writes align.json next to this file.
"""
import json
import os

import h5py
import numpy as np

CACHE = r'D:\temp\allen_drilldown'
HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = 733744649
FS_NOMINAL = 29999.9916646214
SYNC_HZ = 100000.0
GAP = 5.0                      # s; a rising edge after this long a gap starts
                               # a new barcode burst


def burst_starts(times, gap=GAP):
    if times.size == 0:
        return times
    keep = np.concatenate(([True], np.diff(times) > gap))
    return times[keep]


def main():
    # ---- probe clock ---------------------------------------------------
    ev = np.load(os.path.join(CACHE, f'{PROBE}_event_timestamps.npy'))
    cs = np.load(os.path.join(CACHE, f'{PROBE}_channel_states.npy'))
    probe_rise = ev[cs == 1].astype(np.float64)          # barcode line rising
    probe_t = probe_rise / FS_NOMINAL
    pb = burst_starts(probe_t)
    pb_samples = burst_starts(probe_t) * FS_NOMINAL
    print(f'probe : {probe_rise.size} rising edges -> {pb.size} bursts, '
          f'{pb[0]:.3f} .. {pb[-1]:.3f} s')

    # ---- master sync clock ---------------------------------------------
    with h5py.File(os.path.join(CACHE, 'sync.h5'), 'r') as f:
        d = f['data'][:]
    t = d[:, 0].astype(np.float64) / SYNC_HZ
    bit = (d[:, 1] & 1).astype(np.int8)                  # line 0 = barcodes
    rise = np.nonzero((bit[1:] == 1) & (bit[:-1] == 0))[0] + 1
    sync_t = t[rise]
    sb = burst_starts(sync_t)
    print(f'sync  : {sync_t.size} rising edges -> {sb.size} bursts, '
          f'{sb[0]:.3f} .. {sb[-1]:.3f} s')

    # ---- match the two burst lists -------------------------------------
    # Both should be the same barcodes; allow the probe to have started late
    # or stopped early by trying every integer shift and keeping the one whose
    # inter-burst spacings agree best.
    best = None
    for shift in range(-20, 21):
        i0 = max(0, shift)
        j0 = max(0, -shift)
        n = min(pb.size - i0, sb.size - j0)
        if n < 20:
            continue
        p, s = pb[i0:i0 + n], sb[j0:j0 + n]
        # spacing agreement is shift- and offset-invariant
        err = np.abs(np.diff(p) - np.diff(s)).max()
        if best is None or err < best[0]:
            best = (err, shift, i0, j0, n)
    err, shift, i0, j0, n = best
    print(f'\nbest shift {shift}: {n} paired bursts, '
          f'max spacing mismatch {err*1000:.3f} ms')

    p_samp = pb_samples[i0:i0 + n]
    s_time = sb[j0:j0 + n]
    A, B = np.polyfit(p_samp, s_time, 1)
    resid = s_time - (A * p_samp + B)
    print(f'\ncoarse fit from burst starts:')
    print(f'  t_session = {A:.12e} * sample + {B:.9f}')
    print(f'  offset {B:+.6f} s, residuals rms {resid.std()*1e6:.1f} us, '
          f'max {np.abs(resid).max()*1e6:.1f} us')

    # ---- refine on EVERY barcode edge ----------------------------------
    # Burst starts alone leave a few hundred microseconds of slop. Use the
    # coarse map to pair up all ~2750 individual edges, then fit those.
    pred = A * probe_rise + B
    k = np.searchsorted(sync_t, pred)
    k = np.clip(k, 1, sync_t.size - 1)
    left, right = sync_t[k - 1], sync_t[k]
    nearest = np.where(np.abs(pred - left) < np.abs(pred - right), left, right)
    err = nearest - pred
    good = np.abs(err) < 0.005                       # 5 ms pairing window
    print(f'\ndense pairing: {good.sum()} of {probe_rise.size} edges matched '
          f'within 5 ms')
    P, S = probe_rise[good], nearest[good]
    A2, B2 = np.polyfit(P, S, 1)
    r2 = S - (A2 * P + B2)
    print(f'  t_session = {A2:.12e} * sample + {B2:.9f}')
    print(f'  implied sampling rate {1/A2:.6f} Hz '
          f'(probes.csv says {FS_NOMINAL:.6f})')
    print(f'  offset {B2:+.6f} s')
    print(f'  residuals rms {r2.std()*1e6:.1f} us, '
          f'max {np.abs(r2).max()*1e6:.1f} us')

    # ---- knots for interpolation: ONE PER BURST, not one per edge ---------
    # Interpolating between individual edges is a trap. Edges inside a barcode
    # burst are only milliseconds apart, so a few tens of microseconds of
    # detection jitter produces a local slope that is wrong by percent, and
    # spike positions computed from it scatter by a millisecond. Averaging
    # each burst to a single knot (bursts are ~30 s apart) makes the local
    # slope accurate to parts per million while still tracking clock drift.
    burst_id = np.cumsum(np.concatenate(([0], np.diff(P) / FS_NOMINAL > GAP)))
    kn_s, kn_t = [], []
    for b in np.unique(burst_id):
        m = burst_id == b
        kn_s.append(P[m].mean())
        kn_t.append(S[m].mean())
    kn_s, kn_t = np.asarray(kn_s), np.asarray(kn_t)
    print(f'\ninterpolation knots: {kn_s.size} (one per burst, '
          f'{np.median(np.diff(kn_t)):.1f} s apart)')

    # The quantity that matters is the LOCAL slope: a snippet is 200 ms long,
    # and anything reading it assumes a constant sample rate over that span.
    # Comparing individual edges to the burst-mean knots is not informative
    # (an edge sits up to half a second away from its own knot), so measure
    # the slope spread instead.
    slope = np.diff(kn_t) / np.diff(kn_s)
    ppm = (slope.max() / slope.min() - 1) * 1e6
    print(f'  local sample rate ranges {1/slope.max():.3f} .. '
          f'{1/slope.min():.3f} Hz, spread {ppm:.1f} ppm')
    print(f'  -> over a 200 ms snippet that is at most '
          f'{ppm*1e-6*0.2*FS_NOMINAL:.3f} samples of error')

    A, B, resid = A2, B2, r2
    out = dict(sessionId=732592105, probeId=PROBE,
               A=A, B=B, samplingRate=1 / A, nBursts=int(n),
               nEdges=int(P.size),
               residualRmsUs=float(resid.std() * 1e6),
               maxResidualUs=float(np.abs(resid).max() * 1e6),
               knotSamples=kn_s.tolist(), knotTimes=kn_t.tolist(),
               knotSlopePpm=float(ppm),
               nChannels=384, uvPerBit=0.195)
    with open(os.path.join(HERE, 'align.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print('\nwrote align.json')


if __name__ == '__main__':
    main()
