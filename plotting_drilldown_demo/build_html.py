"""
Build web/index.html -- a standalone, dependency-free HTML version of the
drill-down demo.

Everything runs in the browser: the PSTHs are binned and smoothed in
JavaScript from the embedded spike times, the plots are drawn on <canvas>, and
the raw voltage is simulated on demand exactly as generate_data.py does it
(white noise, plus a negative 2-D Gaussian followed by a wider positive one per
spike). There is no server, no CDN, and no library -- the file is entirely
self-contained.

Only the spikes that fall inside a trial window are embedded, stored relative
to their trial onset, which is both all the page ever displays and much kinder
to float32 than absolute session time.

Run:  python build_html.py
"""

import base64
import json
import os

import numpy as np
import matplotlib.pyplot as plt

import drilldown_core as core
import generate_data as gd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'web')
OUT = os.path.join(OUT_DIR, 'index.html')


def b64(arr, dtype):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode()


def build_payload():
    S = core.Session(os.path.join(HERE, 'data', 'dataset.npz'))

    # Spikes inside trial windows only, relative to onset, neuron-major.
    times, bounds = [], [0]
    for i in range(S.n_neurons):
        for j in range(S.n_trials):
            t = S.trial_spikes(i, j)
            times.append(t)
            bounds.append(bounds[-1] + t.size)
    times = np.concatenate(times)

    colors = ['#%02x%02x%02x' % tuple(int(round(255 * c)) for c in rgb[:3])
              for rgb in S.colors]
    lut = lambda name: (np.asarray(plt.get_cmap(name)(np.linspace(0, 1, 256)))
                        [:, :3] * 255).round().astype(np.uint8)

    return S, dict(
        spT=b64(times, np.float32),
        spIdx=b64(np.array(bounds), np.uint32),
        depth=b64(S.depth, np.float32),
        wfAmp=b64(S.wf_amp, np.float32),
        stim=b64(S.stim, np.uint8),
        byDepth=b64(S.by_depth, np.uint16),
        magma=b64(lut('magma'), np.uint8),
        rdbu=b64(lut('RdBu_r'), np.uint8),
        colors=colors,
        trialNorms=list(core.TRIAL_NORMS),
        neuronNorms=list(core.NEURON_NORMS),
        stats=list(core.STATS),
        rateFloor=core.RATE_FLOOR, sdFloor=core.SD_FLOOR,
        nBoot=core.N_BOOT, bootSeed=core.BOOT_SEED,
        stimNames=S.stim_names,
        stimColors=list(core.STIM_COLORS),
        nNeurons=S.n_neurons, nTrials=S.n_trials, nStim=S.n_stim,
        pre=S.pre, post=S.post, bin=core.BIN, smoothSd=core.SMOOTH_SD,
        hitTol=core.HIT_TOL,
        nChannels=S.n_channels, chPitch=S.ch_pitch, fs=S.fs_volt,
        noiseSd=gd.NOISE_SD_UV, voltClim=core.VOLT_CLIM,
        voltWidth=core.VOLT_WIDTH, voltCentre=core.VOLT_CENTRE,
        voltSeed=gd.SEED_VOLT,
        wf=dict(stTr=gd.WF_TROUGH_ST, stPk=gd.WF_PEAK_ST,
                delay=gd.WF_PEAK_DELAY, frac=gd.WF_PEAK_FRAC,
                sdTr=gd.WF_SIGMA_D_TR, sdPk=gd.WF_SIGMA_D_PK),
    )


# ===========================================================================
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drill-down: from the population average to the raw voltage</title>
<style>
  :root{
    --ink:#1b1b1d; --mid:#5a5f66; --faint:#9aa0a8;
    --rule:#e2e5ea; --panel:#fff; --page:#f4f6f8;
  }
  *{box-sizing:border-box}
  body{
    margin:0; background:var(--page); color:var(--ink);
    font:15px/1.55 Arial, "Helvetica Neue", Helvetica, sans-serif;
  }
  .wrap{max-width:1280px; margin:0 auto; padding:28px 20px 64px}
  header h1{font-size:25px; margin:0 0 8px; letter-spacing:-.01em}
  header p{margin:0 0 6px; color:var(--mid); max-width:76ch}
  header .lede{color:var(--ink)}
  .steps{
    display:flex; flex-wrap:wrap; gap:8px; margin:18px 0 4px; padding:0;
    list-style:none; font-size:13px;
  }
  .steps li{
    background:#fff; border:1px solid var(--rule); border-radius:999px;
    padding:5px 13px; color:var(--mid);
  }
  .steps li b{color:var(--ink); font-weight:700}
  .controls{
    display:flex; flex-wrap:wrap; gap:18px 26px; align-items:flex-end;
    margin-top:20px; padding:14px 16px; background:#fff;
    border:1px solid var(--rule); border-radius:10px;
  }
  .controls .field{display:flex; flex-direction:column; gap:4px; min-width:0}
  .controls .field b{font-size:12px; letter-spacing:.02em}
  .controls .field span{font-size:11px; color:var(--faint)}
  .controls select{
    font:13px Arial, sans-serif; padding:5px 8px; border-radius:6px;
    border:1px solid #c9ced6; background:#fff; color:var(--ink); max-width:100%;
  }
  .controls .why{
    flex:1 1 260px; font-size:12px; color:var(--mid); line-height:1.45;
    min-width:220px;
  }
  .floornote{
    margin:8px 2px 0; font-size:12px; font-style:italic; color:#a33;
    min-height:15px;
  }
  .grid{
    display:grid; grid-template-columns:repeat(2, minmax(0,1fr));
    gap:16px; margin-top:14px;
  }
  @media (max-width:900px){ .grid{grid-template-columns:1fr} }
  .panel{
    background:var(--panel); border:1px solid var(--rule); border-radius:10px;
    padding:10px 12px 12px; min-width:0;
  }
  .panel canvas{width:100%; height:330px; display:block; cursor:default}
  .panel.clickable canvas{cursor:pointer}
  .cap{
    font-size:12px; color:var(--faint); margin:6px 2px 0; min-height:16px;
    font-style:italic;
  }
  footer{
    margin-top:34px; padding-top:18px; border-top:1px solid var(--rule);
    color:var(--mid); font-size:13px; max-width:80ch;
  }
  footer code{
    background:#eceff3; padding:1px 5px; border-radius:4px; font-size:12px;
  }
  kbd{
    background:#fff; border:1px solid #c9ced6; border-bottom-width:2px;
    border-radius:4px; padding:0 5px; font:12px/1.5 inherit;
  }
  a{color:#0060b0}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>From the population average to the raw voltage</h1>
  <p class="lede">This starts with the figure everyone publishes &mdash; the
  average PSTH across all neurons, one trace per stimulus &mdash; and lets you
  click your way down to the data it was computed from.</p>
  <ul class="steps">
    <li><b>1.</b> click a trace</li>
    <li><b>2.</b> click a neuron's row</li>
    <li><b>3.</b> click a trial row</li>
    <li><b>4.</b> you are looking at voltage</li>
  </ul>
</header>

<div class="controls">
  <label class="field"><b>normalize each TRIAL</b>
    <span>before averaging trials</span>
    <select id="sel-trial"></select></label>
  <label class="field"><b>normalize each NEURON</b>
    <span>before averaging neurons</span>
    <select id="sel-neuron"></select></label>
  <label class="field"><b>average with</b>
    <span>&nbsp;</span>
    <select id="sel-stat"></select></label>
  <p class="why">Try <b>divide by baseline</b> on the TRIAL selector, then move
  it to the NEURON selector instead. Per neuron the peak is a sane
  ~3.9&times; over a baseline of 1. Per trial it jumps to ~6.2&times; and the
  whole post-stimulus period floats near 1.8 &mdash; because each denominator
  is a baseline estimated from a couple of spikes. Averaging ratios with noisy
  denominators biases the answer upward.</p>
</div>
<p class="floornote" id="floornote"></p>

<div class="grid">
  <div class="panel clickable" id="p-sum"><canvas id="c-sum"></canvas>
    <p class="cap">Click a trace to see the neurons behind it.</p></div>
  <div class="panel" id="p-mat"><canvas id="c-mat"></canvas>
    <p class="cap" id="cap-mat"></p></div>
  <div class="panel" id="p-ras"><canvas id="c-ras"></canvas>
    <p class="cap" id="cap-ras"></p></div>
  <div class="panel" id="p-psth"><canvas id="c-psth"></canvas>
    <p class="cap"></p></div>
  <div class="panel" id="p-trial"><canvas id="c-trial"></canvas>
    <p class="cap" id="cap-trial"></p></div>
  <div class="panel" id="p-volt"><canvas id="c-volt"></canvas>
    <p class="cap" id="cap-volt"></p></div>
</div>

<footer>
  <p><b>The data are synthetic.</b> 50 neurons, 120 trials (40 each of stimuli
  A, B and C, in random order). Every neuron's response amplitude to each
  stimulus is drawn from a Gaussian whose mean depends on the stimulus
  (A&nbsp;&asymp;&nbsp;14, B&nbsp;&asymp;&nbsp;6, C&nbsp;&asymp;&nbsp;5
  spikes/s), so A is bigger <i>on average</i> while individual neurons vary a
  lot and some are suppressed. The response time course is a gamma-shaped
  kernel &mdash; fast rise, slower decay, peak at ~42&nbsp;ms, essentially over
  within 200&nbsp;ms. Spikes are an inhomogeneous Poisson process.</p>

  <p><b>The voltage is simulated in your browser</b>, on demand, from the spike
  times of whichever trial you clicked: white noise on 64 channels at
  20&nbsp;&micro;m pitch, plus one waveform per spike &mdash; a negative 2-D
  Gaussian in time&nbsp;&times;&nbsp;depth followed 0.65&nbsp;ms later by a
  wider-in-time positive one. So the coloured dots are <b>ground truth</b>, not
  the output of a spike sorter: every dot sits exactly on the waveform that
  produced it. Scroll to zoom the voltage panel; press <kbd>d</kbd> to hide the
  dots.</p>

  <p>The whole page is one self-contained HTML file &mdash; no server, no
  libraries. Python versions of the same demo (a standalone
  <code>drilldown.py</code> and a Jupyter notebook) live in the
  <code>plotting_drilldown_demo</code> folder of the
  <a href="https://github.com/SteinmetzLab/GoodAnalysisPractice">Good Analysis
  Practice</a> repository.</p>
</footer>
</div>

<script>
"use strict";
const RAW = __DATA__;

/* ---------------------------------------------------------------- decode */
function b64(s, T){
  const bin = atob(s), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return new T(u.buffer);
}
const SP_T = b64(RAW.spT, Float32Array), SP_IDX = b64(RAW.spIdx, Uint32Array);
const DEPTH = b64(RAW.depth, Float32Array), WFAMP = b64(RAW.wfAmp, Float32Array);
const STIM = b64(RAW.stim, Uint8Array), BY_DEPTH = b64(RAW.byDepth, Uint16Array);
const MAGMA = b64(RAW.magma, Uint8Array), RDBU = b64(RAW.rdbu, Uint8Array);

const NN = RAW.nNeurons, NT = RAW.nTrials, NS = RAW.nStim;
const PRE = RAW.pre, POST = RAW.post, BIN = RAW.bin;
const NB = Math.round((PRE + POST) / BIN);
const NCH = RAW.nChannels, PITCH = RAW.chPitch, FS = RAW.fs;
const CLIM = RAW.voltClim;
const SC = RAW.stimColors, SNAME = RAW.stimNames, NCOL = RAW.colors;

const TC = new Float64Array(NB);                       // bin centres
for (let b = 0; b < NB; b++) TC[b] = -PRE + (b + 0.5) * BIN;

function spikes(i, j){ const k = i * NT + j; return SP_T.subarray(SP_IDX[k], SP_IDX[k+1]); }

const TRIALS_OF = [];
for (let s = 0; s < NS; s++) TRIALS_OF.push([]);
for (let j = 0; j < NT; j++) TRIALS_OF[STIM[j]].push(j);

/* ------------------------------------------------- PSTHs (as in Python) */
const RATES = new Float32Array(NN * NT * NB);          // [neuron][trial][bin]
(function(){
  const sd = RAW.smoothSd / BIN, half = Math.ceil(4 * sd);
  const kern = new Float64Array(2 * half + 1);
  let sum = 0;
  for (let q = -half; q <= half; q++){
    kern[q + half] = Math.exp(-0.5 * (q / sd) * (q / sd));
    sum += kern[q + half];
  }
  for (let q = 0; q < kern.length; q++) kern[q] /= sum;

  const tmp = new Float64Array(NB);
  for (let i = 0; i < NN; i++) for (let j = 0; j < NT; j++){
    tmp.fill(0);
    const s = spikes(i, j);
    for (let m = 0; m < s.length; m++){
      const b = Math.floor((s[m] + PRE) / BIN);
      if (b >= 0 && b < NB) tmp[b] += 1 / BIN;
    }
    const off = (i * NT + j) * NB;
    for (let b = 0; b < NB; b++){                      // edge-padded convolve
      let acc = 0;
      for (let q = -half; q <= half; q++){
        let bb = b + q;
        if (bb < 0) bb = 0; else if (bb >= NB) bb = NB - 1;
        acc += tmp[bb] * kern[q + half];
      }
      RATES[off + b] = acc;
    }
  }
})();

/* ---------------------------------------------- normalisation & statistic
 * Mirrors Session.configure() in drilldown_core.py. There are two places you
 * can normalise and they are not equivalent: per trial (denominator estimated
 * from a couple of spikes) or per neuron (denominator pooled over all of that
 * neuron's trials). The neuron step is kept affine so level 2 can apply the
 * identical transform to single trials.
 */
const BASE_N = Math.round(PRE / BIN);            // pre-stimulus bins
const RATE_FLOOR = RAW.rateFloor, SD_FLOOR = RAW.sdFloor;
const N_BOOT = RAW.nBoot;

const CFG = {trialNorm:'none', neuronNorm:'none', stat:'mean'};
let RATES_N, PSTH, NN_OFF, NN_SC, POP;
let VALUE_LABEL, STAT_LABEL, SPREAD_LABEL, N_FLOORED, N_FLOORED_NEURONS;

const SCRATCH = new Float64Array(512);
function central(buf, n){
  if (CFG.stat === 'mean'){
    let a = 0;
    for (let k = 0; k < n; k++) a += buf[k];
    return a / n;
  }
  const s = SCRATCH.subarray(0, n);
  s.set(buf.subarray(0, n));
  s.sort();
  const h = n >> 1;
  return (n & 1) ? s[h] : 0.5 * (s[h-1] + s[h]);
}

/** Central tendency per bin, plus the standard error OF THAT STATISTIC.
 *  Mean -> SEM. Median -> bootstrap SE, with resample indices drawn once and
 *  shared across bins (as the numpy version does). */
function aggregate(pull, n){
  const m = new Float64Array(NB), se = new Float64Array(NB);
  const buf = new Float64Array(n);
  const even = (n % 2) === 0, h = n >> 1;
  let sorted = null, bLo = null, bHi = null;

  if (CFG.stat === 'median'){
    // A bootstrap resample's median is an ORDER STATISTIC of the original
    // sample: median{x[r1]..x[rn]} == xSorted[median{r1..rn}], because
    // xSorted is monotone. So draw the ranks once, reduce each resample to
    // the one or two ranks its median needs, and reuse those for every time
    // bin. That replaces N_BOOT sorts per bin with a single sort per bin --
    // exact, not an approximation, and ~40x faster here.
    const rng = mulberry32(RAW.bootSeed);
    const ranks = new Int32Array(n);
    bLo = new Int32Array(N_BOOT);
    bHi = new Int32Array(N_BOOT);
    for (let r = 0; r < N_BOOT; r++){
      for (let k = 0; k < n; k++) ranks[k] = (rng() * n) | 0;
      ranks.sort();
      bLo[r] = even ? ranks[h - 1] : ranks[h];
      bHi[r] = ranks[h];
    }
    sorted = new Float64Array(n);
  }

  for (let b = 0; b < NB; b++){
    for (let k = 0; k < n; k++) buf[k] = pull(k, b);
    if (CFG.stat === 'mean'){
      let a = 0;
      for (let k = 0; k < n; k++) a += buf[k];
      m[b] = a / n;
      let v = 0;
      for (let k = 0; k < n; k++){ const d = buf[k] - m[b]; v += d * d; }
      se[b] = Math.sqrt(v / (n - 1)) / Math.sqrt(n);
    } else {
      sorted.set(buf);
      sorted.sort();
      m[b] = even ? 0.5 * (sorted[h - 1] + sorted[h]) : sorted[h];
      let a = 0;
      for (let r = 0; r < N_BOOT; r++) a += 0.5 * (sorted[bLo[r]] + sorted[bHi[r]]);
      a /= N_BOOT;
      let v = 0;
      for (let r = 0; r < N_BOOT; r++){
        const d = 0.5 * (sorted[bLo[r]] + sorted[bHi[r]]) - a;
        v += d * d;
      }
      se[b] = Math.sqrt(v / (N_BOOT - 1));
    }
  }
  return {m, se};
}

function configure(){
  // --- per (neuron, trial) baseline
  const bMean = new Float64Array(NN * NT), bSd = new Float64Array(NN * NT);
  for (let i = 0; i < NN; i++) for (let j = 0; j < NT; j++){
    const k = i * NT + j, off = k * NB;
    let a = 0;
    for (let b = 0; b < BASE_N; b++) a += RATES[off + b];
    const m = a / BASE_N;
    bMean[k] = m;
    let v = 0;
    for (let b = 0; b < BASE_N; b++){ const d = RATES[off + b] - m; v += d * d; }
    bSd[k] = Math.sqrt(v / (BASE_N - 1));
  }

  // --- trial-level normalisation
  N_FLOORED = 0;
  if (CFG.trialNorm === 'none'){
    RATES_N = RATES;
  } else {
    RATES_N = new Float32Array(NN * NT * NB);
    for (let i = 0; i < NN; i++) for (let j = 0; j < NT; j++){
      const k = i * NT + j, off = k * NB;
      let sub = 0, div = 1;
      if (CFG.trialNorm === 'subtract baseline'){
        sub = bMean[k];
      } else if (CFG.trialNorm === 'divide by baseline'){
        if (bMean[k] < RATE_FLOOR) N_FLOORED++;
        div = Math.max(bMean[k], RATE_FLOOR);
      } else {                                    // baseline z-score
        if (bSd[k] < SD_FLOOR) N_FLOORED++;
        sub = bMean[k];
        div = Math.max(bSd[k], SD_FLOOR);
      }
      for (let b = 0; b < NB; b++) RATES_N[off + b] = (RATES[off + b] - sub) / div;
    }
  }

  // --- aggregate over trials
  const raw = new Float64Array(NN * NS * NB);
  const buf = new Float64Array(NT);
  for (let i = 0; i < NN; i++) for (let s = 0; s < NS; s++){
    const tr = TRIALS_OF[s], n = tr.length, o = (i * NS + s) * NB;
    for (let b = 0; b < NB; b++){
      for (let k = 0; k < n; k++) buf[k] = RATES_N[(i * NT + tr[k]) * NB + b];
      raw[o + b] = central(buf, n);
    }
  }

  // --- neuron-level normalisation, baseline pooled across stimuli
  NN_OFF = new Float64Array(NN);
  NN_SC = new Float64Array(NN).fill(1);
  N_FLOORED_NEURONS = 0;
  const nb = NS * BASE_N;
  for (let i = 0; i < NN; i++){
    let a = 0;
    for (let s = 0; s < NS; s++) for (let b = 0; b < BASE_N; b++)
      a += raw[(i * NS + s) * NB + b];
    const m = a / nb;
    if (CFG.neuronNorm === 'subtract baseline'){
      NN_OFF[i] = m;
    } else if (CFG.neuronNorm === 'divide by baseline'){
      if (m < RATE_FLOOR) N_FLOORED_NEURONS++;
      NN_SC[i] = Math.max(m, RATE_FLOOR);
    } else if (CFG.neuronNorm === 'peak = 1'){
      let pk = 0;
      for (let s = 0; s < NS; s++) for (let b = 0; b < NB; b++){
        const v = Math.abs(raw[(i * NS + s) * NB + b]);
        if (v > pk) pk = v;
      }
      NN_SC[i] = Math.max(pk, 1e-6);
    } else if (CFG.neuronNorm === 'baseline z-score'){
      let v = 0;
      for (let s = 0; s < NS; s++) for (let b = 0; b < BASE_N; b++){
        const d = raw[(i * NS + s) * NB + b] - m;
        v += d * d;
      }
      const sd = Math.sqrt(v / (nb - 1));
      if (sd < SD_FLOOR) N_FLOORED_NEURONS++;
      NN_OFF[i] = m;
      NN_SC[i] = Math.max(sd, SD_FLOOR);
    }
  }

  PSTH = new Float32Array(NN * NS * NB);
  for (let i = 0; i < NN; i++) for (let s = 0; s < NS; s++)
    for (let b = 0; b < NB; b++){
      const o = (i * NS + s) * NB + b;
      PSTH[o] = (raw[o] - NN_OFF[i]) / NN_SC[i];
    }

  POP = [];
  for (let s = 0; s < NS; s++)
    POP.push(aggregate((k, b) => PSTH[(k * NS + s) * NB + b], NN));

  const last = CFG.neuronNorm !== 'none' ? CFG.neuronNorm : CFG.trialNorm;
  VALUE_LABEL = {
    'none':               'firing rate (spikes/s)',
    'subtract baseline':  'Δ firing rate (spikes/s)',
    'divide by baseline': 'rate / baseline',
    'peak = 1':           'normalized rate (peak = 1)',
    'baseline z-score':   'baseline z-score (SD)',
  }[last];
  STAT_LABEL = CFG.stat === 'mean' ? 'Average' : 'Median';
  SPREAD_LABEL = CFG.stat === 'mean' ? 'SEM' : 'bootstrap SE';
}

/** central tendency and its standard error across trials, one neuron */
function neuronStat(i, s){
  const tr = TRIALS_OF[s], off = NN_OFF[i], sc = NN_SC[i];
  return aggregate((k, b) => (RATES_N[(i * NT + tr[k]) * NB + b] - off) / sc,
                   tr.length);
}

/* ------------------------------------------------------- voltage, in JS */
function mulberry32(a){
  return function(){
    a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

const VCACHE = new Map();

function trialVoltage(trial){
  if (VCACHE.has(trial)) return VCACHE.get(trial);
  const nSamp = Math.round((PRE + POST) * FS);
  const v = new Float32Array(NCH * nSamp);

  const rng = mulberry32(RAW.voltSeed + trial), sd = RAW.noiseSd;
  for (let k = 0; k < v.length; k += 2){                // Box-Muller
    let u = rng(); if (u < 1e-12) u = 1e-12;
    const r = Math.sqrt(-2 * Math.log(u)) * sd, th = 2 * Math.PI * rng();
    v[k] = r * Math.cos(th);
    if (k + 1 < v.length) v[k+1] = r * Math.sin(th);
  }

  const W = RAW.wf, nhw = Math.round(0.003 * FS);
  const trT = new Float64Array(2*nhw+1), pkT = new Float64Array(2*nhw+1);
  for (let q = -nhw; q <= nhw; q++){
    const tt = q / FS;
    trT[q+nhw] = Math.exp(-tt*tt / (2*W.stTr*W.stTr));
    pkT[q+nhw] = Math.exp(-(tt-W.delay)*(tt-W.delay) / (2*W.stPk*W.stPk));
  }

  for (let i = 0; i < NN; i++){
    const s = spikes(i, trial);
    if (!s.length) continue;
    const d0 = DEPTH[i], amp = WFAMP[i];
    const c0 = Math.max(0, Math.ceil((d0 - 3*W.sdPk) / PITCH));
    const c1 = Math.min(NCH - 1, Math.floor((d0 + 3*W.sdPk) / PITCH));
    if (c1 < c0) continue;
    const sTr = new Float64Array(c1-c0+1), sPk = new Float64Array(c1-c0+1);
    for (let c = c0; c <= c1; c++){
      const dd = c * PITCH - d0;
      sTr[c-c0] = Math.exp(-dd*dd / (2*W.sdTr*W.sdTr));
      sPk[c-c0] = Math.exp(-dd*dd / (2*W.sdPk*W.sdPk));
    }
    for (let m = 0; m < s.length; m++){
      const centre = Math.round((s[m] + PRE) * FS);
      let a0 = centre - nhw, b0 = centre + nhw + 1, pa = 0;
      if (a0 < 0){ pa = -a0; a0 = 0; }
      if (b0 > nSamp) b0 = nSamp;
      if (b0 <= a0) continue;
      for (let c = c0; c <= c1; c++){
        const row = c * nSamp, tr = sTr[c-c0] * amp, pk = sPk[c-c0] * amp * W.frac;
        let p = pa;
        for (let x = a0; x < b0; x++, p++) v[row + x] += -tr * trT[p] + pk * pkT[p];
      }
    }
  }
  if (VCACHE.size >= 4) VCACHE.delete(VCACHE.keys().next().value);
  VCACHE.set(trial, {v, nSamp});
  return VCACHE.get(trial);
}

/* ------------------------------------------------------- plotting layer */
function niceTicks(a, b, target){
  const lo = Math.min(a,b), hi = Math.max(a,b), span = hi - lo;
  if (!(span > 0)) return {vals:[lo], step:1};
  const raw = span / target, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const n = raw / mag;
  const step = (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
  const vals = [];
  for (let v = Math.ceil(lo/step)*step; v <= hi + step*1e-9; v += step)
    vals.push(Math.abs(v) < step*1e-9 ? 0 : v);
  return {vals, step};
}
function fmtTick(v, step){
  const d = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
  return v.toFixed(d);
}

class Plot {
  constructor(canvas, margins){
    this.c = canvas;
    this.ctx = canvas.getContext('2d');
    this.m = Object.assign({l:64, r:16, t:42, b:46}, margins || {});
    this.hasData = false;
  }
  resize(){
    const r = this.c.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.dpr = dpr;
    this.c.width = Math.max(1, Math.round(r.width * dpr));
    this.c.height = Math.max(1, Math.round(r.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width; this.h = r.height;
  }
  get area(){
    return {x0:this.m.l, x1:this.w - this.m.r, y0:this.m.t, y1:this.h - this.m.b};
  }
  limits(x0, x1, y0, y1){ this.xl = [x0, x1]; this.yl = [y0, y1]; }
  X(v){ const a = this.area; return a.x0 + (v - this.xl[0]) / (this.xl[1] - this.xl[0]) * (a.x1 - a.x0); }
  Y(v){ const a = this.area; return a.y1 - (v - this.yl[0]) / (this.yl[1] - this.yl[0]) * (a.y1 - a.y0); }
  invX(px){ const a = this.area; return this.xl[0] + (px - a.x0) / (a.x1 - a.x0) * (this.xl[1] - this.xl[0]); }
  invY(py){ const a = this.area; return this.yl[0] + (a.y1 - py) / (a.y1 - a.y0) * (this.yl[1] - this.yl[0]); }
  inside(px, py){
    const a = this.area;
    return px >= a.x0 && px <= a.x1 && py >= a.y0 && py <= a.y1;
  }
  clear(){
    this.resize();
    this.ctx.clearRect(0, 0, this.w, this.h);
    this.hasData = false;
  }
  placeholder(text){
    this.clear();
    const x = this.ctx;
    x.fillStyle = '#9aa0a8';
    x.font = 'italic 13px Arial, sans-serif';
    x.textAlign = 'center'; x.textBaseline = 'middle';
    x.fillText(text, this.w/2, this.h/2);
  }
  frame(o){
    const x = this.ctx, a = this.area;
    x.save();
    x.strokeStyle = '#1b1b1d'; x.fillStyle = '#1b1b1d'; x.lineWidth = 1;
    x.font = '11px Arial, sans-serif';

    const xt = niceTicks(this.xl[0], this.xl[1], 6);
    x.textAlign = 'center'; x.textBaseline = 'top';
    for (const v of xt.vals){
      const px = Math.round(this.X(v)) + 0.5;
      if (px < a.x0 - 0.6 || px > a.x1 + 0.6) continue;
      x.beginPath(); x.moveTo(px, a.y1); x.lineTo(px, a.y1 + 4); x.stroke();
      x.fillText(fmtTick(v, xt.step), px, a.y1 + 7);
    }
    const yt = niceTicks(this.yl[0], this.yl[1], 5);
    x.textAlign = 'right'; x.textBaseline = 'middle';
    for (const v of yt.vals){
      const py = Math.round(this.Y(v)) + 0.5;
      if (py < a.y0 - 0.6 || py > a.y1 + 0.6) continue;
      x.beginPath(); x.moveTo(a.x0, py); x.lineTo(a.x0 - 4, py); x.stroke();
      x.fillText(fmtTick(v, yt.step), a.x0 - 7, py);
    }
    // despined: bottom and left only
    x.beginPath();
    x.moveTo(a.x0 + 0.5, a.y0); x.lineTo(a.x0 + 0.5, a.y1 + 0.5);
    x.lineTo(a.x1, a.y1 + 0.5); x.stroke();

    x.font = '12px Arial, sans-serif';
    x.textAlign = 'center'; x.textBaseline = 'bottom';
    if (o.xlabel) x.fillText(o.xlabel, (a.x0 + a.x1) / 2, this.h - 6);
    if (o.ylabel){
      x.save(); x.translate(12, (a.y0 + a.y1) / 2); x.rotate(-Math.PI/2);
      x.textBaseline = 'top'; x.fillText(o.ylabel, 0, 0); x.restore();
    }
    if (o.title){
      x.fillStyle = o.titleColor || '#1b1b1d';
      x.textBaseline = 'top';
      const lines = o.title.split('\n');
      // Shrink the title until it fits: on a narrow phone the panel is only
      // ~300 px wide and the full-size text would run off the canvas.
      let fs = 13;
      for (; fs > 9; fs--){
        x.font = fs + 'px Arial, sans-serif';
        if (Math.max(...lines.map(L => x.measureText(L).width)) <= this.w - 10) break;
      }
      x.font = fs + 'px Arial, sans-serif';
      lines.forEach((L, k) => x.fillText(L, this.w / 2, 4 + k * (fs + 2)));
    }
    x.restore();
    this.hasData = true;
  }
  clip(){
    const a = this.area, x = this.ctx;
    x.save(); x.beginPath();
    x.rect(a.x0, a.y0, a.x1 - a.x0, a.y1 - a.y0); x.clip();
  }
  line(xs, ys, color, lw){
    this.clip();
    const x = this.ctx;
    x.strokeStyle = color; x.lineWidth = lw || 2;
    x.lineJoin = 'round'; x.beginPath();
    for (let i = 0; i < xs.length; i++){
      const px = this.X(xs[i]), py = this.Y(ys[i]);
      i ? x.lineTo(px, py) : x.moveTo(px, py);
    }
    x.stroke(); x.restore();
  }
  band(xs, lo, hi, color, alpha){
    this.clip();
    const x = this.ctx;
    x.globalAlpha = alpha; x.fillStyle = color; x.beginPath();
    for (let i = 0; i < xs.length; i++){
      const px = this.X(xs[i]), py = this.Y(hi[i]);
      i ? x.lineTo(px, py) : x.moveTo(px, py);
    }
    for (let i = xs.length - 1; i >= 0; i--) x.lineTo(this.X(xs[i]), this.Y(lo[i]));
    x.closePath(); x.fill(); x.restore();
  }
  hline(v, color, lw){
    this.clip();
    const x = this.ctx, a = this.area, py = Math.round(this.Y(v)) + 0.5;
    x.strokeStyle = color; x.lineWidth = lw || 1;
    x.beginPath(); x.moveTo(a.x0, py); x.lineTo(a.x1, py); x.stroke();
    x.restore();
  }
  vline(v, color, lw){
    this.clip();
    const x = this.ctx, a = this.area, px = Math.round(this.X(v)) + 0.5;
    x.strokeStyle = color; x.lineWidth = lw || 1;
    x.beginPath(); x.moveTo(px, a.y0); x.lineTo(px, a.y1); x.stroke();
    x.restore();
  }
  hspan(v, halfHeight, color){
    this.clip();
    const x = this.ctx, a = this.area;
    x.fillStyle = color;
    x.fillRect(a.x0, this.Y(v + halfHeight), a.x1 - a.x0,
               Math.abs(this.Y(v - halfHeight) - this.Y(v + halfHeight)));
    x.restore();
  }
  vspan(x0, x1, color){
    this.clip();
    const x = this.ctx, a = this.area;
    x.fillStyle = color;
    x.fillRect(this.X(x0), a.y0, this.X(x1) - this.X(x0), a.y1 - a.y0);
    x.restore();
  }
  /** vertical raster ticks: rows = [{t:Float32Array, y:number, color}] */
  raster(rows, halfHeight, lw){
    this.clip();
    const x = this.ctx;
    x.lineWidth = lw || 1;
    for (const r of rows){
      const y0 = this.Y(r.y - halfHeight), y1 = this.Y(r.y + halfHeight);
      x.strokeStyle = r.color; x.beginPath();
      for (let k = 0; k < r.t.length; k++){
        const px = Math.round(this.X(r.t[k])) + 0.5;
        x.moveTo(px, y0); x.lineTo(px, y1);
      }
      x.stroke();
    }
    x.restore();
  }
  dots(pts, radius){
    this.clip();
    const x = this.ctx;
    x.lineWidth = 1; x.strokeStyle = '#fff';
    for (const p of pts){
      const px = this.X(p.x), py = this.Y(p.y);
      x.fillStyle = p.color;
      x.beginPath(); x.arc(px, py, radius, 0, 6.2832); x.fill(); x.stroke();
    }
    x.restore();
  }
  /** Paint the plot area from a callback that fills one RGBA device pixel. */
  image(fill){
    const a = this.area, d = this.dpr, x = this.ctx;
    const w = Math.max(1, Math.round((a.x1 - a.x0) * d));
    const h = Math.max(1, Math.round((a.y1 - a.y0) * d));
    const img = x.createImageData(w, h);
    fill(img.data, w, h);
    x.save(); x.setTransform(1, 0, 0, 1, 0, 0);
    x.putImageData(img, Math.round(a.x0 * d), Math.round(a.y0 * d));
    x.restore();
  }
  colorbar(vmin, vmax, lut, label){
    const x = this.ctx, a = this.area;
    const bx = a.x1 + 12, bw = 13, by = a.y0, bh = a.y1 - a.y0;
    const g = x.createLinearGradient(0, by + bh, 0, by);
    for (let k = 0; k <= 10; k++){
      const i = Math.round(k / 10 * 255) * 3;
      g.addColorStop(k / 10, `rgb(${lut[i]},${lut[i+1]},${lut[i+2]})`);
    }
    x.fillStyle = g; x.fillRect(bx, by, bw, bh);
    x.strokeStyle = '#c9ced6'; x.lineWidth = 1;
    x.strokeRect(bx + 0.5, by + 0.5, bw - 1, bh - 1);
    x.fillStyle = '#1b1b1d'; x.font = '11px Arial, sans-serif';
    x.textAlign = 'left'; x.textBaseline = 'middle';
    const t = niceTicks(vmin, vmax, 4);
    for (const v of t.vals){
      const py = by + bh - (v - vmin) / (vmax - vmin) * bh;
      x.fillText(fmtTick(v, t.step), bx + bw + 4, py);
    }
    x.save();
    // rotate(-90) maps local +y onto global +x, so 'bottom' keeps the glyphs
    // to the LEFT of the origin and on the canvas.
    x.translate(this.w - 3, (by + by + bh) / 2); x.rotate(-Math.PI/2);
    x.textAlign = 'center'; x.textBaseline = 'bottom';
    x.font = '11px Arial, sans-serif'; x.fillText(label, 0, 0);
    x.restore();
  }
}

/* --------------------------------------------------------------- panels */
const P = {
  sum:   new Plot(document.getElementById('c-sum')),
  mat:   new Plot(document.getElementById('c-mat'), {r:74}),
  ras:   new Plot(document.getElementById('c-ras'), {t:34}),
  psth:  new Plot(document.getElementById('c-psth')),
  trial: new Plot(document.getElementById('c-trial')),
  volt:  new Plot(document.getElementById('c-volt')),
};
const CAP = {
  mat: document.getElementById('cap-mat'),
  ras: document.getElementById('cap-ras'),
  trial: document.getElementById('cap-trial'),
  volt: document.getElementById('cap-volt'),
};
const state = {stim:null, neuron:null, rowTrial:null, trial:null,
               voltX:[RAW.voltCentre - RAW.voltWidth/2,
                      RAW.voltCentre + RAW.voltWidth/2],
               showDots:true};

const TCA = Array.from(TC);

/* ---- level 0 */
/** y-limits that fit every trace and band, with headroom for the legend */
function fitY(series){
  let lo = Infinity, hi = -Infinity;
  for (const {m, se} of series) for (let b = 0; b < NB; b++){
    lo = Math.min(lo, m[b] - se[b]);
    hi = Math.max(hi, m[b] + se[b]);
  }
  if (!(hi > lo)) { lo -= 1; hi += 1; }
  const span = hi - lo;
  return [Math.min(lo - 0.06 * span, lo), hi + 0.20 * span];
}

function drawSummary(){
  const p = P.sum;
  p.clear();
  const [y0, y1] = fitY(POP);
  p.limits(TC[0], TC[NB-1], y0, y1);
  p.frame({xlabel:'time from stimulus onset (s)', ylabel:VALUE_LABEL,
           title:`${STAT_LABEL} across ${NN} neurons `
                 + `(shading: ±${SPREAD_LABEL} across neurons)`});
  p.vline(0, '#999', 1);
  if (y0 < 0) p.hline(0, '#ccc', 1);
  for (let s = 0; s < NS; s++){
    const lo = [], up = [];
    for (let b = 0; b < NB; b++){
      lo.push(POP[s].m[b] - POP[s].se[b]);
      up.push(POP[s].m[b] + POP[s].se[b]);
    }
    p.band(TCA, lo, up, SC[s], 0.5);
    p.line(TCA, Array.from(POP[s].m), SC[s], 2);
  }
  legend(p, SNAME.map((n,s)=>({label:'stim '+n, color:SC[s]})));
}
function legend(p, items){
  const x = p.ctx, a = p.area;
  x.save(); x.font = '12px Arial, sans-serif'; x.textBaseline = 'middle';
  let y = a.y0 + 12;
  for (const it of items){
    x.strokeStyle = it.color; x.lineWidth = 2.5;
    x.beginPath(); x.moveTo(a.x1 - 74, y); x.lineTo(a.x1 - 52, y); x.stroke();
    x.fillStyle = '#1b1b1d'; x.textAlign = 'left';
    x.fillText(it.label, a.x1 - 47, y);
    y += 17;
  }
  x.restore();
}

/* ---- level 1 */
function drawMatrix(){
  const p = P.mat, s = state.stim;
  p.clear();
  if (s === null){ p.placeholder('click a trace in the panel to the left'); return; }

  const vals = new Float64Array(NN * NB);
  let vmin = Infinity;
  for (let i = 0; i < NN; i++) for (let b = 0; b < NB; b++){
    const v = PSTH[(i * NS + s) * NB + b];
    vals[i * NB + b] = v;
    if (v < vmin) vmin = v;
  }
  // Normalisation can send values below zero; a sequential map would hide the
  // sign, so switch to a symmetric diverging one.
  const diverging = vmin < -1e-9;
  let lo, hi, LUT;
  if (diverging){
    const mag = Float64Array.from(vals, Math.abs).sort();
    hi = mag[Math.floor(0.995 * (mag.length - 1))];
    lo = -hi; LUT = RDBU;
  } else {
    const srt = Float64Array.from(vals).sort();
    lo = 0; hi = srt[Math.floor(0.995 * (srt.length - 1))]; LUT = MAGMA;
  }
  const span = (hi - lo) || 1;

  p.limits(TC[0], TC[NB-1], NN - 0.5, -0.5);
  p.image((buf, w, h) => {
    for (let py = 0; py < h; py++){
      const i = Math.min(NN-1, Math.max(0, Math.floor(py / h * NN)));
      for (let px = 0; px < w; px++){
        const b = Math.min(NB-1, Math.max(0, Math.floor(px / w * NB)));
        let f = (PSTH[(i * NS + s) * NB + b] - lo) / span;
        f = f < 0 ? 0 : f > 1 ? 1 : f;
        const c = Math.round(f * 255) * 3, o = (py * w + px) * 4;
        buf[o] = LUT[c]; buf[o+1] = LUT[c+1]; buf[o+2] = LUT[c+2]; buf[o+3] = 255;
      }
    }
  });
  p.frame({xlabel:'time from stimulus onset (s)', ylabel:'neuron #',
           title:`Stim ${SNAME[s]}: every neuron behind that average\n(${NN} neurons, ${TRIALS_OF[s].length} trials each)`,
           titleColor:SC[s]});
  p.vline(0, 'rgba(255,255,255,.6)', 1);
  p.colorbar(lo, hi, LUT, VALUE_LABEL);
  CAP.mat.textContent = 'Click a row to see that neuron.';
}

/* ---- level 2 */
function drawNeuron(){
  const pr = P.ras, pp = P.psth, i = state.neuron;
  pr.clear(); pp.clear();
  if (i === null){
    pr.placeholder('click a row in the neurons × time panel');
    pp.placeholder('');
    CAP.ras.textContent = '';
    return;
  }
  const rows = [], rowTrial = [], bounds = [];
  let y = 0;
  for (let s = 0; s < NS; s++){
    for (const j of TRIALS_OF[s]){
      rows.push({t:spikes(i, j), y:y, color:SC[s]});
      rowTrial.push(j); y++;
    }
    bounds.push(y);
  }
  state.rowTrial = rowTrial;

  pr.limits(-PRE, POST, rows.length - 0.5, -0.5);
  pr.frame({xlabel:'time from stimulus onset (s)',
            ylabel:'trial (grouped by stimulus)',
            title:`Neuron ${i}: all ${NT} trials`});
  pr.vline(0, '#999', 1);
  pr.raster(rows, 0.42, 1);
  const cx = pr.ctx;
  cx.save(); cx.strokeStyle = '#b8bcc2'; cx.lineWidth = 1;
  for (let k = 0; k < bounds.length - 1; k++){
    const py = Math.round(pr.Y(bounds[k] - 0.5)) + 0.5;
    cx.beginPath(); cx.moveTo(pr.area.x0, py); cx.lineTo(pr.area.x1, py); cx.stroke();
  }
  cx.font = 'bold 12px Arial, sans-serif'; cx.textBaseline = 'middle';
  cx.textAlign = 'right';
  for (let s = 0; s < NS; s++){
    const lo = s === 0 ? 0 : bounds[s-1];
    cx.fillStyle = SC[s];
    cx.fillText(SNAME[s], pr.area.x1 - 6, pr.Y((lo + bounds[s] - 1) / 2));
  }
  cx.restore();
  CAP.ras.textContent = 'Click a trial row to see the raw data for that trial.';

  const mm = [];
  for (let s = 0; s < NS; s++) mm.push(neuronStat(i, s));
  const [y0, y1] = fitY(mm);
  pp.limits(-PRE, POST, y0, y1);
  pp.frame({xlabel:'time from stimulus onset (s)', ylabel:VALUE_LABEL,
            title:`Neuron ${i} ${CFG.stat} PSTHs\n`
                  + `(shading: ±${SPREAD_LABEL} across trials)`});
  pp.vline(0, '#999', 1);
  if (y0 < 0) pp.hline(0, '#ccc', 1);
  for (let s = 0; s < NS; s++){
    const lo = [], up = [];
    for (let b = 0; b < NB; b++){
      lo.push(mm[s].m[b] - mm[s].se[b]);
      up.push(mm[s].m[b] + mm[s].se[b]);
    }
    pp.band(TCA, lo, up, SC[s], 0.5);
    pp.line(TCA, Array.from(mm[s].m), SC[s], 2);
  }
  legend(pp, SNAME.map((n,s)=>({label:'stim '+n, color:SC[s]})));
}

/* ---- level 3 */
function drawTrial(){
  const pt = P.trial, pv = P.volt, trial = state.trial;
  pt.clear(); pv.clear();
  if (trial === null){
    pt.placeholder('click a trial row in the raster above');
    pv.placeholder('');
    CAP.trial.textContent = ''; CAP.volt.textContent = '';
    return;
  }
  const s = STIM[trial];
  const dlo = -PITCH/2, dhi = (NCH-1)*PITCH + PITCH/2;

  const rows = [];
  for (let k = 0; k < NN; k++){
    const i = BY_DEPTH[k];
    rows.push({t:spikes(i, trial), y:DEPTH[i], color:NCOL[i]});
  }
  pt.limits(-PRE, POST, dlo, dhi);
  pt.frame({xlabel:'time from stimulus onset (s)', ylabel:'depth on probe (µm)',
            title:`Trial ${trial} (stim ${SNAME[s]}): all ${NN} neurons\ngrey band = neuron ${state.neuron}`,
            titleColor:SC[s]});
  pt.hspan(DEPTH[state.neuron], 9, 'rgba(0,0,0,.10)');
  pt.vspan(state.voltX[0], state.voltX[1], 'rgba(90,95,102,.16)');
  pt.vline(0, '#999', 1);
  pt.raster(rows, 6.5, 1.2);
  CAP.trial.textContent = 'Click to move the voltage window (grey band).';

  const {v, nSamp} = trialVoltage(trial);
  const [x0, x1] = state.voltX;
  pv.limits(x0, x1, dlo, dhi);
  pv.image((buf, w, h) => {
    // Average the samples that land in each output column. Taking the most
    // extreme sample instead makes waveforms pop, but it also pushes the
    // noise distribution outward and the image ends up far harsher than the
    // matplotlib version this is meant to match.
    const col = new Float32Array(NCH * w);
    for (let px = 0; px < w; px++){
      const ta = x0 + px / w * (x1 - x0), tb = x0 + (px + 1) / w * (x1 - x0);
      let s0 = Math.floor((ta + PRE) * FS), s1 = Math.ceil((tb + PRE) * FS);
      if (s0 < 0) s0 = 0;
      if (s1 > nSamp) s1 = nSamp;
      if (s1 <= s0) continue;
      const n = s1 - s0;
      for (let c = 0; c < NCH; c++){
        const row = c * nSamp;
        let acc = 0;
        for (let k = s0; k < s1; k++) acc += v[row + k];
        col[c * w + px] = acc / n;
      }
    }
    for (let py = 0; py < h; py++){
      const depth = dhi - (py + 0.5) / h * (dhi - dlo);
      let c = Math.round(depth / PITCH);
      if (c < 0) c = 0; else if (c >= NCH) c = NCH - 1;
      for (let px = 0; px < w; px++){
        let g = (col[c * w + px] + CLIM) / (2 * CLIM) * 255;
        g = g < 0 ? 0 : g > 255 ? 255 : g;
        const o = (py * w + px) * 4;
        buf[o] = buf[o+1] = buf[o+2] = g; buf[o+3] = 255;
      }
    }
  });
  pv.frame({xlabel:'time from stimulus onset (s)', ylabel:'depth on probe (µm)',
            title:`Raw voltage, ${NCH} channels (±${CLIM} µV grey scale)\ndots = spikes, coloured by neuron`});
  if (state.showDots){
    const pts = [];
    for (let i = 0; i < NN; i++){
      const t = spikes(i, trial);
      for (let k = 0; k < t.length; k++)
        if (t[k] >= x0 && t[k] <= x1) pts.push({x:t[k], y:DEPTH[i], color:NCOL[i]});
    }
    pv.dots(pts, 4);
  }
  CAP.volt.textContent =
    'Scroll to zoom · click to re-centre · press "d" to '
    + (state.showDots ? 'hide' : 'show') + ' the dots';
}

function redraw(){ drawSummary(); drawMatrix(); drawNeuron(); drawTrial(); }

/* ---------------------------------------------------------- interaction */
function local(plot, ev){
  const r = plot.c.getBoundingClientRect();
  return [ev.clientX - r.left, ev.clientY - r.top];
}
function markClickable(){
  document.getElementById('p-mat').classList.toggle('clickable', state.stim !== null);
  document.getElementById('p-ras').classList.toggle('clickable', state.neuron !== null);
  document.getElementById('p-trial').classList.toggle('clickable', state.trial !== null);
  document.getElementById('p-volt').classList.toggle('clickable', state.trial !== null);
}

P.sum.c.addEventListener('click', ev => {
  const p = P.sum, [px, py] = local(p, ev);
  if (!p.inside(px, py)) return;
  const xd = p.invX(px), yd = p.invY(py);
  let best = -1, bestD = Infinity;
  for (let s = 0; s < NS; s++){
    let b = Math.round((xd + PRE) / BIN - 0.5);
    b = Math.max(0, Math.min(NB-1, b));
    const d = Math.abs(POP[s].m[b] - yd);
    if (d < bestD){ bestD = d; best = s; }
  }
  if (bestD > RAW.hitTol * Math.abs(p.yl[1] - p.yl[0])) return;
  state.stim = best; state.neuron = null; state.trial = null;
  drawMatrix(); drawNeuron(); drawTrial(); markClickable();
});

P.mat.c.addEventListener('click', ev => {
  if (state.stim === null) return;
  const p = P.mat, [px, py] = local(p, ev);
  if (!p.inside(px, py)) return;
  const i = Math.round(p.invY(py));
  if (i < 0 || i >= NN) return;
  state.neuron = i; state.trial = null;
  drawNeuron(); drawTrial(); markClickable();
});

P.ras.c.addEventListener('click', ev => {
  if (state.neuron === null) return;
  const p = P.ras, [px, py] = local(p, ev);
  if (!p.inside(px, py)) return;
  const row = Math.round(p.invY(py));
  if (row < 0 || row >= state.rowTrial.length) return;
  state.trial = state.rowTrial[row];
  state.voltX = [RAW.voltCentre - RAW.voltWidth/2, RAW.voltCentre + RAW.voltWidth/2];
  drawTrial(); markClickable();
});

function recentre(t){
  const w = state.voltX[1] - state.voltX[0];
  state.voltX = [t - w/2, t + w/2];
  drawTrial();
}
P.trial.c.addEventListener('click', ev => {
  if (state.trial === null) return;
  const p = P.trial, [px, py] = local(p, ev);
  if (p.inside(px, py)) recentre(p.invX(px));
});
P.volt.c.addEventListener('click', ev => {
  if (state.trial === null) return;
  const p = P.volt, [px, py] = local(p, ev);
  if (p.inside(px, py)) recentre(p.invX(px));
});
P.volt.c.addEventListener('wheel', ev => {
  if (state.trial === null) return;
  ev.preventDefault();
  const p = P.volt, [px] = local(p, ev);
  const anchor = p.invX(px), f = ev.deltaY > 0 ? 1.25 : 0.8;
  let [a, b] = state.voltX;
  let w = (b - a) * f;
  w = Math.max(0.004, Math.min(PRE + POST, w));
  const frac = (anchor - a) / (b - a);
  state.voltX = [anchor - frac * w, anchor + (1 - frac) * w];
  drawTrial();
}, {passive:false});

window.addEventListener('keydown', ev => {
  if (ev.key === 'd' && state.trial !== null){
    state.showDots = !state.showDots;
    drawTrial();
  }
});

// Canvases are sized by CSS, so their pixel buffers have to follow the
// layout rather than the window: a ResizeObserver catches the first layout
// pass (which the inline script beats) as well as later container resizes.
/* ---- normalisation selectors */
function fillSelect(el, options, value){
  el.innerHTML = '';
  for (const o of options){
    const opt = document.createElement('option');
    opt.value = o; opt.textContent = o;
    el.appendChild(opt);
  }
  el.value = value;
}
const selTrial = document.getElementById('sel-trial');
const selNeuron = document.getElementById('sel-neuron');
const selStat = document.getElementById('sel-stat');
const floorNote = document.getElementById('floornote');
fillSelect(selTrial, RAW.trialNorms, CFG.trialNorm);
fillSelect(selNeuron, RAW.neuronNorms, CFG.neuronNorm);
fillSelect(selStat, RAW.stats, CFG.stat);

function updateFloorNote(){
  const bits = [];
  if (N_FLOORED)
    bits.push(`${N_FLOORED} of ${NN * NT} per-trial baselines`);
  if (N_FLOORED_NEURONS)
    bits.push(`${N_FLOORED_NEURONS} of ${NN} per-neuron baselines`);
  floorNote.textContent = bits.length
    ? 'divide-by-almost-zero floor applied to ' + bits.join(' and ')
    : '';
}

function onConfigChange(){
  CFG.trialNorm = selTrial.value;
  CFG.neuronNorm = selNeuron.value;
  CFG.stat = selStat.value;
  // The median path bootstraps, which takes a moment; let the browser paint
  // the disabled controls before we block on it.
  for (const el of [selTrial, selNeuron, selStat]) el.disabled = true;
  setTimeout(() => {
    configure();
    updateFloorNote();
    redraw();               // level 3 is spikes and voltage; unaffected but cheap
    for (const el of [selTrial, selNeuron, selStat]) el.disabled = false;
  }, 0);
}
for (const el of [selTrial, selNeuron, selStat])
  el.addEventListener('change', onConfigChange);

let rt = null;
function scheduleRedraw(){
  clearTimeout(rt);
  rt = setTimeout(redraw, 80);
}
const ro = new ResizeObserver(scheduleRedraw);
ro.observe(document.querySelector('.grid'));
window.addEventListener('resize', scheduleRedraw);

configure();
updateFloorNote();
redraw();
markClickable();
</script>
</body>
</html>
"""


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    S, payload = build_payload()
    html = PAGE.replace('__DATA__', json.dumps(payload, separators=(',', ':')))
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    kb = os.path.getsize(OUT) / 1024
    n_in = len(base64.b64decode(payload['spT'])) // 4
    print(f'wrote {OUT}  ({kb:.0f} KB)')
    print(f'  {n_in} in-window spikes embedded '
          f'(of {S.spike_time.size} in the session)')


if __name__ == '__main__':
    main()
