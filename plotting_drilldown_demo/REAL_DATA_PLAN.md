# What it would take to run this on real data

Scoping notes for pointing the drill-down at Allen Institute Visual Coding –
Neuropixels or IBL public data, with an emphasis on the **web version**, since
that is where the constraints actually bite.

The short answer: **levels 0–2 are easy, and level 3 is more achievable than I
expected** — the Allen raw spike-band files are flat interleaved int16, which
means a browser can HTTP-Range-request exactly the 150 ms × 384-channel window
it needs, with no decompression at all. The main open question is CORS, not
data volume.

---

## 1. The thing that changes everything: you can't embed the data

The synthetic demo embeds all 44,217 in-window spikes as base64 in a 296 KB
HTML file. One real Allen session is ~500–1000 curated units and hundreds to
thousands of trials, and the raw voltage for one probe is tens to hundreds of
GB. So the page has to **fetch**, and the design question becomes: what is
precomputed and small, and what is fetched on demand?

That splits cleanly along the levels:

| Level | What it needs | Size for one Allen session | Delivery |
|---|---|---|---|
| 0, 1 | PSTH per unit per condition | 1000 units × 3 cond × 120 bins × f32 ≈ **1.4 MB** (≈19 MB if you expose all 40 drifting-grating conditions) | one gzipped fetch on load |
| 2 | one unit's spikes across all trials | ~6 KB per unit | Range request into a blob sorted by unit |
| 3 (raster) | all units' spikes on one trial | ~24 KB per trial | Range request into a blob sorted by trial |
| 3 (voltage) | 150 ms × 384 ch of raw | **3.46 MB** contiguous | Range request into `spike_band.dat` |

Store the trial-windowed spikes **twice** — once ordered by unit, once by
trial — with a `uint32` offset index for each. Storage roughly doubles (to
~15 MB for a session) and both level-2 and level-3 views become a single range
request instead of a scan. Quantise relative spike times to `uint16` over the
trial window (1.2 s / 65535 ≈ 18 µs resolution, far finer than a PSTH bin) and
that halves again.

Levels 0–2 therefore need **no new science**, just a preprocessing script and a
fetch layer. `drilldown_core.py` already computes everything; it would emit
static binaries instead of holding arrays.

---

## 2. Allen Institute Visual Coding – Neuropixels

**Access.** AllenSDK `EcephysProjectCache`, or straight from S3. 58 sessions,
6 probes each, ~99k units. Session NWBs (spike times, units table with QC
metrics, stimulus presentation tables) total ~146 GB; LFP NWBs ~707 GB.

**The good surprise.** I assumed raw AP-band was not distributed. It is. The
`allen-brain-observatory` bucket carries *"continuous voltage traces for every
Neuropixels experiment"* at:

```
s3://allen-brain-observatory/visual-coding-neuropixels/raw-data/<session_id>/<probe_id>/spike_band.dat
```

`spike_band.dat` is a flat, channel-interleaved int16 file at 30 kHz. That
format is close to ideal for a web app:

- A time window is **contiguous bytes**, because samples interleave channels.
  Byte offset = `sample_index × n_channels × 2`.
- **No decompression.** A plain `Range:` header gets you exactly the window.
- 150 ms × 30 kHz × 384 ch × 2 B = **3.46 MB per click.** A second or two on a
  decent connection, and cacheable.

So the browser can pull genuine Allen voltage and render it with the same
canvas code the demo already has. You'd fetch all 384 channels for those
samples (they're interleaved, no way around it) and subset channels
client-side — 3.46 MB is fine.

**Stimulus mapping.** Drifting gratings (8 directions × 5 temporal
frequencies × 15 reps), static gratings, natural scenes, natural movies,
flashes. Pick any three conditions to stand in for A/B/C, or expose a condition
picker. Stimulus times in the NWB are already aligned to the spike clock.

**Waveforms.** `session.mean_waveforms` gives each unit's mean waveform across
channels. Even without raw data this supports a **reconstruction** level 3
(§4b) at trivial cost: keep the ~20 channels around each unit's peak and you
need only 1000 × 20 × 82 × int16 ≈ **3.3 MB** for a whole session.

---

## 3. IBL public data

**Access.** ONE / Alyx, with data on FlatIron and AWS. The Brain Wide Map
release is ~700 sessions. Raw AP band **is** public.

**Format.** Three files per probe: `.cbin` (losslessly compressed raw binary),
`.meta` (SpikeGLX), and `.ch` (compression header holding the byte address of
every chunk). Compression is **1-second chunks**, and the `.ch` index is
exactly what makes partial reads possible. IBL ships a `Streamer` object
(`brainbox.io.spikeglx`) that retrieves only the chunks covering a requested
time range — you never uncompress the whole file.

**For Python, this is strictly nicer than Allen.** For the browser it is
harder:

- You'd need mtscomp decoding in JS: inflate the chunk, then cumulative-sum
  (the format stores int16 *differences* along time). Modern browsers give you
  zlib free via `DecompressionStream('deflate')`, so this is genuinely
  feasible — but it is real work and needs careful validation against the
  Python reader.
- Granularity is 1 s minimum: ~23 MB uncompressed, maybe 8–12 MB over the
  wire, per click. Versus 3.46 MB for Allen.

**Where IBL wins:** the task is a decision task, so trials carry choice,
contrast, side, reaction time and block. The drill-down could branch on
*behaviour*, not just stimulus identity — a much richer teaching object than
A/B/C. IBL is also the better fit if you want the demo to connect to the
pseudosession/block material already in `nb4_walkthrough`.

**Recommendation:** Allen for the web version, IBL if you want behaviour.

---

## 4. Level 3, in increasing order of fidelity and cost

**(a) Drop it.** Stop at the trial raster. Zero cost, keeps 3 of 4 levels, and
the biggest pedagogical drop.

**(b) Reconstruct from real mean waveforms.** Place each unit's *actual* mean
waveform at its *actual* spike times on its *actual* channels, over noise
matched to the recording. This is exactly what the demo's JS already does, with
real templates instead of Gaussians. ~3.3 MB per session, works offline, no
CORS, no range requests. **Must be labelled a reconstruction** — but it is
honest and it looks right. *Best effort-to-payoff ratio.*

**(c) Pre-extract real snippets server-side.** Choose the trials you want
clickable (say 100), cut a 150 ms × 64-channel window around each, scale to
int8: 150 ms × 30 kHz × 64 × 1 B ≈ **288 KB/trial → ~29 MB for 100 trials**.
Static files, one range request each, no live dependency on anyone's bucket.
**For a course this is probably the right answer** — deterministic, fast, and
it survives a lecture hall with bad wifi.

**(d) Live range requests against Allen S3.** The real thing, any trial, any
session. 3.46 MB/click, no decompression. Gated entirely on §6.

**(e) Live streaming from IBL.** Same, but requires the mtscomp decoder in JS.
1–2 weeks on its own, and the least certain.

---

## 5. What gets scientifically harder (and more interesting)

These are not implementation details; they change what the demo *teaches*.

- **The dots stop being ground truth.** In the synthetic demo, every coloured
  dot sits exactly on the waveform that produced it, because we put it there.
  On real data the dots are **spike-sorter output** — the very thing that might
  be wrong. You will see merges, splits, and missed spikes. That is a *better*
  lesson than the synthetic version, and it should be said out loud in the UI.
- **Curation becomes visible.** Which units are in the average? Allen's
  `quality == 'good'`, ISI violations, amplitude cutoff, presence ratio. The
  level-1 matrix makes the consequences of that filter obvious. Worth a
  selector of its own, in the same spirit as the normalisation ones.
- **Pooling across sessions/mice** turns the level-0 average into a live
  instance of `nb7` (pseudo-replication — the unit of analysis is the session)
  and `nb5` (Simpson's paradox). Default to one session; make cross-session
  pooling an explicit, labelled choice.
- **Unequal trial counts** across conditions change the error bars.
- **Drift** means the level-3 depth axis is only approximate over a long
  session.
- **Real baselines are often near zero**, so the divide-by-baseline selectors
  that already exist will misbehave far more dramatically than they do on
  synthetic data. That is the `nb6` Jensen lesson with real teeth.

---

## 5b. What CORS is, and the "just host it ourselves" option

**CORS in one paragraph.** When a page served from one domain (say
`drilldown-psth-to-voltage.netlify.app`) asks the browser to fetch data from a
*different* domain (say `s3.amazonaws.com`), the browser blocks the read unless
the second server explicitly opts in with a response header
(`Access-Control-Allow-Origin`). It is a rule enforced **by browsers only** —
Python, `curl`, `wget` and `boto3` ignore it entirely. So CORS can never affect
the standalone script or the notebook. It only decides whether the *web page*
may read Allen's bucket directly. Data can be fully public and still be
unreadable from a web page, purely because nobody set that header.

**So yes: hosting a copy ourselves is a clean way around it, and it is probably
what we should do anyway.** If the bytes come from a domain we control, we set
the header and the question disappears.

**And we would not need to download a whole probe.** This is the part worth
knowing: because `spike_band.dat` is flat int16, the *same* byte-range trick
that would let a browser grab one window also works from Python with `boto3`.
So the offline extraction step can pull only the snippets it wants:

| What | Size |
|---|---|
| One whole probe's raw file | 30 kHz × 384 ch × 2 B ≈ **23 MB/s** → a few hundred GB for a multi-hour session |
| 100 trials × 150 ms × 384 ch, range-requested | **~350 MB** |
| Same, cropped to 64 channels and scaled to int8 | **~29 MB** |
| Levels 0–2 for the whole session (PSTHs + both spike orderings) | **~20 MB** |

So a complete, self-hosted, real-data version of all four levels is **well
under 1 GB** — no bulk download, no storage conversation. Downloading a full
probe (a few hundred GB) only buys the ability to cut a snippet for *any*
trial after the fact, which is not worth it for a course.

Where to host: an S3 bucket or Cloudflare R2 we own (R2 has no egress fees, and
both let us set CORS), or Netlify itself for the small assets. Netlify's free
tier includes 100 GB/month of bandwidth, which a ~300 MB payload comfortably
fits unless the page gets popular.

## 6. Verify these first — they decide the architecture

A half-day spike, before writing anything:

1. **Does `allen-brain-observatory` return CORS headers on an anonymous
   ranged GET?** This single fact decides whether option (d) exists. Test with
   a browser `fetch` with a `Range` header from a different origin, not with
   curl — curl will succeed even when a browser is blocked.
2. **Is the bucket requester-pays?** The AllenSDK AWS wiki says you "need to
   have an AWS account," which hints that anonymous access may not be
   available. If it is requester-pays, (d) is dead in the browser and you fall
   back to (c) — or proxy through a Netlify function, which then puts the
   egress on your account.
3. **Same two questions for the IBL public bucket.**
4. **Confirm the exact `spike_band.dat` layout** — channel count, whether the
   sync channel is included, sample order, and the µV-per-bit scaling — against
   a snippet you also read with the SDK. Getting this wrong yields a plausible
   but wrong picture, which is the worst outcome.
5. **Confirm sample-clock alignment** between `spike_band.dat` sample index and
   the NWB stimulus times.

If 1 and 2 come back unfavourable, go straight to (c). Nothing else changes.

---

## 7. Effort

**What the numbers below mean.** They are conventional person-day estimates
for one developer who already knows this codebase — the usual unit for a
planning doc, and useful if this ever gets handed to a student. They are *not*
an estimate of how long it takes if Claude writes it: the code-writing
collapses (the three-version demo in this folder was built in a single
session). What does **not** collapse is everything else — waiting on downloads
and compute, the verification loop in §6, and the back-and-forth where a human
looks at a figure and says "that's wrong." Realistically, with Claude doing
the implementation, expect **a session or two per phase**, with phases 0 and 1
likely finishing in one sitting, and the schedule set by data transfer and
review rather than by typing.

| Phase | Work | Estimate |
|---|---|---|
| 0 | Verification spike (§6) | 0.5–1 day |
| 1 | Preprocessing: AllenSDK → QC filter → PSTHs → static binaries + indices | 2–3 days |
| 2 | Front ends fetch instead of embed; loading states; 1000-row level 1 needs scroll/sort/filter; condition picker | 2–3 days |
| 3b | Level 3 via waveform reconstruction | 1–2 days |
| 3c | Level 3 via pre-extracted real snippets | 1–2 days + a bulk download and compute run |
| 3d | Level 3 via live Allen range requests | 2–4 days *if* §6 passes |
| 3e | Level 3 via IBL mtscomp-in-JS | 1–2 weeks, least certain |
| 4 | Teaching polish: curation selector, honesty copy, session picker | 1–2 days |

**A good target: ~1 week** of person-time for a real-Allen-data version with
pre-extracted voltage snippets (phases 0, 1, 2, 3c) — which for a course is
likely better than live streaming anyway, and which sidesteps CORS entirely by
hosting the snippets ourselves (§5b).
**~2 weeks** if live range requests work and you want any-trial access.

The Python standalone and notebook versions are much cheaper throughout: they
can call AllenSDK or ONE directly and skip the entire static-asset and CORS
problem. If the goal is "show this in class on real data," **do the Python
version first** — it is a day or two of work on top of what exists, because
`drilldown_core.py` only needs a different `Session` loader.
