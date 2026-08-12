# Drill-down on real data: orientation tuning in mouse V1

The same click-your-way-down figure as
[`plotting_drilldown_demo`](../plotting_drilldown_demo), but over **real Allen
Institute Visual Coding – Neuropixels data** instead of a simulation.

**[Open it →](https://allen-v1-orientation-drilldown.netlify.app)**

![The population average and the units behind it](preview.png)

## The result worth showing a class

The population average across all 110 units is **almost flat** — the eight
grating directions sit on top of each other between about 4 and 6 spikes/s.
It looks like V1 barely cares about orientation.

One click down, that story falls apart. **74 of the 110 units have an
orientation selectivity index above 0.3** (median OSI 0.63), and their
preferred directions are spread evenly across all eight (9–21 units each). Unit
31 fires at 60 spikes/s for 90° and 270° — the same orientation, both
directions — and is essentially silent for the other six.

The tuning is not weak. It cancels. Averaging over neurons that prefer
different things gives you a flat line, and the flat line is what gets
published.

## What was used

| | |
|---|---|
| Session | `732592105` (`brain_observatory_1.1`, wild-type male, P100) |
| Probe | `733744649` (probeC) |
| Region | `VISp`, primary visual cortex |
| Units | 110 passing Allen's default QC |
| Stimulus | `drifting_gratings`, 8 directions × 5 temporal frequencies × 15 repeats |
| Conditions | the 8 directions, pooling across temporal frequency (~75 trials each) |
| Window | −0.5 to +2.5 s around onset; the grating is on for 2.0 s |

This session/probe pair was picked mechanically: it has the most good V1 units
of any `brain_observatory_1.1` probe. QC is Allen's default — `quality = good`,
amplitude cutoff < 0.1, presence ratio > 0.95, ISI violations < 0.5.

Because trials pool across temporal frequency (1–15 Hz), the averages carry a
residual ripple from units phase-locking to the grating. That is real signal,
not a plotting artifact.

## The levels

| Panel | Shows | Click |
|---|---|---|
| top-left | 8 average PSTHs, mean ± SEM across units | a **trace**, or its **legend entry** (easier with 8 overlapping traces) |
| top-right | units × time for that direction | a **row** |
| middle | that unit's raster for all 598 trials, and its 8 PSTHs | a **trial row** |
| bottom | that trial: all 110 units by depth on the probe | — |

Rows in the units × time panel are ordered **by depth, not by response** —
sorting by response would manufacture a diagonal, which is what
[`nb3`](../nb3) is about.

The normalization and mean/median selectors from the synthetic version are all
here, and they misbehave more interestingly on real data, because many V1 units
have baselines near zero.

## What is missing

**The raw voltage level.** Allen does publish continuous spike-band traces, but
their S3 bucket returns **no `Access-Control-Allow-Origin` header** — verified,
not assumed — so a browser cannot read it cross-origin. Ranged GETs work fine
from Python (`206`, `Accept-Ranges: bytes`), so the route is to pre-extract
snippets and host them next to the page. See
[`REAL_DATA_PLAN.md`](../plotting_drilldown_demo/REAL_DATA_PLAN.md) §5b.

## Rebuilding

```bash
python build_data.py     # NWB -> web/data.bin   (needs h5py)
python check_data.py     # sanity check + check_tuning.png
python build_html.py     # -> web/index.html
```

`build_data.py` expects the session NWB at
`D:\temp\allen_drilldown\session_732592105.nwb` (2.9 GB). It is not in the
repo. Fetch it anonymously — no AllenSDK, no credentials:

```bash
curl -o session_732592105.nwb https://allen-brain-observatory.s3.us-west-2.amazonaws.com/visual-coding-neuropixels/ecephys-cache/session_732592105/session_732592105.nwb
```

`web/data.bin` (2.3 MB) **is** committed, so the page can be rebuilt and
redeployed without the 2.9 GB download.

## The payload format

`data.bin` is a `uint32` header length, a JSON header, then raw little-endian
arrays at 8-byte-aligned offsets recorded in the header — so the browser wraps
them as typed-array views with no copying. Only spikes inside a trial window
are included, stored relative to the window as `uint16` (46 µs resolution,
far finer than the 25 ms display bin). 1,011,368 spikes fit in 2.3 MB.
