"""
Build drilldown_notebook.ipynb -- the Jupyter version of the drill-down demo.

The standalone script opens a new window per level. A notebook cannot do that,
so the notebook version puts all four levels in ONE ipympl figure as a 3x2
grid of panels that fill in as you click:

    [ level 0 summary   ] [ level 1 neurons x time ]
    [ level 2 raster    ] [ level 2 PSTHs          ]
    [ level 3 raster    ] [ level 3 raw voltage    ]

The notebook is written WITHOUT outputs on purpose: the output of an ipympl
cell is a live widget, which does not survive being serialised to a file and
renders on GitHub as an empty box. You have to run it.

Run:  python build_notebook.py
"""

import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'drilldown_notebook.ipynb')

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
co = lambda s: cells.append(new_code_cell(s))

# ===========================================================================
md("""\
# Drill-down: from the population average to the raw voltage

This notebook starts with the figure everyone publishes — the average PSTH
across all neurons, one trace per stimulus — and lets you **click your way down
to the data it was computed from.**

| Panel | Shows | What you click |
|---|---|---|
| top-left | three average PSTHs, mean ± SEM across neurons | a **trace** |
| top-right | neurons × time: every neuron's PSTH for that stimulus | a **row** |
| middle | that neuron's raster for all trials of all three stimuli, and its three average PSTHs | a **trial row** (left) |
| bottom | that trial: all neurons (y = depth), and the raw voltage on the probe with a dot on every spike | the raster, to move the voltage window |

**Run every cell, then click the top-left panel.** The lower panels start empty
and fill in as you go. Clicking higher up clears everything below, so you can
never be looking at a mismatched set of panels.

> Needs `ipympl` (`pip install ipympl`) for the interactive canvas — the same
> dependency as `nb4_walkthrough`. With the default inline backend the figure
> renders once as a static image and clicking does nothing.
""")

# ---------------------------------------------------------------------------
co("""\
%matplotlib widget""")

co("""\
import os
import numpy as np
import matplotlib.pyplot as plt

import generate_data as gd
import drilldown_core as core

core.apply_style()

# The dataset is ~0.6 MB and regenerates in a few seconds if it is missing.
DATA = os.path.join('data', 'dataset.npz')
if not os.path.exists(DATA):
    gd.main()

S = core.Session(DATA)
print(f'{S.n_neurons} neurons, {S.n_trials} trials '
      f'({len(S.trials_of[0])} per stimulus), '
      f'{S.spike_time.size} spikes')""")

# ---------------------------------------------------------------------------
md("""\
## What is in the data

50 neurons and 120 trials — 40 each of stimuli A, B and C, in random order.
Every neuron has a response amplitude to each stimulus drawn from a Gaussian
whose mean depends on the stimulus (A ≈ 14, B ≈ 6, C ≈ 5 spikes/s), so **A is
bigger on average** but individual neurons vary a lot and some are suppressed.
The response time course is a gamma-shaped kernel — fast rise, slower decay,
peaked at ~42 ms and essentially over within 200 ms.

Raw voltage is *not* stored; it is simulated on demand for whichever trial you
click, from that trial's spike times. Each spike contributes a negative 2-D
Gaussian (in time × depth) followed by a wider-in-time positive one, on top of
white noise. So the coloured dots are **ground truth**, not the output of a
spike sorter: every dot sits exactly on the waveform that produced it.
""")

# ---------------------------------------------------------------------------
md("""\
## The clickable figure

Everything below is bookkeeping: build the six panels, then translate a click
in one panel into a redraw of the panels beneath it. The actual plotting lives
in `drilldown_core.py` so that this notebook and the standalone
`drilldown.py` draw exactly the same things.
""")

co('''\
fig, axes = plt.subplots(3, 2, figsize=(10.5, 11.5))
(ax_sum, ax_mat), (ax_ras, ax_psth), (ax_trial, ax_volt) = axes
fig.canvas.header_visible = False
fig.subplots_adjust(left=0.085, right=0.86, top=0.955, bottom=0.05,
                    hspace=0.50, wspace=0.30)

# A dedicated slot for the level-1 colourbar. It has to hang off the FIGURE,
# not off ax_mat: inset axes are children of their parent, and the parent's
# .clear() on every redraw would take the colourbar with it.
_p = ax_mat.get_position()
cax = fig.add_axes([_p.x1 + 0.012, _p.y0, 0.014, _p.height])

# `state` remembers what each level is currently showing; clicking higher up
# throws away everything below so the panels can never disagree.
state = {}

PROMPTS = {
    1: (ax_mat,   'click a trace in the panel to the left'),
    2: (ax_ras,   'click a row in the neurons × time panel'),
    3: (ax_trial, 'click a trial row in the raster above'),
}


def clear_from(level):
    """Blank level `level` and everything below it."""
    for lv in range(level, 4):
        ax, msg = PROMPTS[lv]
        core.placeholder(ax, msg)
        state.pop(lv, None)
    if level <= 1:
        cax.clear()
        cax.set_axis_off()
    if level <= 2:
        core.placeholder(ax_psth, '')
    if level <= 3:
        core.placeholder(ax_volt, '')


core.draw_summary(ax_sum, S)
clear_from(1)


def on_click(ev):
    if ev.inaxes is None or ev.xdata is None:
        return
    if fig.canvas.toolbar is not None and fig.canvas.toolbar.mode:
        return              # the toolbar is in pan/zoom mode; not our click

    if ev.inaxes is ax_sum:                                   # -> level 1
        s = core.hit_summary(ax_sum, S, ev.xdata, ev.ydata)
        if s is None:
            return
        clear_from(2)
        core.draw_neuron_matrix(ax_mat, S, s, cax=cax)
        state[1] = s

    elif ev.inaxes is ax_mat and 1 in state:                  # -> level 2
        i = core.hit_neuron_matrix(S, ev.ydata)
        if i is None:
            return
        clear_from(3)
        state[2] = (i, core.draw_neuron(ax_ras, ax_psth, S, i))

    elif ev.inaxes is ax_ras and 2 in state:                  # -> level 3
        neuron, row_trial = state[2]
        tr = core.hit_neuron(row_trial, ev.ydata)
        if tr is None:
            return
        state[3] = core.draw_trial(ax_trial, ax_volt, S, tr, neuron)

    elif ev.inaxes is ax_trial and 3 in state:      # re-centre the voltage
        half = np.diff(ax_volt.get_xlim())[0] / 2
        ax_volt.set_xlim(ev.xdata - half, ev.xdata + half)

    else:
        return
    fig.canvas.draw_idle()


def on_key(ev):
    """'d' hides the spike dots, so you can see the waveforms unobscured."""
    if ev.key == 'd' and 3 in state:
        for dot in state[3][1]:
            dot.set_visible(not dot.get_visible())
        fig.canvas.draw_idle()


fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.draw_idle()''')

# ---------------------------------------------------------------------------
md("""\
### Notes on using it

- The bottom-right voltage panel opens zoomed to a **150 ms window** around
  stimulus onset. A spike waveform is ~2 ms wide — at full-trial scale it is
  thinner than a pixel. The grey band on the raster to its left marks exactly
  which slice you are looking at.
- **Click the bottom-left raster** to move that window, or use the toolbar's
  pan/zoom. Press **`d`** (with the figure focused) to hide the dots.
- The first click into any given trial simulates ~6 MB of voltage and takes a
  moment; the last few trials you visited are cached.

### The point

The top-left panel is true, and nearly uninformative about what any neuron did.
The panel beside it shows the average is carried by a minority of strongly
responsive cells. The middle row shows how few spikes per trial that actually
is. The bottom row shows what a "spike" was in the first place.

Every summary statistic you compute sits on a stack like this one. It is worth
knowing how far down it goes.
""")

# ===========================================================================
nb = new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                   'name': 'python3'},
    'language_info': {'name': 'python'},
})

with open(OUT, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f'wrote {OUT}  ({len(cells)} cells, unexecuted by design)')
