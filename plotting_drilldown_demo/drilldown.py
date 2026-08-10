#!/usr/bin/env python
"""
Interactive drill-down from a population average to the raw voltage.

Start with the figure everyone publishes -- the average PSTH across all
neurons, one trace per stimulus -- and click your way down to the data it was
computed from:

  level 0  three average PSTHs (mean +/- SEM across neurons)
              click a trace
  level 1  neurons x time: every neuron's PSTH for that stimulus
              click a row
  level 2  one neuron: spike raster for all trials of all three stimuli,
           plus that neuron's three average PSTHs
              click a trial row
  level 3  one trial: raster of all neurons (y = depth) next to the raw
           voltage on the probe (channels x time), with a coloured dot on
           every spike

The level-0 window also carries three selectors: normalise each trial before
averaging trials, normalise each neuron before averaging neurons, and average
with the mean or the median. Changing any of them rebuilds levels 0-2 in
place, so you can watch the published figure change shape.

Each level opens its own window. Closing or pressing Escape on a window also
closes everything below it.

Run:  python drilldown.py            (regenerates data/dataset.npz if missing)
      python drilldown.py --regen    (force regeneration)
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_data as gd
import drilldown_core as core

core.apply_style()


# ----------------------------------------------------------------------
# Window stack. Each level records how to redraw itself, so that changing a
# normalisation selector can refresh the open windows instead of closing them.
# ----------------------------------------------------------------------
FIGS = {}


def close_from(level):
    for lv in sorted([lv for lv in FIGS if lv >= level], reverse=True):
        entry = FIGS.pop(lv, None)
        if entry is not None:
            plt.close(entry['fig'])


def refresh_from(level):
    """Redraw open windows at `level` and below, in place."""
    for lv in sorted(lv for lv in FIGS if lv >= level):
        entry = FIGS[lv]
        entry['redraw']()
        entry['fig'].canvas.draw_idle()


def place_window(fig, x, y):
    """Best-effort window positioning; silently does nothing if unsupported."""
    try:
        win = fig.canvas.manager.window
        if hasattr(win, 'wm_geometry'):
            win.wm_geometry(f"+{x}+{y}")
        elif hasattr(win, 'move'):
            win.move(x, y)
    except Exception:
        pass


def register(level, fig, redraw):
    FIGS[level] = {'fig': fig, 'redraw': redraw}

    def on_close(ev, lv=level, f=fig):
        close_from(lv + 1)
        if FIGS.get(lv, {}).get('fig') is f:
            FIGS.pop(lv)

    fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.mpl_connect(
        'key_press_event',
        lambda ev, f=fig: plt.close(f) if ev.key == 'escape' else None)
    place_window(fig, 60 + 90 * level, 60 + 60 * level)
    fig.show()


def footer(fig, text):
    """One line of usage hints along the bottom of a figure."""
    return fig.text(0.012, 0.012, text, fontsize=9, style='italic',
                    color='0.35', ha='left', va='bottom')


FOOTER_RECT = (0, 0.055, 1, 1)   # tight_layout rect leaving room for footer


def idle(fig):
    """True unless the toolbar is in pan/zoom mode (then clicks aren't ours)."""
    return not (fig.canvas.toolbar is not None and fig.canvas.toolbar.mode)


# ----------------------------------------------------------------------
# Level 0 -- the summary, plus the normalisation controls
# ----------------------------------------------------------------------
def _radio(fig, cell, title, labels, active, on_change):
    ax = fig.add_subplot(cell)
    ax.set_title(title, fontsize=9, color='0.25', pad=5)
    ax.set_facecolor('#f2f4f7')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    w = RadioButtons(ax, labels, active=active)
    for t in w.labels:
        t.set_fontsize(8.5)
    w.on_clicked(on_change)
    return w


def show_summary(S):
    fig = plt.figure(figsize=(10.6, 6.4))
    fig.canvas.manager.set_window_title('0: population average')
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.4],
                          width_ratios=[1.15, 1.15, 0.7],
                          left=0.075, right=0.975, top=0.905, bottom=0.105,
                          hspace=0.34, wspace=0.10)
    ax = fig.add_subplot(gs[1, :])

    note = fig.text(0.5, 0.965, '', fontsize=8.5, style='italic',
                    color='#a33', ha='center', va='center')

    def apply(_=None):
        S.configure(trial_norm=r_tn.value_selected,
                    neuron_norm=r_nn.value_selected,
                    stat=r_st.value_selected)
        bits = []
        if S.n_floored:
            bits.append(f'{S.n_floored} of {S.n_neurons * S.n_trials} '
                        f'per-trial baselines')
        if S.n_floored_neurons:
            bits.append(f'{S.n_floored_neurons} of {S.n_neurons} '
                        f'per-neuron baselines')
        note.set_text(
            'divide-by-almost-zero floor applied to ' + ' and '.join(bits)
            if bits else '')
        refresh_from(0)

    r_tn = _radio(fig, gs[0, 0], 'normalize each TRIAL\n(before averaging trials)',
                  core.TRIAL_NORMS, 0, apply)
    r_nn = _radio(fig, gs[0, 1], 'normalize each NEURON\n(before averaging neurons)',
                  core.NEURON_NORMS, 0, apply)
    r_st = _radio(fig, gs[0, 2], 'average with', core.STATS, 0, apply)
    # RadioButtons are garbage-collected if nothing holds a reference to them,
    # and then they silently stop responding. Park them on the figure.
    fig.selectors = (r_tn, r_nn, r_st)

    footer(fig, 'click a trace to see the neurons behind it   ·   '
                'the selectors rebuild levels 0-2 in place')

    def redraw():
        core.draw_summary(ax, S)

    redraw()

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None or not idle(fig):
            return
        s = core.hit_summary(ax, S, ev.xdata, ev.ydata)
        if s is None:
            print('  (click closer to one of the traces)')
        else:
            show_neuron_matrix(S, s)

    fig.canvas.mpl_connect('button_press_event', on_click)
    register(0, fig, redraw)
    return fig


# ----------------------------------------------------------------------
# Level 1 -- neurons x time for one stimulus
# ----------------------------------------------------------------------
def show_neuron_matrix(S, s):
    close_from(1)
    fig = plt.figure(figsize=(6.8, 5.2))
    fig.canvas.manager.set_window_title(
        f'1: neurons x time, stim {S.stim_names[s]}')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.11, right=0.84, top=0.87, bottom=0.13)
    # The colourbar axes hangs off the figure, not off ax: ax.clear() on every
    # redraw would take an inset child with it.
    p = ax.get_position()
    cax = fig.add_axes([p.x1 + 0.02, p.y0, 0.025, p.height])

    def redraw():
        core.draw_neuron_matrix(ax, S, s, cax=cax)

    redraw()
    footer(fig, 'click a row to see that neuron')

    def on_click(ev):
        if ev.inaxes is not ax or ev.ydata is None or not idle(fig):
            return
        i = core.hit_neuron_matrix(S, ev.ydata)
        if i is not None:
            show_neuron(S, i)

    fig.canvas.mpl_connect('button_press_event', on_click)
    register(1, fig, redraw)
    return fig


# ----------------------------------------------------------------------
# Level 2 -- one neuron: all trials, all stimuli
# ----------------------------------------------------------------------
def show_neuron(S, neuron):
    close_from(2)
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(10.4, 5.4),
                                   gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.canvas.manager.set_window_title(f'2: neuron {neuron}')
    state = {}

    def redraw():
        state['row_trial'] = core.draw_neuron(axr, axp, S, neuron)
        fig.tight_layout(rect=FOOTER_RECT)

    redraw()
    footer(fig, 'click a trial row (left panel) to see the raw data '
                'for that trial')

    def on_click(ev):
        if ev.inaxes is not axr or ev.ydata is None or not idle(fig):
            return
        tr = core.hit_neuron(state['row_trial'], ev.ydata)
        if tr is not None:
            show_trial(S, tr, neuron)

    fig.canvas.mpl_connect('button_press_event', on_click)
    register(2, fig, redraw)
    return fig


# ----------------------------------------------------------------------
# Level 3 -- one trial: all neurons, and the raw voltage
# ----------------------------------------------------------------------
def show_trial(S, trial, highlight_neuron):
    close_from(3)
    fig, (axr, axv) = plt.subplots(1, 2, figsize=(12.4, 5.8), sharey=True,
                                   gridspec_kw=dict(width_ratios=[1, 1.35]))
    fig.canvas.manager.set_window_title(f'3: trial {trial}')
    band, dots = core.draw_trial(axr, axv, S, trial, highlight_neuron)
    axv.set_ylabel('')
    footer(fig, "click the left panel to re-centre the voltage window   ·   "
                "'d' toggles the spike dots   ·   pan and zoom with the toolbar")
    fig.tight_layout(rect=FOOTER_RECT)

    def on_click(ev):
        if ev.inaxes is not axr or ev.xdata is None or not idle(fig):
            return
        w = np.diff(axv.get_xlim())[0]
        axv.set_xlim(ev.xdata - w / 2, ev.xdata + w / 2)
        fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key == 'd':
            for d in dots:
                d.set_visible(not d.get_visible())
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    # Level 3 shows spikes and voltage, which no normalisation touches.
    register(3, fig, lambda: None)
    return fig


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--regen', action='store_true',
                   help='regenerate the dataset before starting')
    args = p.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'data', 'dataset.npz')
    if args.regen or not os.path.exists(path):
        gd.main()

    if matplotlib.get_backend().lower() in ('agg', 'pdf', 'svg', 'ps'):
        sys.exit(f"matplotlib is using the non-interactive "
                 f"'{matplotlib.get_backend()}' backend; this viewer needs an "
                 f"interactive one (e.g. TkAgg or QtAgg).")

    S = core.Session(path)
    print(__doc__)
    show_summary(S)
    plt.show()


if __name__ == '__main__':
    main()
