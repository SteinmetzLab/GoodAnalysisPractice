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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_data as gd
import drilldown_core as core
from drilldown_core import STIM_COLORS

core.apply_style()


# ----------------------------------------------------------------------
# Window stack: one figure per level, deeper ones close with their parent
# ----------------------------------------------------------------------
FIGS = {}


def close_from(level):
    for lv in sorted([lv for lv in FIGS if lv >= level], reverse=True):
        fig = FIGS.pop(lv, None)
        if fig is not None:
            plt.close(fig)


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


def register(level, fig):
    FIGS[level] = fig

    def on_close(ev, lv=level, f=fig):
        close_from(lv + 1)
        if FIGS.get(lv) is f:
            FIGS.pop(lv)

    fig.canvas.mpl_connect('close_event', on_close)
    fig.canvas.mpl_connect(
        'key_press_event',
        lambda ev, f=fig: plt.close(f) if ev.key == 'escape' else None)
    place_window(fig, 60 + 90 * level, 60 + 60 * level)
    fig.show()


def footer(fig, text):
    """One line of usage hints along the bottom of a figure."""
    fig.text(0.012, 0.012, text, fontsize=9, style='italic', color='0.35',
             ha='left', va='bottom')


FOOTER_RECT = (0, 0.055, 1, 1)   # tight_layout rect leaving room for footer


def idle(fig):
    """True unless the toolbar is in pan/zoom mode (then clicks aren't ours)."""
    return not (fig.canvas.toolbar is not None and fig.canvas.toolbar.mode)


# ----------------------------------------------------------------------
def show_summary(S):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    fig.canvas.manager.set_window_title('0: population average')
    core.draw_summary(ax, S)
    footer(fig, 'click a trace to see the neurons behind it')
    fig.tight_layout(rect=FOOTER_RECT)

    def on_click(ev):
        if ev.inaxes is not ax or ev.xdata is None or not idle(fig):
            return
        s = core.hit_summary(ax, S, ev.xdata, ev.ydata)
        if s is None:
            print('  (click closer to one of the traces)')
        else:
            show_neuron_matrix(S, s)

    fig.canvas.mpl_connect('button_press_event', on_click)
    register(0, fig)
    return fig


def show_neuron_matrix(S, s):
    close_from(1)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    fig.canvas.manager.set_window_title(
        f'1: neurons x time, stim {S.stim_names[s]}')
    im = core.draw_neuron_matrix(ax, S, s)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label('firing rate (spikes/s)')
    footer(fig, 'click a row to see that neuron')
    fig.tight_layout(rect=FOOTER_RECT)

    def on_click(ev):
        if ev.inaxes is not ax or ev.ydata is None or not idle(fig):
            return
        i = core.hit_neuron_matrix(S, ev.ydata)
        if i is not None:
            show_neuron(S, i)

    fig.canvas.mpl_connect('button_press_event', on_click)
    register(1, fig)
    return fig


def show_neuron(S, neuron):
    close_from(2)
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(10.4, 5.4),
                                   gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.canvas.manager.set_window_title(f'2: neuron {neuron}')
    row_trial = core.draw_neuron(axr, axp, S, neuron)
    footer(fig, 'click a trial row (left panel) to see the raw data '
                'for that trial')
    fig.tight_layout(rect=FOOTER_RECT)

    def on_click(ev):
        if ev.inaxes is not axr or ev.ydata is None or not idle(fig):
            return
        tr = core.hit_neuron(row_trial, ev.ydata)
        if tr is not None:
            show_trial(S, tr, neuron)

    fig.canvas.mpl_connect('button_press_event', on_click)
    register(2, fig)
    return fig


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
    register(3, fig)
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
