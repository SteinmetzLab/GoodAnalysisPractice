"""Build student and solution notebooks for Notebook 7 (correlated observations)."""
import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))

PREAMBLE = """\
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams['font.family']       = 'Arial'
plt.rcParams['font.sans-serif']   = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['pdf.fonttype']      = 42
plt.rcParams['ps.fonttype']       = 42
"""

LOAD = """\
d = np.load('data/data.npz')
activity = d['activity']        # (sessions, neurons, trials) firing rate
condition = d['condition']      # (sessions, trials)  +1 = A, -1 = B
arousal = d['arousal']          # (sessions, trials)  arousal per trial
n_sessions, n_neurons, n_trials = activity.shape
print(f'{n_sessions} sessions x {n_neurons} neurons = {n_sessions*n_neurons} neurons total')
"""

# ---------------- STUDENT ----------------
student = []
student.append(new_markdown_cell("""\
# Notebook 7 — Does the population distinguish condition A from B?

**Summer course: data-analysis pitfalls**

We recorded 600 neurons (across a handful of sessions) while the animal
experienced two conditions, A and B. For each neuron we compare its mean firing
on A trials vs. B trials. A colleague tested whether, across the population,
firing differs between conditions:

> ### The claim
> *"The population robustly distinguishes the two conditions: firing is higher on
> A trials than B trials (paired test across n = 600 neurons, p < 1e-12)."*

Run it and you'll get an astronomically small p-value.

**The claim is not supported by this analysis.** Your job is to work out *why*.
`hints.md` has guiding questions and `notebook7_solution.ipynb` has the answer.
"""))
student.append(new_code_cell(PREAMBLE))
student.append(new_code_cell(LOAD))
student.append(new_markdown_cell("""\
## Test A vs. B across all neurons

For each neuron, take its mean firing on A trials minus its mean on B trials,
then test whether that difference is non-zero across the population.
"""))
student.append(new_code_cell("""\
# per-neuron A-minus-B, pooling all neurons from all sessions
A = activity[:, :, condition[0] == 1].mean(2)     # (sessions, neurons) mean on A trials
B = activity[:, :, condition[0] == -1].mean(2)     # (sessions, neurons) mean on B trials
AmB = (A - B).ravel()                              # 600 neuron differences

t, p = stats.ttest_1samp(AmB, 0)
print(f'mean A-B = {AmB.mean():+.2f} spikes/s across {AmB.size} neurons')
print(f'one-sample t-test: t = {t:.1f}, p = {p:.1e}')

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.hist(AmB, bins=30, color='0.6')
ax.axvline(0, color='k', ls='--'); ax.axvline(AmB.mean(), color='tab:red', lw=2)
ax.set_xlabel('firing A - B (spikes/s)'); ax.set_ylabel('number of neurons')
ax.set_title(f'Almost every neuron fires more on A (p = {p:.1e})')
plt.show()
"""))
student.append(new_markdown_cell("""\
Nearly every one of 600 neurons fires more on A than B, with a vanishingly small
p-value. Strong population coding of condition.
"""))

# ---------------- SOLUTION ----------------
solution = []
solution.append(new_markdown_cell("""\
# Notebook 7 — SOLUTION

## Short version

**The neurons do not encode the condition.** Firing here is
`baseline + loading * arousal + noise`, with no condition term at all. What
happened is that, in this dataset, **arousal was a bit higher on A trials than B
trials** -- and since every neuron in a session shares that arousal, every neuron
shows A > B. The 600 neurons are **not 600 independent observations**: neurons
within a session are strongly correlated, so they mostly re-report the same
handful of session-level arousal fluctuations. The correct unit of replication is
the session (n = 6), not the neuron (n = 600), and at that level the effect is
not significant. Regressing out arousal removes it entirely.
"""))
solution.append(new_code_cell(PREAMBLE))
solution.append(new_code_cell(LOAD))
solution.append(new_code_cell("""\
A = activity[:, :, condition[0] == 1].mean(2)      # (sessions, neurons)
B = activity[:, :, condition[0] == -1].mean(2)
AmB = A - B                                         # (sessions, neurons)
t, p = stats.ttest_1samp(AmB.ravel(), 0)
print(f'naive test across {AmB.size} neurons: p = {p:.1e}')
"""))

solution.append(new_markdown_cell("""\
## Evidence 1 — The 600 "observations" are really 6 clusters

Neurons within a session share the same arousal signal, so their A-B differences
are nearly identical -- the population is not 600 independent points, it is 6
tight clusters (one per session). A few correlated sessions are masquerading as
hundreds of independent measurements.
"""))
solution.append(new_code_cell("""\
fig, ax = plt.subplots(figsize=(6.5, 4))
for s in range(n_sessions):
    jitter = np.random.default_rng(s).normal(s, 0.05, n_neurons)
    ax.scatter(jitter, AmB[s], s=10, alpha=0.5)
    ax.plot(s, AmB[s].mean(), 'k_', ms=25, mew=2)
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('session'); ax.set_ylabel('firing A - B (spikes/s)')
ax.set_title('Within each session all neurons agree (they share arousal)')
plt.show()

# mean pairwise correlation of neurons' trial-by-trial activity within a session
rs = [np.corrcoef(activity[s])[np.triu_indices(n_neurons, 1)].mean() for s in range(n_sessions)]
print(f'mean within-session pairwise neuron correlation: {np.mean(rs):.2f} '
      f'(independent neurons would be ~0)')
"""))

solution.append(new_markdown_cell("""\
## Evidence 2 — Use the right unit of replication: the session

Average A-B within each session, then test across the 6 sessions. Now the huge
session-to-session variability is visible, and the effect is not significant --
because there really are only ~6 independent measurements, not 600.
"""))
solution.append(new_code_cell("""\
session_means = AmB.mean(1)               # one number per session
t_s, p_s = stats.ttest_1samp(session_means, 0)
print('per-session mean A-B:', np.round(session_means, 2))
print(f'test across n = {n_sessions} sessions: t = {t_s:.2f}, p = {p_s:.2f}')
print(f'(naive across {AmB.size} neurons was p = {p:.1e})')
"""))

solution.append(new_markdown_cell("""\
## Evidence 3 — It's arousal: regress it out

Because firing tracks arousal and arousal differed between conditions, the "A-B"
effect is a confound. Regress each neuron's firing on arousal and test the
*residual* condition difference: it vanishes.
"""))
solution.append(new_code_cell("""\
resid_AmB = np.zeros((n_sessions, n_neurons))
for s in range(n_sessions):
    a = arousal[s]; cond = condition[s]
    Xd = np.vstack([np.ones_like(a), a]).T           # design: intercept + arousal
    coef, *_ = np.linalg.lstsq(Xd, activity[s].T, rcond=None)   # fit every neuron at once
    resid = activity[s].T - Xd @ coef                # remove the arousal-explained part
    resid_AmB[s] = resid[cond == 1].mean(0) - resid[cond == -1].mean(0)

t_r, p_r = stats.ttest_1samp(resid_AmB.ravel(), 0)
print(f'after regressing out arousal: mean residual A-B = {resid_AmB.mean():+.3f} spikes/s, '
      f'p = {p_r:.2f}')
"""))

solution.append(new_markdown_cell("""\
## Ground truth
"""))
solution.append(new_code_cell("""\
gt = np.load('data/ground_truth.npz', allow_pickle=True)
print('per-session A-vs-B arousal gap:', np.round(gt['arousal_gap'], 2))
print('note:', str(gt['note']))
"""))

solution.append(new_markdown_cell("""\
## The lesson

- **p-values assume independent observations.** Neurons recorded together share
  brain state (arousal, movement, drift), so they are correlated -- counting each
  as an independent sample shrinks the p-value toward zero and hugely overstates
  the evidence (pseudo-replication).
- **The unit of analysis is the independent unit.** Here that is the session (or
  animal), n = 6, not the neuron, n = 600. Summarize within unit and test across
  units, or use a mixed-effects model with a per-session random effect.
- **A shared confound multiplies into "many" fake observations.** One arousal
  difference, common to all neurons, reappears 600 times and looks overwhelming.
  Measure candidate confounds and regress them out; check that effects survive at
  the level of independent units.
"""))

# ------------------------------------------------------------------
def build(cells, path):
    nb = new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'display_name': 'Python 3 (swdb2026)',
                                 'language': 'python', 'name': 'python3'}
    NotebookClient(nb, timeout=180, kernel_name='python3',
                   resources={'metadata': {'path': HERE}}).execute()
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print('wrote', path)

build(student, os.path.join(HERE, 'notebook7_student.ipynb'))
build(solution, os.path.join(HERE, 'notebook7_solution.ipynb'))
