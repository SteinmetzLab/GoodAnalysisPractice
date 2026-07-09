"""Build student and solution notebooks for Notebook 5 (Simpson's paradox)."""
import os
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))

PREAMBLE = """\
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family']       = 'Arial'
plt.rcParams['font.sans-serif']   = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['pdf.fonttype']      = 42
plt.rcParams['ps.fonttype']       = 42
"""

LOAD = """\
d = np.load('data/data.npz')
pupil = d['pupil']                 # arousal proxy, per trial (0-1)
firing_rate = d['firing_rate']     # spikes/s, per trial
session_id = d['session_id']       # which session each trial came from
n_sessions = session_id.max() + 1
print(f'{len(pupil)} trials from {n_sessions} sessions')
"""

# ---------------- STUDENT ----------------
student = []
student.append(new_markdown_cell("""\
# Notebook 5 — Does arousal drive firing rate up or down?

**Summer course: data-analysis pitfalls**

Across many sessions we recorded a neuron's firing rate and, on each trial, the
animal's pupil size (a proxy for arousal). A colleague pooled all the trials and
fit a line:

> ### The claim
> *"Firing rate increases with arousal: across the dataset, pupil size and
> firing rate are positively correlated."*

Run it and you'll get a clear positive slope.

**The claim is backwards.** Your job is to work out *why*. `hints.md` has guiding
questions and `notebook5_solution.ipynb` has the answer.
"""))
student.append(new_code_cell(PREAMBLE))
student.append(new_code_cell(LOAD))
student.append(new_markdown_cell("""\
## Pool all trials and fit firing rate vs. pupil
"""))
student.append(new_code_cell("""\
slope, intercept = np.polyfit(pupil, firing_rate, 1)   # linear fit to all trials
r = np.corrcoef(pupil, firing_rate)[0, 1]

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(pupil, firing_rate, s=12, color='0.4', alpha=0.7)
xs = np.array([pupil.min(), pupil.max()])
ax.plot(xs, intercept + slope * xs, color='tab:red', lw=2)
ax.set_xlabel('pupil size (arousal)'); ax.set_ylabel('firing rate (spikes/s)')
ax.set_title(f'Pooled: slope = {slope:+.1f} spikes/s per unit pupil, r = {r:+.2f}')
plt.show()
"""))
student.append(new_markdown_cell("""\
A clear positive relationship: more arousal, more firing.
"""))

# ---------------- SOLUTION ----------------
solution = []
solution.append(new_markdown_cell("""\
# Notebook 5 — SOLUTION

## Short version

**Within every session, firing rate goes DOWN as pupil goes up.** The positive
pooled slope is **Simpson's paradox**: the sessions differ both in their average
pupil and in their baseline firing rate, and higher-arousal sessions happen to
have higher baseline rates. Pooling mixes this *between-session* trend with the
*within-session* relationship, and the between-session trend wins -- flipping the
sign. The pooled correlation answers a different question ("do high-arousal
*sessions* fire more?") than the one we care about ("does arousal drive firing
*within* a session?").
"""))
solution.append(new_code_cell(PREAMBLE))
solution.append(new_code_cell(LOAD))

solution.append(new_markdown_cell("""\
## Evidence 1 — Color the same scatter by session

The pooled cloud is made of several tilted sub-clouds, one per session. Each
slopes *down*; they are stacked up-and-to-the-right, which is what creates the
illusory positive pooled slope.
"""))
solution.append(new_code_cell("""\
slope, intercept = np.polyfit(pupil, firing_rate, 1)

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for s in range(n_sessions):
    m = session_id == s
    ax.scatter(pupil[m], firing_rate[m], s=14, alpha=0.8, label=f'session {s}')
    sl, ic = np.polyfit(pupil[m], firing_rate[m], 1)      # within-session fit
    xs = np.array([pupil[m].min(), pupil[m].max()])
    ax.plot(xs, ic + sl * xs, color='k', lw=1)
xs = np.array([pupil.min(), pupil.max()])
ax.plot(xs, intercept + slope * xs, color='tab:red', lw=3, ls='--', label='pooled fit')
ax.set_xlabel('pupil size (arousal)'); ax.set_ylabel('firing rate (spikes/s)')
ax.set_title('Each session slopes down; pooled fit (red) slopes up')
ax.legend(frameon=False, fontsize=8, ncol=2)
plt.show()
"""))

solution.append(new_markdown_cell("""\
## Evidence 2 — Within-session slopes vs. the pooled slope

Every session's slope is negative; only the pooled slope is positive.
"""))
solution.append(new_code_cell("""\
within = np.array([np.polyfit(pupil[session_id == s], firing_rate[session_id == s], 1)[0]
                   for s in range(n_sessions)])

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.bar(range(n_sessions), within, color='0.6', label='within-session slope')
ax.axhline(slope, color='tab:red', lw=2, label=f'pooled slope ({slope:+.0f})')
ax.axhline(0, color='k', lw=0.8)
ax.set_xlabel('session'); ax.set_ylabel('slope (spikes/s per unit pupil)')
ax.set_title('All within-session slopes are negative')
ax.legend(frameon=False, fontsize=8)
plt.show()
print(f'mean within-session slope: {within.mean():+.1f}   pooled slope: {slope:+.1f}')
"""))

solution.append(new_markdown_cell("""\
## Evidence 3 — The fix: analyze the within-session relationship

Remove each session's mean from both variables (so only within-session variation
remains), then pool. Equivalently, fit a model with a per-session intercept (a
mixed / within-subject model). The relationship is negative, as it truly is.
"""))
solution.append(new_code_cell("""\
pupil_c = pupil.copy(); firing_c = firing_rate.copy()
for s in range(n_sessions):                    # subtract each session's means
    m = session_id == s
    pupil_c[m] -= pupil[m].mean()
    firing_c[m] -= firing_rate[m].mean()
slope_within, ic_within = np.polyfit(pupil_c, firing_c, 1)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(pupil_c, firing_c, s=12, color='0.4', alpha=0.7)
xs = np.array([pupil_c.min(), pupil_c.max()])
ax.plot(xs, ic_within + slope_within * xs, color='tab:green', lw=2)
ax.set_xlabel('pupil (session-mean removed)'); ax.set_ylabel('firing rate (session-mean removed)')
ax.set_title(f'Within-session slope = {slope_within:+.1f} spikes/s per unit pupil')
plt.show()
"""))

solution.append(new_markdown_cell("""\
## Ground truth
"""))
solution.append(new_code_cell("""\
gt = np.load('data/ground_truth.npz', allow_pickle=True)
print('true within-session slope:', float(gt['within_slope']), 'spikes/s per unit pupil')
print('note:', str(gt['note']))
"""))

solution.append(new_markdown_cell("""\
## The lesson

- **A relationship pooled across groups can reverse the relationship within each
  group (Simpson's paradox).** It happens whenever the grouping variable
  (session, animal, subject) is related to *both* variables.
- **Decide which question you are asking.** "Do high-arousal sessions fire more?"
  (between-session) and "does arousal drive firing within a session?"
  (within-session) are different questions with, here, opposite answers. The
  pooled fit silently answers the first.
- **Respect the grouping structure.** Analyze within group (center by group, or
  fit a mixed-effects model with per-group intercepts) rather than pooling
  trials as if they were independent and exchangeable.
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

build(student, os.path.join(HERE, 'notebook5_student.ipynb'))
build(solution, os.path.join(HERE, 'notebook5_solution.ipynb'))
