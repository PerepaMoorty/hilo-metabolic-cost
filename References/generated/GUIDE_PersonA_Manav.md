# Team 13 ML Mini-Project - Build guide for Manav Dewangan (Person A)

> **Manav: how to use this file.** Open **Claude Code** in an empty folder where you keep projects, give it this file and say: *"Read this guide and follow it step by step for me."* Claude writes the code, runs the tests and explains everything. **You type every git / gh command yourself** (Claude shows each one with a comment explaining it), so every commit and PR is yours. It also tells you when to send a link to Moorty or wait for a merge. Keep Moorty updated at every CHECKPOINT.

## 0. Instructions for Claude (read this whole section first)

You are Claude Code helping **Manav Dewangan** (PES2UG24CS262) complete their half of a two-person university
machine-learning project. Manav has uploaded this guide; follow it **step by step, in order**.

Rules:
1. **Exact files.** Create every file with *exactly* the content given in this guide. Do not
   refactor, reformat, rename or "improve" code - the two halves of the project were written and
   tested together and depend on these exact names.
2. **Manav runs every git and gh command - you never do.** Do not run `git` or `gh` at all
   (not even `git status`, `git init`, `git add` or `git config`). Wherever this guide shows
   "**Manav types these commands**", show the block, explain each line in one sentence, and wait
   until Manav says it worked (ask them to paste the output if anything looks wrong). There is one
   commit per `commit:` heading, with that exact message and only the files listed. Small, separate
   commits typed by Manav are what makes their contribution visible to the evaluators.
3. **Stay in your lane.** Only touch the files assigned to Manav. Never edit or
   "fix" these files owned by Moorty: `.gitignore`, `.gitattributes`, `requirements.txt`, `pytest.ini`, `src/__init__.py`, `src/config.py`, `src/models.py`, `tests/test_models.py`, `src/feature_selection.py`, `tests/test_feature_selection.py`, `src/viz.py`, `scripts/__init__.py`, `scripts/run_experiments.py`, `notebooks/03_models_and_results.ipynb`, `notebooks/04_live_demo.ipynb`. If one of them seems broken, stop and tell
   Manav to contact Moorty.
4. **Tests gate everything.** You run `python -m pytest -q` before every commit and push. If it
   fails, fix the cause before showing Manav any git command.
   (Before any test file exists, pytest reports "no tests ran" - that is fine.)
5. **Checkpoints.** At every **CHECKPOINT**, stop, tell Manav exactly what to do (e.g. send a link
   to Moorty, wait for a merge) and wait for them to confirm before continuing.
6. **Shell.** Commands are written for bash (Git Bash / macOS / Linux). On Windows PowerShell,
   translate them (e.g. `source .venv/bin/activate` -> `.venv\Scripts\Activate.ps1`, `cp` ->
   `Copy-Item`, `mkdir -p` -> `New-Item -ItemType Directory -Force`). Always use the project's
   virtual environment's `python`.
7. **Keep out of the repo:** `.venv/`, notebook builder scripts, `__pycache__/` and any secrets
   must never be staged (the `git add` lines in this guide already list exact files).
8. **Teach as you go.** After each PR, explain to Manav in plain English what was built and why;
   they must answer questions about it in a live viva. Section "Q&A preparation" lists what they
   need to know - quiz them on it at the end.
9. If something in this guide does not work as written, stop and explain the problem to Manav
   rather than improvising a different design.

## 1. The project in one page

**Course:** UE24CS352A Machine Learning mini-project. **Team 13, Problem 18:** *Predicting Metabolic
Cost During Human-in-the-Loop Optimization*, based on the Stanford CS229 (2018) project by Krimsky &
Ng (paper, poster and MATLAB code were provided as references).

**Deadlines:** review + live demo on **Thu 8 Oct 2026**; final deliverables (GitHub repo, 2-page PDF
write-up) by **Sat 10 Oct 2026, 11:59 PM**. Marks: deliverable quality, code functionality, **repository
maintenance**, write-up clarity, presentation, live demo, Q&A, and **individual contribution**.

**Problem.** An exoskeleton's controller is tuned while a person walks, using metabolic cost (energy
use) as the objective. Metabolic cost is slow, noisy and needs a lab mask, so we predict it from
signals measurable outside the lab: gait/step features and EMG (muscle activity), plus the
exoskeleton's 4 control parameters.

**Data.** `processed_data.mat` from the reference repo: 2 subjects (S1, S2) x 5 walking days x 36
controller settings = 180 samples each (179 for S2 after dropping one impossible negative value).
29 features = 9 step + 16 EMG + 4 control. Target = metabolic cost normalised to normal walking.
Rows are stored day by day, so we know which day each sample came from.

**What we do (Python / scikit-learn) - and how it differs from the reference:**

| Reference (MATLAB, 2018) | Our project |
|---|---|
| LASSO + 1-hidden-layer tanh NN (Bayesian regularisation) | Same model families, re-implemented in scikit-learn (L2-penalised MLP, L-BFGS) + Ridge baseline |
| Random k-fold CV only | Random k-fold **and leave-one-day-out CV** (test on an unseen walking day - the paper's own suggested future work) |
| Data scaled on the full dataset before splitting (leakage) | Every scaler / tuner / selector / PCA fitted inside the training folds only |
| NN size picked by looking at test-fold error | **Nested CV**: size and penalty chosen on inner folds |
| Forward selection / PCA run on all data | Nested forward selection and PCA, plus selection *frequency* across folds |
| MATLAB scripts | Tested Python package, experiment runner, notebooks, live-demo notebook |

**Headline results** (MSE, all features; lower is better; noise floor ~0.0025):

| | S1 k-fold | S1 leave-one-day-out | S2 k-fold | S2 leave-one-day-out |
|---|---|---|---|---|
| Constant (predict mean) | 0.0666 | 0.0917 | 0.0362 | 0.0478 |
| LASSO | 0.0091 (paper 0.0089) | 0.0205 | 0.0135 (paper 0.0176) | 0.0309 |
| Ridge | 0.0086 | 0.0143 | 0.0146 | 0.0358 |
| Neural net (nested-tuned) | 0.0101 (paper 0.0089) | 0.0223 | 0.0150 (paper 0.0301) | 0.0315 |

Story: we reproduce the paper under its protocol, but predicting a **new day** roughly doubles the
error, because the subject keeps adapting (mean cost falls from ~1.1 on day 1 to ~0.7 on day 5).
On 180 samples, regularised linear models are as good as the shallow neural network.

## 2. Who does what

| | Person A - **Manav Dewangan** | Person B - **Moorty Perepa** |
|---|---|---|
| Code | `src/data.py`, `src/evaluation.py`, `src/baselines.py`, `src/eda.py` + their tests | `src/config.py` (scaffold), `src/models.py`, `src/feature_selection.py`, `src/viz.py`, `scripts/run_experiments.py` + their tests |
| Notebooks | `01_eda`, `02_baselines` | `03_models_and_results`, `04_live_demo` |
| Docs | README (setup/run), write-up sections 1-3 | README results, write-up sections 4-6, compiled PDF |
| Repo | - | owns the GitHub repo, adds collaborators |
| Slides | slides 1-6 | slides 7-12 |

**You are Person A - data, evaluation protocols, baselines, EDA, README.**

### Order of pull requests

Moorty owns the repository. Each PR is reviewed and merged by the *other* person.

| Order | PR | Branch | Owner | Depends on | Reviewed & merged by |
|---|---|---|---|---|---|
| 1 | (direct push) Repository scaffold | `main` | Moorty | - | - |
| 2 | #1 Data loading and cross-validation protocols | `feature/data-and-cv` | Manav | scaffold | Moorty |
| 3 | #2 Baseline models and exploratory data analysis | `feature/baselines-and-eda` | Manav | PR #1 | Moorty |
| 4 | #3 Neural network with nested hyper-parameter tuning | `feature/neural-network` | Moorty | PR #1 | Manav |
| 5 | #4 Nested forward selection and PCA | `feature/feature-selection` | Moorty | PR #3 | Manav |
| 6 | #5 Experiment runner, results and live demo | `feature/experiments-and-demo` | Moorty | PR #2, #4 | Manav |
| 7 | #6 README and write-up skeleton | `docs/readme-and-report` | Manav | PR #2 | Moorty |
| 8 | #7 Results in README and write-up | `docs/results-and-report` | Moorty | PR #6 | Manav |

Rough timeline:

- **Wed 7 Oct (today):** scaffold -> PR #1 -> (#2 in parallel with #3, #4) -> #5 -> #6. Then build slides together and rehearse the demo.
- **Thu 8 Oct:** before the review, Manav does a *fresh clone* check (Section "Final checks"). Review + live demo.
- **Fri 9 - Sat 10 Oct:** PR #7, compile the write-up PDF on Overleaf, commit PDFs, add faculty to the repo, submit.

### Branching rules for both of you
- Never commit directly to `main` after the scaffold. One branch per PR, named as in the table.
- Pull `main` before starting each new branch.
- Merge with **merge commits** (`gh pr merge N --merge`), so every individual commit and its author stay in history.

## 3. One-time setup (Manav)

**Install once** (skip what is already installed):
- Git - https://git-scm.com/downloads
- Python **3.11** (3.10-3.13 work) - https://www.python.org/downloads/ (tick "Add to PATH" on Windows)
- GitHub CLI `gh` - https://cli.github.com/
- VS Code with the Python + Jupyter extensions (optional, for viewing notebooks)

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# log in to GitHub from the terminal (choose GitHub.com -> HTTPS -> login with browser)
gh auth login
# check it worked: should say 'Logged in to github.com account <you>'
gh auth status
```

Claude: ask Manav for the **repo URL** Moorty sent, and check he has **accepted the collaborator
invite** (`https://github.com/<moorty-username>/hilo-metabolic-cost/invitations`). Then:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# download the shared repository (creates a folder called hilo-metabolic-cost)
git clone <repo-url>
cd hilo-metabolic-cost
# stamp your commits with your name and your GitHub email (only for this repo)
git config user.name "Manav Dewangan"
git config user.email "<your-github-email>"
```

Claude then sets up the Python environment inside the repo folder (this is not git, so Claude runs it):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q                 # 'no tests ran' is fine at this stage
```

You should see Moorty's scaffold: `README.md`, `requirements.txt`, `pytest.ini`, `.gitignore`,
`.gitattributes`, `src/__init__.py`, `src/config.py`. If the repo is empty, Moorty has not pushed
yet - wait for him.

## Step 1 - PR #1: Data loading and cross-validation protocols

_Everything else imports load_subject() and cross_validate_model(), so this goes first._

Start a fresh branch from the latest `main`.

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
git checkout main                 # go back to the shared main branch
git pull                          # download everything merged so far
git checkout -b feature/data-and-cv   # create your own working branch for this PR
```

### A1.1 - commit: `data: add processed HILO dataset and description`

Download **`data/processed_data.mat`** (the processed dataset published with the reference project) and verify its checksum. Run from the repo root:

```bash
python -c "import os, urllib.request, hashlib; os.makedirs('data', exist_ok=True); urllib.request.urlretrieve('https://github.com/ngeley/cs229project/raw/master/processed_data.mat', 'data/processed_data.mat'); print(hashlib.md5(open('data/processed_data.mat','rb').read()).hexdigest())"
```

The printed hash **must** be `f386194db18a12532497cead93f791a8` (file size 78,669 bytes). If the download fails, ask Moorty to send the file (it is in his `References/cs229project-master/` folder) and copy it to `data/processed_data.mat`.

Create **`data/README.md`** with exactly this content:

```markdown
# Data

`processed_data.mat` is the processed dataset released with the CS229 (Autumn 2018) project
"Predicting Metabolic Cost During Human-in-the-Loop Optimization" by Erez Krimsky and Eley Ng
(https://github.com/ngeley/cs229project). The raw recordings come from an ongoing study at the
Stanford Biomechatronics Laboratory and are not public; this file is the only data used here.

## Contents

| Variable | Shape | Meaning |
|---|---|---|
| `michael_data` | 180 x 29 | Features for subject **S1** |
| `michael_metabolics` | 180 x 1 | Target for S1 |
| `eley_data` | 180 x 29 | Features for subject **S2** |
| `eley_metabolics` | 180 x 1 | Target for S2 |
| `data_labels` | 1 x 29 | Feature names |

Each subject walked on **5 days**; each day is one 72-minute CMA-ES optimisation trial with
**36 controller settings** (2 minutes each). Rows are stored in day order, 36 per day, so the
day of each row is `row_index // 36`. Each row summarises the last 30 s of one 2-minute condition.

## Features (columns)

| Columns | Group | Description |
|---|---|---|
| 0-3 | step (left) | peak vertical ground force (body-weight normalised), peak dorsiflexion and plantarflexion ankle angle, step time - medians over the 30 s window |
| 4-7 | step (right) | same for the right leg |
| 8 | step | step width (from centre of pressure) |
| 9-24 | EMG | RMS of 16 filtered, rectified EMG channels (8 per leg), normalised by the peak during normal walking that day |
| 25-28 | control | the 4 exoskeleton torque-profile parameters set by the optimiser (the paper lists peak time, rise time, peak torque and settling time; the column order is not documented, so they are named ctrl1-ctrl4) |

## Target

Net metabolic cost (minus quiet-standing baseline) divided by the net cost of normal walking on the
same day. 1.0 = same effort as normal walking without the exoskeleton.

One S2 sample has a negative value (below standing) and is removed as a sensor error by
`src.data.load_subject`, leaving 179 S2 samples.
```

Then the commit:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add data/processed_data.mat data/README.md
# save this step under your name
git commit -m "data: add processed HILO dataset and description"
```

### A1.2 - commit: `feat(data): load subjects with day labels and feature groups`

Create **`src/data.py`** with exactly this content:

```python
"""Loading the processed HILO dataset into a convenient per-subject structure."""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.config import CONDITIONS_PER_DAY, DATA_PATH, FEATURE_GROUPS, N_DAYS, SUBJECTS


@dataclass(frozen=True)
class SubjectData:
    """Features, target and day labels for one subject.

    X     : (n_samples, n_features) feature matrix
    y     : (n_samples,) normalised metabolic cost
    day   : (n_samples,) walking day index 0..4, used for leave-one-day-out CV
    feature_names : column names of X
    """

    subject: str
    X: np.ndarray
    y: np.ndarray
    day: np.ndarray
    feature_names: list

    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    def subset(self, group: str) -> "SubjectData":
        """Return a copy that keeps only one feature group ('all', 'step_emg', 'step', 'emg')."""
        cols = FEATURE_GROUPS[group]
        return replace(self, X=self.X[:, cols], feature_names=[self.feature_names[c] for c in cols])

    def to_frame(self) -> pd.DataFrame:
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df["day"] = self.day
        df["metabolic_cost"] = self.y
        return df


def load_subject(subject: str, path=DATA_PATH, drop_invalid: bool = True) -> SubjectData:
    """Load one subject ('S1' or 'S2') from processed_data.mat.

    Rows are stored day by day (36 conditions per day), so the day label is
    recovered from the row position. A normalised metabolic cost below zero
    means less energy than quiet standing, which is physically impossible
    while walking, so those rows are treated as sensor errors and dropped.
    """
    if subject not in SUBJECTS:
        raise ValueError(f"Unknown subject {subject!r}; expected one of {list(SUBJECTS)}")
    mat = loadmat(path)
    prefix = SUBJECTS[subject]
    X = np.asarray(mat[f"{prefix}_data"], dtype=float)
    y = np.asarray(mat[f"{prefix}_metabolics"], dtype=float).ravel()
    names = [str(n[0]) for n in mat["data_labels"].ravel()]

    expected = N_DAYS * CONDITIONS_PER_DAY
    if X.shape[0] != expected:
        raise ValueError(f"Expected {expected} rows for {subject}, found {X.shape[0]}")
    day = np.repeat(np.arange(N_DAYS), CONDITIONS_PER_DAY)

    if drop_invalid:
        keep = y >= 0
        X, y, day = X[keep], y[keep], day[keep]

    return SubjectData(subject=subject, X=X, y=y, day=day, feature_names=names)


def load_all(**kwargs) -> dict:
    """Load every subject: {'S1': SubjectData, 'S2': SubjectData}."""
    return {s: load_subject(s, **kwargs) for s in SUBJECTS}
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add src/data.py
# save this step under your name
git commit -m "feat(data): load subjects with day labels and feature groups"
```

### A1.3 - commit: `feat(eval): k-fold and leave-one-day-out cross-validation`

Create **`src/evaluation.py`** with exactly this content:

```python
"""Cross-validation protocols and metrics shared by every model.

Two protocols are compared throughout the project:

* ``kfold`` - shuffled 5-fold CV over all rows, repeated with different seeds.
  This mirrors the protocol of Krimsky & Ng (2018) and lets us compare
  numbers with their paper.
* ``lodo``  - leave-one-day-out CV: train on 4 walking days, test on the
  unseen 5th day. This is the stricter test the paper proposes as future
  work, because conditions from the same day are strongly correlated.

Every model passed in here must be a full sklearn estimator/pipeline that
contains its own scaler, so scaling is learned on the training folds only
(the reference implementation scaled the whole dataset before splitting).
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict

from src.config import RANDOM_STATE

PROTOCOLS = ("kfold", "lodo")


@dataclass
class EvalResult:
    protocol: str
    mse: float
    mse_std: float
    r2: float
    y_pred: np.ndarray
    fold_mse: list = field(default_factory=list)

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.mse))


def make_splitter(protocol: str, seed: int = RANDOM_STATE, n_splits: int = 5):
    if protocol == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if protocol == "lodo":
        return LeaveOneGroupOut()
    raise ValueError(f"Unknown protocol {protocol!r}; expected one of {PROTOCOLS}")


def _fold_mse(y, y_pred, splitter, X, groups):
    return [float(mean_squared_error(y[test], y_pred[test])) for _, test in splitter.split(X, y, groups)]


def cross_validate_model(model, data, protocol: str, n_repeats: int = 3, n_jobs: int = -1) -> EvalResult:
    """Out-of-fold MSE / R^2 of ``model`` on ``data`` (a SubjectData).

    For ``kfold`` the split is repeated ``n_repeats`` times with different
    seeds and the MSE is averaged; ``y_pred`` comes from the first repeat.
    For ``lodo`` the split is deterministic, so it is run once and
    ``fold_mse`` holds the error on each held-out day.
    """
    X, y = data.X, data.y
    groups = data.day if protocol == "lodo" else None  # KFold ignores (and warns about) groups
    repeats = n_repeats if protocol == "kfold" else 1

    mses, first_pred, first_folds = [], None, None
    for r in range(repeats):
        splitter = make_splitter(protocol, seed=RANDOM_STATE + r)
        y_pred = cross_val_predict(clone(model), X, y, groups=groups, cv=splitter, n_jobs=n_jobs)
        mses.append(mean_squared_error(y, y_pred))
        if first_pred is None:
            first_pred = y_pred
            first_folds = _fold_mse(y, y_pred, splitter, X, groups)

    spread = np.std(mses) if repeats > 1 else np.std(first_folds)
    return EvalResult(
        protocol=protocol,
        mse=float(np.mean(mses)),
        mse_std=float(spread),
        r2=float(r2_score(y, first_pred)),
        y_pred=first_pred,
        fold_mse=first_folds,
    )
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add src/evaluation.py
# save this step under your name
git commit -m "feat(eval): k-fold and leave-one-day-out cross-validation"
```

### A1.4 - commit: `test: data loading and CV protocol tests`

Create **`tests/test_data.py`** with exactly this content:

```python
import numpy as np
import pytest

from src.config import FEATURE_GROUPS
from src.data import load_all, load_subject


def test_shapes_and_days():
    s1 = load_subject("S1")
    assert s1.X.shape == (180, 29)
    assert len(s1.feature_names) == 29
    assert np.bincount(s1.day).tolist() == [36] * 5


def test_invalid_target_dropped_for_s2():
    s2 = load_subject("S2")
    raw = load_subject("S2", drop_invalid=False)
    assert s2.n_samples == 179 and raw.n_samples == 180
    assert (s2.y >= 0).all()


def test_feature_group_subset():
    s1 = load_subject("S1")
    for name, cols in FEATURE_GROUPS.items():
        sub = s1.subset(name)
        assert sub.X.shape == (180, len(cols))
        assert sub.feature_names[0] == s1.feature_names[cols[0]]
    assert s1.subset("emg").feature_names[0] == "emg1"


def test_load_all_and_unknown_subject():
    assert set(load_all()) == {"S1", "S2"}
    with pytest.raises(ValueError):
        load_subject("S3")
```

Create **`tests/test_evaluation.py`** with exactly this content:

```python
import numpy as np
from sklearn.dummy import DummyRegressor

from src.data import load_subject
from src.evaluation import cross_validate_model, make_splitter


def test_lodo_holds_out_one_whole_day():
    d = load_subject("S1")
    folds = list(make_splitter("lodo").split(d.X, d.y, d.day))
    assert len(folds) == 5
    for train, test in folds:
        assert len(np.unique(d.day[test])) == 1
        assert not set(d.day[train]) & set(d.day[test])


def test_constant_model_matches_target_variance():
    d = load_subject("S1")
    res = cross_validate_model(DummyRegressor(), d, "kfold", n_repeats=2)
    assert abs(res.mse - d.y.var()) < 0.01
    assert len(res.y_pred) == d.n_samples


def test_lodo_reports_one_error_per_day():
    d = load_subject("S2")
    res = cross_validate_model(DummyRegressor(), d, "lodo")
    assert len(res.fold_mse) == 5
    assert res.rmse > 0
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add tests/test_data.py tests/test_evaluation.py
# save this step under your name
git commit -m "test: data loading and CV protocol tests"
```

### A1 - open the pull request

Claude runs the full test suite one last time: `python -m pytest -q` (it must say 'passed' with no failures).

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# upload your branch to GitHub
git push -u origin feature/data-and-cv
# open the pull request (prints the PR link)
gh pr create --base main --head feature/data-and-cv --title "Data loading and cross-validation protocols" --body "Everything else imports load_subject() and cross_validate_model(), so this goes first. Part of Manav's work (data, evaluation protocols, baselines, EDA, README)."
```

> **CHECKPOINT** - Claude: ask Manav for the PR link printed above and tell them: "Send this link to Moorty to review and merge." Then give Manav a 4-6 bullet plain-English summary of what this PR does and why, so they can explain it in the Q&A.

> **Wait** until Moorty tells Manav that PR #1 is merged (the PR page on GitHub also shows *Merged*), then continue.

## Step 2 - PR #2: Baseline models and exploratory data analysis

_Baselines are needed by the experiment runner and the demo notebook._

Start a fresh branch from the latest `main`.

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
git checkout main                 # go back to the shared main branch
git pull                          # download everything merged so far
git checkout -b feature/baselines-and-eda   # create your own working branch for this PR
```

### A2.1 - commit: `feat(baselines): constant, LASSO and ridge models`

Create **`src/baselines.py`** with exactly this content:

```python
"""Baseline regressors: constant prediction and regularised linear models."""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Same regularisation range the reference study searched over
LASSO_ALPHAS = np.logspace(-5, -1, 25)
RIDGE_ALPHAS = np.logspace(-3, 3, 25)


def make_constant():
    """Always predicts the training-set mean. Its MSE equals the target variance."""
    return DummyRegressor(strategy="mean")


def make_lasso():
    """Standardise -> LASSO, with the L1 strength chosen by an inner 5-fold CV."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("lasso", LassoCV(alphas=LASSO_ALPHAS, cv=5, max_iter=50_000)),
    ])


def make_ridge():
    """Standardise -> Ridge, with the L2 strength chosen by efficient leave-one-out CV."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
    ])


BASELINES = {
    "constant": make_constant,
    "lasso": make_lasso,
    "ridge": make_ridge,
}
```

Create **`tests/test_baselines.py`** with exactly this content:

```python
from src.baselines import BASELINES, make_constant, make_lasso
from src.data import load_subject
from src.evaluation import cross_validate_model


def test_all_baselines_fit_and_predict():
    d = load_subject("S1")
    for name, factory in BASELINES.items():
        pred = factory().fit(d.X, d.y).predict(d.X)
        assert pred.shape == d.y.shape, name


def test_lasso_beats_constant_on_kfold():
    d = load_subject("S1")
    lasso = cross_validate_model(make_lasso(), d, "kfold", n_repeats=1)
    const = cross_validate_model(make_constant(), d, "kfold", n_repeats=1)
    assert lasso.mse < const.mse / 2
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add src/baselines.py tests/test_baselines.py
# save this step under your name
git commit -m "feat(baselines): constant, LASSO and ridge models"
```

### A2.2 - commit: `feat(eda): exploratory plotting helpers`

Create **`src/eda.py`** with exactly this content:

```python
"""Exploratory plots of the raw (unscaled) dataset."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CONDITIONS_PER_DAY, N_DAYS

GROUP_COLORS = {"step": "#2a78b5", "emg": "#d4622a", "control": "#5a9e45"}


def _group_of(name: str) -> str:
    if name.startswith("emg"):
        return "emg"
    if name.startswith("ctrl"):
        return "control"
    return "step"


def plot_metabolic_by_day(datasets: dict):
    """Metabolic cost of every condition in walking order, one panel per subject.

    Shows the day-to-day drift (the subject is still learning to walk with the
    exoskeleton), which is why leave-one-day-out CV is the harder test.
    """
    fig, axes = plt.subplots(len(datasets), 1, figsize=(10, 3.2 * len(datasets)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, d) in zip(axes, datasets.items()):
        position = np.arange(len(d.y))
        for k in range(N_DAYS):
            m = d.day == k
            ax.plot(position[m], d.y[m], "o-", ms=3, lw=1, label=f"Day {k + 1}")
            ax.hlines(d.y[m].mean(), position[m].min(), position[m].max(), colors="k", linestyles="--", lw=1)
        ax.set_title(f"{name}: normalised metabolic cost per condition (dashed = day mean)")
        ax.set_ylabel("Metabolic cost (norm.)")
    axes[-1].set_xlabel(f"Condition index (each day = {CONDITIONS_PER_DAY} conditions)")
    axes[0].legend(ncol=N_DAYS, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def day_summary(datasets: dict) -> pd.DataFrame:
    """Mean and standard deviation of metabolic cost per subject and day."""
    rows = []
    for name, d in datasets.items():
        for k in range(N_DAYS):
            yk = d.y[d.day == k]
            rows.append({"subject": name, "day": k + 1, "n": len(yk), "mean": yk.mean(), "std": yk.std()})
    return pd.DataFrame(rows).round(3)


def feature_target_correlation(data) -> pd.Series:
    """Pearson correlation of each feature with metabolic cost."""
    df = pd.DataFrame(data.X, columns=data.feature_names)
    return df.corrwith(pd.Series(data.y)).rename("corr")


def plot_feature_correlations(datasets: dict):
    """Bar chart of feature-vs-target correlation, coloured by feature group."""
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 7), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, d) in zip(axes, datasets.items()):
        corr = feature_target_correlation(d)
        colors = [GROUP_COLORS[_group_of(n)] for n in corr.index]
        ax.barh(corr.index, corr.values, color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{name}: correlation with metabolic cost")
        ax.set_xlabel("Pearson r")
        ax.invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    axes[-1].legend(handles, GROUP_COLORS.keys(), loc="lower right")
    fig.tight_layout()
    return fig


def plot_feature_heatmap(data):
    """Feature-feature correlation matrix, to show redundancy (e.g. left vs right leg)."""
    corr = np.corrcoef(data.X, rowvar=False)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(data.feature_names)), data.feature_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(data.feature_names)), data.feature_names, fontsize=7)
    ax.set_title(f"{data.subject}: feature correlation matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add src/eda.py
# save this step under your name
git commit -m "feat(eda): exploratory plotting helpers"
```

### A2.3 - commit: `notebook: exploratory data analysis`

Generate **`notebooks/01_eda.ipynb`** with the builder script below. Save the script **outside the repository** (for example in your system temp folder) as `build_01_eda.py` - it must not be committed. Then, from the repo root with the virtual environment active:

```bash
python <path-to>/build_01_eda.py
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

The executed notebook (with outputs and plots) is what gets committed. Check it contains no error outputs, then delete the builder script.

```python
import os

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 01 - Exploratory Data Analysis\n"
       "Dataset: processed HILO walking data for two subjects (5 days x 36 control conditions each).\n\n"
       "Target: metabolic cost normalised by the subject's normal-walking cost on the same day.\n\n"
       "Features: 9 step features, 16 EMG channels, 4 exoskeleton control parameters."),
    code("import sys, pathlib, warnings\n"
         "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
         "warnings.filterwarnings('ignore')\n"
         "import matplotlib.pyplot as plt\n"
         "from src.config import FIGURES_DIR, FEATURE_GROUPS\n"
         "from src.data import load_all\n"
         "from src import eda\n"
         "FIGURES_DIR.mkdir(parents=True, exist_ok=True)\n"
         "datasets = load_all()\n"
         "for name, d in datasets.items():\n"
         "    print(name, 'X', d.X.shape, '| target mean %.3f, variance %.4f' % (d.y.mean(), d.y.var()))"),
    md("## Feature groups"),
    code("{k: len(v) for k, v in FEATURE_GROUPS.items()}"),
    code("datasets['S1'].to_frame().describe().T.round(3)"),
    md("## Metabolic cost drifts from day to day\n"
       "The subject is still learning to walk with the exoskeleton, so the mean cost drops over the five days. "
       "Conditions from the same day are therefore correlated, which is why random k-fold CV is optimistic "
       "and leave-one-day-out CV is the more honest test."),
    code("eda.day_summary(datasets)"),
    code("fig = eda.plot_metabolic_by_day(datasets)\n"
         "fig.savefig(FIGURES_DIR / 'eda_metabolic_by_day.png', dpi=150)\n"
         "plt.show()"),
    md("## Which features track metabolic cost?"),
    code("fig = eda.plot_feature_correlations(datasets)\n"
         "fig.savefig(FIGURES_DIR / 'eda_feature_correlations.png', dpi=150)\n"
         "plt.show()"),
    code("for name, d in datasets.items():\n"
         "    top = eda.feature_target_correlation(d).sort_values(key=abs, ascending=False).head(6)\n"
         "    print(name, top.round(2).to_dict())"),
    md("## Redundancy between features\n"
       "Left and right step times are almost perfectly correlated (r close to 1), so the two legs carry largely "
       "the same timing information - consistent with the reference paper's observation."),
    code("fig = eda.plot_feature_heatmap(datasets['S1'])\n"
         "fig.savefig(FIGURES_DIR / 'eda_feature_heatmap_S1.png', dpi=150)\n"
         "plt.show()"),
    md("## Observations\n"
       "1. **Strong day effect**: mean cost falls from about 1.1 on day 1 to about 0.7 by day 5 for both subjects.\n"
       "2. **Step features** (peak plantarflexion, step time) and a few **EMG channels** (emg2, emg3, emg10) correlate "
       "most with metabolic cost; the strongest predictors differ between subjects.\n"
       "3. **Left/right redundancy**: step times of both legs are almost identical.\n"
       "4. One S2 row has a negative cost (below quiet standing) and is dropped as a sensor error, leaving 179 rows."),
]
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/01_eda.ipynb")
print("wrote notebooks/01_eda.ipynb")
```

**`results/figures/eda_metabolic_by_day.png`** is written automatically when the notebook above is executed.

**`results/figures/eda_feature_correlations.png`** is written automatically when the notebook above is executed.

**`results/figures/eda_feature_heatmap_S1.png`** is written automatically when the notebook above is executed.

Then the commit:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add notebooks/01_eda.ipynb results/figures/eda_metabolic_by_day.png results/figures/eda_feature_correlations.png results/figures/eda_feature_heatmap_S1.png
# save this step under your name
git commit -m "notebook: exploratory data analysis"
```

### A2.4 - commit: `notebook: baselines, CV protocols and leakage check`

Generate **`notebooks/02_baselines.ipynb`** with the builder script below. Save the script **outside the repository** (for example in your system temp folder) as `build_02_baselines.py` - it must not be committed. Then, from the repo root with the virtual environment active:

```bash
python <path-to>/build_02_baselines.py
jupyter nbconvert --to notebook --execute --inplace notebooks/02_baselines.ipynb
```

The executed notebook (with outputs and plots) is what gets committed. Check it contains no error outputs, then delete the builder script.

```python
import os

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 02 - Baselines and evaluation protocols\n"
       "Constant prediction, LASSO and Ridge regression, evaluated under two protocols:\n\n"
       "* **kfold** - shuffled 5-fold CV, repeated 3 times (the reference paper's protocol)\n"
       "* **lodo** - leave-one-day-out CV: train on 4 days, test on the unseen day (our stricter protocol)"),
    code("import sys, pathlib, warnings\n"
         "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
         "warnings.filterwarnings('ignore')\n"
         "from dataclasses import replace\n"
         "import matplotlib.pyplot as plt\n"
         "import pandas as pd\n"
         "from sklearn.linear_model import LassoCV\n"
         "from sklearn.preprocessing import StandardScaler\n"
         "from src.baselines import BASELINES, LASSO_ALPHAS, make_lasso\n"
         "from src.config import FEATURE_GROUPS, FIGURES_DIR\n"
         "from src.data import load_all\n"
         "from src.evaluation import PROTOCOLS, cross_validate_model\n"
         "FIGURES_DIR.mkdir(parents=True, exist_ok=True)\n"
         "datasets = load_all()"),
    md("## Baseline MSE for every subject, feature set and protocol"),
    code("rows = []\n"
         "for subject, data in datasets.items():\n"
         "    for fs in FEATURE_GROUPS:\n"
         "        for protocol in PROTOCOLS:\n"
         "            for name, factory in BASELINES.items():\n"
         "                res = cross_validate_model(factory(), data.subset(fs), protocol)\n"
         "                rows.append(dict(subject=subject, feature_set=fs, protocol=protocol, model=name, mse=res.mse))\n"
         "baselines = pd.DataFrame(rows)\n"
         "baselines.pivot_table(index=['subject', 'feature_set'], columns=['protocol', 'model'], values='mse').round(4)"),
    md("**Reading the table**: the constant model's MSE equals the target variance. Regularised linear models cut the "
       "random-k-fold error by roughly 60-85% (S1 LASSO about 0.009, matching the paper's 0.0089), but the error roughly "
       "doubles when a whole day is held out."),
    md("## Error on each held-out day (LASSO, all features)"),
    code("fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)\n"
         "for ax, (subject, data) in zip(axes, datasets.items()):\n"
         "    res = cross_validate_model(make_lasso(), data, 'lodo')\n"
         "    ax.bar([f'Day {k + 1}' for k in range(len(res.fold_mse))], res.fold_mse, color='#2a78b5')\n"
         "    ax.axhline(res.mse, color='k', ls='--', lw=1, label=f'overall {res.mse:.4f}')\n"
         "    ax.set_title(f'{subject}: LASSO error per held-out day')\n"
         "    ax.legend()\n"
         "axes[0].set_ylabel('MSE')\n"
         "fig.tight_layout()\n"
         "fig.savefig(FIGURES_DIR / 'baseline_lodo_per_day.png', dpi=150)\n"
         "plt.show()"),
    md("## Why scaling must happen inside the CV loop\n"
       "The reference code standardised the whole dataset before splitting it. Under leave-one-day-out this leaks the "
       "held-out day's mean and spread into training. We compare that against our pipeline, which fits the scaler on "
       "training folds only."),
    code("rows = []\n"
         "for subject, data in datasets.items():\n"
         "    leaky_data = replace(data, X=StandardScaler().fit_transform(data.X))\n"
         "    leaky_model = LassoCV(alphas=LASSO_ALPHAS, cv=5, max_iter=50_000)\n"
         "    for protocol in PROTOCOLS:\n"
         "        rows.append(dict(subject=subject, protocol=protocol,\n"
         "                         correct_pipeline=cross_validate_model(make_lasso(), data, protocol).mse,\n"
         "                         scaled_before_split=cross_validate_model(leaky_model, leaky_data, protocol).mse))\n"
         "pd.DataFrame(rows).round(4)"),
    md("With random k-fold the difference is negligible, but under leave-one-day-out the leaky version looks noticeably "
       "better for S2 - an over-optimistic number that the correct pipeline avoids."),
]
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/02_baselines.ipynb")
print("wrote notebooks/02_baselines.ipynb")
```

**`results/figures/baseline_lodo_per_day.png`** is written automatically when the notebook above is executed.

Then the commit:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add notebooks/02_baselines.ipynb results/figures/baseline_lodo_per_day.png
# save this step under your name
git commit -m "notebook: baselines, CV protocols and leakage check"
```

### A2 - open the pull request

Claude runs the full test suite one last time: `python -m pytest -q` (it must say 'passed' with no failures).

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# upload your branch to GitHub
git push -u origin feature/baselines-and-eda
# open the pull request (prints the PR link)
gh pr create --base main --head feature/baselines-and-eda --title "Baseline models and exploratory data analysis" --body "Baselines are needed by the experiment runner and the demo notebook. Part of Manav's work (data, evaluation protocols, baselines, EDA, README)."
```

> **CHECKPOINT** - Claude: ask Manav for the PR link printed above and tell them: "Send this link to Moorty to review and merge." Then give Manav a 4-6 bullet plain-English summary of what this PR does and why, so they can explain it in the Q&A.

> Start this once your PR #2 is merged. It only touches `README.md` and `report/`, so it does not conflict with Moorty's work. Review Moorty's PRs #3-#5 as they arrive (see "Reviewing Moorty's pull requests").

## Step 3 - PR #6: README and write-up skeleton

_Setup/run instructions (a graded item) and the first half of the write-up._

Start a fresh branch from the latest `main`.

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
git checkout main                 # go back to the shared main branch
git pull                          # download everything merged so far
git checkout -b docs/readme-and-report   # create your own working branch for this PR
```

### A3.1 - commit: `docs: README with setup, structure and run instructions`

Create **`README.md`** with exactly this content:

````markdown
# Predicting Metabolic Cost During Human-in-the-Loop Optimization

**UE24CS352A Machine Learning - Mini Project, Team 13 (Problem 18)**

| Member | USN | Responsibility |
|---|---|---|
| Manav Dewangan | PES2UG24CS262 | Data loading, evaluation protocols (k-fold + leave-one-day-out), baselines, EDA |
| Lakshmi Narasimha Moorty Perepa | PES2UG24CS248 | Neural network, nested tuning, feature selection / PCA, experiments, live demo |

## Problem

Human-in-the-loop optimisation (HILO) tunes an ankle exoskeleton's controller while a person walks,
using **metabolic cost** as the objective. Metabolic cost is slow to settle, noisy and needs a
breathing mask in a lab. We predict it from signals that can be measured outside the lab - gait
features and surface EMG - plus the exoskeleton's control parameters.

This project re-implements and extends the CS229 study by Krimsky & Ng (2018) in Python:

* **Reproduction** - LASSO baseline and a one-hidden-layer tanh neural network, compared on four
  feature sets (all, step + EMG, step only, EMG only) with the paper's random k-fold protocol.
* **Extension: leave-one-day-out CV** - train on four walking days and test on a fifth, unseen
  day, the stricter test the paper proposed as future work.
* **Leak-free pipelines** - scaling, hyper-parameter tuning, feature selection and PCA are all fitted
  inside each training fold (the reference code fitted them on the full dataset).

## Dataset

Two subjects x 5 days x 36 controller settings (180 samples per subject, 179 for S2 after removing
one invalid row), 29 features: 9 step features, 16 EMG channels, 4 control parameters. The target is
metabolic cost normalised to normal walking. See [data/README.md](data/README.md).

## Repository layout

```
data/            processed_data.mat + description
src/
  config.py      paths, subjects, feature groups, constants
  data.py        load_subject() -> SubjectData (X, y, day, feature names)
  evaluation.py  k-fold and leave-one-day-out cross-validation, metrics
  baselines.py   constant, LASSO, Ridge
  eda.py         exploratory plots
  models.py      neural network, nested hyper-parameter tuning, neuron sweep
  feature_selection.py  nested forward selection and PCA
  viz.py         result plots
scripts/
  run_experiments.py   runs every experiment -> results/tables/*.csv, results/figures/*.png
notebooks/
  01_eda.ipynb               data exploration
  02_baselines.ipynb         baselines, CV protocols, leakage check
  03_models_and_results.ipynb  neural network, selection, PCA, discussion
  04_live_demo.ipynb         the notebook used for the live demonstration
tests/           pytest unit tests
report/          write-up (LaTeX + PDF) and slides
```

## Setup

Requires Python 3.10+ (tested on 3.11).

```bash
git clone <repo-url>
cd hilo-metabolic-cost
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

```bash
pytest                                   # unit tests (~15 s)
python -m scripts.run_experiments        # all experiments (~10 min) -> results/
python -m scripts.run_experiments --quick  # fast smoke test (~2 min)
jupyter notebook notebooks/              # open the notebooks
```

The results in `results/` are committed, so the notebooks can be opened without re-running the
experiments. For the live demo open `notebooks/04_live_demo.ipynb` and run all cells (~1 min).

## Results

<!-- RESULTS:START -->
_Results are added once the experiments are merged._
<!-- RESULTS:END -->

## Reference

E. Krimsky and E. Ng, "Predicting Metabolic Cost During Human-in-the-Loop Optimization", CS229
project report, Stanford University, 2018. Original MATLAB code: https://github.com/ngeley/cs229project
````

Then the commit:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add README.md
# save this step under your name
git commit -m "docs: README with setup, structure and run instructions"
```

### A3.2 - commit: `docs(report): write-up with problem, dataset and approach`

Create **`report/writeup.tex`** with exactly this content:

```latex
\documentclass[10pt,twocolumn]{article}
\usepackage[a4paper,margin=1.6cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{6pt}{3pt}
\setlength{\parskip}{2pt}

\title{\vspace{-1.2cm}\textbf{Predicting Metabolic Cost During Human-in-the-Loop Optimization}\\
\large UE24CS352A Machine Learning Mini-Project -- Team 13 (Problem 18)}
\author{Manav Dewangan (PES2UG24CS262) \and Lakshmi Narasimha Moorty Perepa (PES2UG24CS248)}
\date{}

\begin{document}
\maketitle
\vspace{-1cm}

%% ---------------- Sections 1-3: Manav Dewangan ----------------
\section{Problem Statement}
Human-in-the-loop optimisation (HILO) tunes the controller of an assistive exoskeleton while a
person walks, using metabolic cost (energy expenditure) as the objective. Metabolic cost is slow to
reach steady state, very noisy, and needs a breathing mask in a laboratory. The goal is to
\emph{predict} normalised metabolic cost from signals that can be measured outside the lab --
gait (step) features and surface EMG -- together with the exoskeleton control parameters. We
re-implement the CS229 study of Krimsky and Ng~\cite{krimsky} in Python and test whether its
accuracy holds when the model must predict a walking day it has never seen.

\section{Dataset}
We use the processed dataset released with~\cite{krimsky}: two naive subjects (S1, S2), each
walking on 5 days; each day is a 72-minute CMA-ES optimisation trial with 36 controller settings
(2\,min each), giving 180 samples per subject (179 for S2 after removing one physically
impossible negative value). Each sample summarises the last 30\,s of a condition with 29 features:
9 step features (per-leg peak vertical force, peak dorsi-/plantar-flexion, step time; step width),
16 EMG RMS values (8 per leg) and 4 control parameters. The target is net metabolic cost divided
by the net cost of normal walking on the same day (variance 0.066 for S1, 0.036 for S2).
Exploratory analysis showed a strong \textbf{day effect}: mean cost falls from about 1.1 on day~1 to
about 0.7 on day~5 as the subject adapts, and left/right step times are almost perfectly
correlated ($r\approx1$).

\section{Approach}
\textbf{Evaluation protocols.} (i) \emph{Random 5-fold CV}, repeated 3 times -- the protocol of
the reference paper, used for comparison; (ii) \emph{leave-one-day-out (LODO) CV} -- train on 4
days and test on the unseen fifth, which the paper proposed as the realistic test. Because
conditions within a day are correlated, random k-fold is expected to be optimistic.
\textbf{Leak-free pipelines.} Every transformation (standardisation, hyper-parameter search,
feature selection, PCA) is fitted on the training folds only. The reference code standardised the
full dataset before splitting; under LODO this made S2's LASSO error look 21\% lower
(0.0243 vs 0.0309), an over-optimistic estimate we avoid.
\textbf{Baselines.} Constant (training mean), LASSO (L1 strength by inner 5-fold CV) and Ridge
(L2 strength by leave-one-out CV), each on four feature sets: all, step+EMG, step only, EMG only.

%% ---------------- Sections 4-6: Moorty Perepa ----------------
%% B-SECTIONS:START
\section{Implementation}
% TODO (Moorty): code structure, neural network, nested tuning, selection/PCA.

\section{Results}
% TODO (Moorty): results table, figure and discussion.

\section{Conclusions}
% TODO (Moorty): conclusions and future work.
%% B-SECTIONS:END

\begin{thebibliography}{9}\small
\bibitem{krimsky} E. Krimsky and E. Ng, ``Predicting Metabolic Cost During Human-in-the-Loop
Optimization,'' CS229 Project Report, Stanford University, 2018.
\bibitem{zhang} J. Zhang \emph{et al.}, ``Human-in-the-loop optimization of exoskeleton
assistance during walking,'' \emph{Science}, 356(6344):1280--1284, 2017.
\end{thebibliography}

\noindent\small Code: private GitHub repository \texttt{hilo-metabolic-cost} (shared with faculty).
\end{document}
```

Then the commit:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add report/writeup.tex
# save this step under your name
git commit -m "docs(report): write-up with problem, dataset and approach"
```

### A3 - open the pull request

Claude runs the full test suite one last time: `python -m pytest -q` (it must say 'passed' with no failures).

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# upload your branch to GitHub
git push -u origin docs/readme-and-report
# open the pull request (prints the PR link)
gh pr create --base main --head docs/readme-and-report --title "README and write-up skeleton" --body "Setup/run instructions (a graded item) and the first half of the write-up. Part of Manav's work (data, evaluation protocols, baselines, EDA, README)."
```

> **CHECKPOINT** - Claude: ask Manav for the PR link printed above and tell them: "Send this link to Moorty to review and merge." Then give Manav a 4-6 bullet plain-English summary of what this PR does and why, so they can explain it in the Q&A.

## Reviewing Moorty's pull requests

You review and merge Moorty's PRs (this is visible evidence of collaboration in the repo):

- PR #3 `feature/neural-network` - Neural network with nested hyper-parameter tuning
- PR #4 `feature/feature-selection` - Nested forward selection and PCA
- PR #5 `feature/experiments-and-demo` - Experiment runner, results and live demo
- PR #7 `docs/results-and-report` - Results in README and write-up

When Moorty sends a PR link (replace `N` with the PR number):

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
git checkout main && git pull     # make sure your main is up to date
gh pr checkout N                    # download the PR's branch to test it locally
```

Claude then runs `python -m pytest -q` and reads through the changed files with Manav,
explaining what they do. If the tests pass:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# approve the PR (shows up on GitHub as your review)
gh pr review N --approve --body "Reviewed: tests pass locally, LGTM."
# merge it with a merge commit, which keeps every individual commit and author
gh pr merge N --merge --delete-branch
# go back to main and download the merged result
git checkout main && git pull
```

Always use **`--merge`** (a merge commit), never squash or rebase - squashing would collapse
Moorty's commits and hide individual contributions. If the tests fail, do not merge:
Manav posts the failure with `gh pr review N --request-changes --body "<what failed>"`.
If GitHub reports a merge conflict, Moorty runs `git pull origin main` on their branch,
resolves it, and pushes again.

## Slides you own (Google Slides deck shared by Moorty)

Keep slides visual: one figure + 3-4 bullets each. Figures are in `results/figures/`.

1. **Title** - project name, Team 13, both names + USNs, course code.
2. **Problem & motivation** - HILO tunes exoskeleton assistance using metabolic cost; metabolic cost is slow (~minutes to steady state), noisy, needs a mask -> predict it from wearable signals.
3. **Dataset** - 2 subjects x 5 days x 36 controller settings; 29 features (9 step, 16 EMG, 4 control); target normalised to normal walking; one invalid row removed.
4. **EDA: the day effect** - figure `eda_metabolic_by_day.png`; cost falls ~1.1 -> ~0.7 over 5 days as the subject adapts; L/R step time r ~ 1 (redundant).
5. **Evaluation protocols** - random 5-fold (paper) vs leave-one-day-out (ours); why LODO is the realistic test; leakage fix: scaling inside folds (S2 LODO LASSO 0.0243 leaky vs 0.0309 correct).
6. **Baselines** - constant / LASSO / Ridge; table from notebook 02; figure `baseline_lodo_per_day.png`.

## Live demo script (Thu 8 Oct)

1. Open `notebooks/04_live_demo.ipynb` in Jupyter/VS Code from the repo root's `.venv` (`jupyter notebook notebooks/04_live_demo.ipynb`).
2. **Kernel -> Restart & Run All** *before* the panel arrives (takes ~15-30 s), so plots are on screen.
3. Walk through: Step 1 data plot (Manav explains), Step 2 live training (Moorty explains), Step 3 results table (Moorty).
4. If asked "what about S2 / EMG only?": change `SUBJECT` / `FEATURE_SET` in the 2nd cell and re-run the cells below it (~15 s).
5. Backup: `python -m pytest -q` in a terminal shows the test-suite passing; `python -m scripts.run_experiments --quick` runs everything end-to-end in ~2 min.

## Q&A preparation (Manav) - know these cold

- **Why leave-one-day-out?** Conditions from the same day share the subject's adaptation level, sensor placement and baseline. Random k-fold puts same-day samples in both train and test, so it is optimistic. Predicting a new day is the realistic use (a new session outside the lab).
- **How do you know which day a row belongs to?** Rows are stored day by day, 36 per day (180 = 5 x 36). We confirmed it: each block of 36 starts with the same initial controller settings (CMA-ES restarts each day).
- **Why drop the negative S2 value?** The target is (cost - quiet standing cost) / (normal walking cost - standing). Negative means walking used less energy than standing still - physically impossible -> sensor error. The reference code also removed it.
- **What is data leakage here?** Using test-fold information during training. The reference scaled the full dataset before CV; we put `StandardScaler` inside a `Pipeline`, so `cross_val_predict` re-fits it on each training fold. Under LODO the leaky version made S2 LASSO look 21% better (0.0243 vs 0.0309).
- **Why LASSO / Ridge?** 29 features vs ~144 training rows -> ordinary least squares over-fits; L1 (LASSO) shrinks and zeroes weak features, L2 (Ridge) shrinks all. Strength chosen by inner CV (`LassoCV` 5-fold, `RidgeCV` efficient leave-one-out).
- **What is the constant baseline and why does its MSE equal the variance?** Predicting the training mean; MSE of predicting the mean = variance of the target. Anything useful must beat it.
- **What does MSE 0.009 mean physically?** RMSE = sqrt(0.009) ~ 0.095, i.e. ~10% of normal-walking metabolic cost. Measurement noise floor is ~0.0025 (~5%).
- **Why repeat k-fold 3 times?** A single random split is noisy on 180 samples; averaging over seeds gives a stabler estimate.
- **Main EDA findings?** Strong day effect (1.1 -> 0.7); best single correlates: plantarflexion angle, step time, EMG 2/3/10; left/right step times redundant (r ~ 1).
- **Code walkthrough:** `load_subject()` -> `SubjectData(X, y, day, feature_names)`; `.subset('emg')` picks a feature group; `cross_validate_model(model, data, 'lodo')` -> `EvalResult(mse, r2, y_pred, fold_mse)`.

## Final checks (Manav, morning of Thu 8 Oct)

Prove the repo works from scratch on a different machine - this is exactly what "code
functionality" is graded on:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
cd <some-temp-folder>
# download a brand-new copy of the repo, exactly as the evaluators will
git clone <repo-url> fresh-check
```

Claude then runs, inside `fresh-check`:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q                                  # all tests pass
python -m scripts.run_experiments --quick            # ~2 min, ends with 'done'
jupyter nbconvert --to notebook --execute --output demo_check.ipynb notebooks/04_live_demo.ipynb
```

Then delete the `fresh-check` folder (nothing from it is committed). Report any failure to Moorty immediately.

## After the demo (Fri 9 - Sat 10 Oct)

1. Review and merge Moorty's PR #7 (results + write-up).
2. Download the slides as PDF (Google Slides -> File -> Download -> PDF). Claude copies it to
   `report/slides.pdf` on a new branch, which Manav creates and commits:

**Manav types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Manav says they worked):

```bash
# start a small branch from the latest main
git checkout main && git pull && git checkout -b docs/slides
# (Claude copies the downloaded PDF to report/slides.pdf at this point)
# stage and commit only the slides
git add report/slides.pdf
git commit -m "docs: presentation slides"
# upload and open the pull request
git push -u origin docs/slides
gh pr create --base main --head docs/slides --title "Presentation slides" --body "Slides used in the review."
```

   Moorty merges it.
3. Proof-read the compiled write-up PDF (sections 1-3 are yours).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Run commands from the repo root; for pytest use `python -m pytest` (pytest.ini adds the root to the path). |
| `InvalidParameterError: The 'loss' parameter of MLPRegressor...` | Someone passed hidden sizes positionally. Newer scikit-learn requires keywords - the code in this guide already uses `hidden_layer_sizes=`. Do not change it. |
| `KeyError: ...joblib_memmapping_folder...` printed after a notebook/script finishes (Windows) | Harmless temp-folder clean-up message from joblib; the run succeeded if the notebook has no error cells. |
| `ConvergenceWarning` spam | Harmless (filtered in scripts/notebooks/tests). |
| Results differ slightly from the expected table | Fine at the 3rd-4th decimal (library/BLAS versions). Larger differences: check you ran without `--quick`. |
| `gh: command not found` | Install GitHub CLI and restart the terminal, or create/merge PRs on github.com (use "Create a merge commit"). |
| Push rejected: `Permission denied` | Manav has not accepted the collaborator invite yet. |
| Merge conflict in a PR | The PR author runs `git checkout <branch> && git pull origin main`; Claude helps resolve the conflicting lines (keep both sides' intended content) and runs `pytest`; the author then runs `git add <file>`, `git commit` and `git push`. |
| Jupyter kernel cannot import `src` | Open the notebook from inside `notebooks/` (the first cell adds the parent folder to `sys.path`) and use the `.venv` kernel. |
