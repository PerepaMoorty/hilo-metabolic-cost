# Team 13 ML Mini-Project - Build guide for Lakshmi Narasimha Moorty Perepa (Person B)

> **Moorty: how to use this file.** Open **Claude Code** in `C:\dev\PES Academics\Sem_5\MachineLearning\MiniProject`, give it this file and say: *"Read this guide and follow it step by step for me."* Claude writes the code, runs the tests and explains everything. **You type every git / gh command yourself** (Claude shows each one with a comment explaining it), so every commit and PR is yours. It also tells you when to send a link to Manav or wait for a merge. Keep Manav updated at every CHECKPOINT.

## 0. Instructions for Claude (read this whole section first)

You are Claude Code helping **Lakshmi Narasimha Moorty Perepa** (PES2UG24CS248) complete their half of a two-person university
machine-learning project. Moorty has uploaded this guide; follow it **step by step, in order**.

Rules:
1. **Exact files.** Create every file with *exactly* the content given in this guide. Do not
   refactor, reformat, rename or "improve" code - the two halves of the project were written and
   tested together and depend on these exact names.
2. **Moorty runs every git and gh command - you never do.** Do not run `git` or `gh` at all
   (not even `git status`, `git init`, `git add` or `git config`). Wherever this guide shows
   "**Moorty types these commands**", show the block, explain each line in one sentence, and wait
   until Moorty says it worked (ask them to paste the output if anything looks wrong). There is one
   commit per `commit:` heading, with that exact message and only the files listed. Small, separate
   commits typed by Moorty are what makes their contribution visible to the evaluators.
3. **Stay in your lane.** Only touch the files assigned to Moorty. Never edit or
   "fix" these files owned by Manav: `data/processed_data.mat`, `data/README.md`, `src/data.py`, `src/evaluation.py`, `tests/test_data.py`, `tests/test_evaluation.py`, `src/baselines.py`, `tests/test_baselines.py`, `src/eda.py`, `notebooks/01_eda.ipynb`, `notebooks/02_baselines.ipynb`. If one of them seems broken, stop and tell
   Moorty to contact Manav.
4. **Tests gate everything.** You run `python -m pytest -q` before every commit and push. If it
   fails, fix the cause before showing Moorty any git command.
   (Before any test file exists, pytest reports "no tests ran" - that is fine.)
5. **Checkpoints.** At every **CHECKPOINT**, stop, tell Moorty exactly what to do (e.g. send a link
   to Manav, wait for a merge) and wait for them to confirm before continuing.
6. **Shell.** Commands are written for bash (Git Bash / macOS / Linux). On Windows PowerShell,
   translate them (e.g. `source .venv/bin/activate` -> `.venv\Scripts\Activate.ps1`, `cp` ->
   `Copy-Item`, `mkdir -p` -> `New-Item -ItemType Directory -Force`). Always use the project's
   virtual environment's `python`.
7. **Keep out of the repo:** `.venv/`, notebook builder scripts, `__pycache__/` and any secrets
   must never be staged (the `git add` lines in this guide already list exact files).
8. **Teach as you go.** After each PR, explain to Moorty in plain English what was built and why;
   they must answer questions about it in a live viva. Section "Q&A preparation" lists what they
   need to know - quiz them on it at the end.
9. If something in this guide does not work as written, stop and explain the problem to Moorty
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

**You are Person B - repository owner, neural network, feature selection, experiments, live demo.**

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

## 3. One-time setup (Moorty)

**Install once** (skip what is already installed):
- Git - https://git-scm.com/downloads
- Python **3.11** (3.10-3.13 work) - https://www.python.org/downloads/ (tick "Add to PATH" on Windows)
- GitHub CLI `gh` - https://cli.github.com/
- VS Code with the Python + Jupyter extensions (optional, for viewing notebooks)

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# log in to GitHub from the terminal (choose GitHub.com -> HTTPS -> login with browser)
gh auth login
# check it worked: should say 'Logged in to github.com account <you>'
gh auth status
```

**Project folder.** Claude creates `hilo-metabolic-cost/` **next to** the `References` folder (the references
are not part of the repo) and writes the Step 1 files into it (see Step 1 below). If
`C:/dev/PES Academics/Sem_5/MachineLearning/MiniProject/hilo-metabolic-cost/` already exists with those
files, Claude only checks they match Step 1 exactly and skips creating them.

Claude then sets up the Python environment inside the repo folder (this is not git, so Claude runs it):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q                 # 'no tests ran' is fine at this stage
```

Then Moorty turns the folder into a git repo, sets his identity and creates the private GitHub repo:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
cd "C:/dev/PES Academics/Sem_5/MachineLearning/MiniProject/hilo-metabolic-cost"
# make this folder a git repository whose main branch is called "main"
git init -b main
# stamp your commits with your name and your GitHub email (only for this repo)
git config user.name "Lakshmi Narasimha Moorty Perepa"
git config user.email "<your-github-email>"
# create an empty PRIVATE repository on GitHub under your account
gh repo create hilo-metabolic-cost --private --description "Predicting metabolic cost during human-in-the-loop optimisation (UE24CS352A Team 13)"
# connect the local folder to that GitHub repository
git remote add origin https://github.com/<your-github-username>/hilo-metabolic-cost.git
# check: should list origin twice (fetch and push)
git remote -v
```

Now do **Step 1 - Repository scaffold** below: its two commits and the push. Then come back here.

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# invite Manav to the private repo (he must accept the invitation)
gh api -X PUT repos/<your-github-username>/hilo-metabolic-cost/collaborators/<manav-github-username>
```

> **CHECKPOINT** - Claude: tell Moorty to send Manav (1) `GUIDE_PersonA_Manav.md` and
> `WORKFLOW_EXPLAINED.md`, (2) the repo URL `https://github.com/<moorty-username>/hilo-metabolic-cost`, and
> (3) a reminder to accept the invite at `https://github.com/<moorty-username>/hilo-metabolic-cost/invitations`.
> Also ask Moorty to create (a) a **Google Slides** deck "Team 13 - Metabolic Cost Prediction"
> shared with Manav (edit access) and (b) an **Overleaf** project (for the write-up later) shared with Manav.

After this checkpoint, Step 1 is already done - continue with the wait note before Step 2.

## Step 1 - Repository scaffold

_Creates the shared skeleton so both of you can start from the same main branch._

This is the very first push, so it goes straight to `main` (there is nothing to review yet).

### B0.1 - commit: `chore: initial project scaffold`

Create **`.gitignore`** with exactly this content:

```text
# Python
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.ipynb_checkpoints/

# Editors / OS
.vscode/
.idea/
.DS_Store
Thumbs.db

# LaTeX build files (commit only the .tex and the final PDF)
report/*.aux
report/*.log
report/*.out
report/*.synctex.gz

# Reference material (original paper, poster, MATLAB code, generated guides) - never part of the repo
References/
references/
```

Create **`.gitattributes`** with exactly this content:

```text
* text=auto
*.mat binary
*.png binary
*.pdf binary
```

Create **`requirements.txt`** with exactly this content:

```text
numpy>=1.26
scipy>=1.11
pandas>=2.1
scikit-learn>=1.4
matplotlib>=3.8
jupyter>=1.0
nbformat>=5.9
nbconvert>=7.0
pytest>=7.0
```

Create **`pytest.ini`** with exactly this content:

```ini
[pytest]
pythonpath = .
testpaths = tests
filterwarnings =
    ignore::sklearn.exceptions.ConvergenceWarning
```

Create **`README.md`** with exactly this content:

```markdown
# Predicting Metabolic Cost During Human-in-the-Loop Optimization

UE24CS352A Machine Learning mini-project - Team 13 (Problem 18).

Manav Dewangan (PES2UG24CS262) and Lakshmi Narasimha Moorty Perepa (PES2UG24CS248).

Work in progress - full setup and run instructions will be added here.
```

Create **`src/__init__.py`** with exactly this content:

```python
"""Predicting metabolic cost during human-in-the-loop optimisation."""
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add .gitignore .gitattributes requirements.txt pytest.ini README.md src/__init__.py
# save this step under your name
git commit -m "chore: initial project scaffold"
```

### B0.2 - commit: `feat(config): shared paths, subjects and feature groups`

Create **`src/config.py`** with exactly this content:

```python
"""Project-wide constants: file paths, subjects, feature groups and reference numbers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed_data.mat"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

RANDOM_STATE = 229

# Each subject walked on 5 separate days; every day is one 72-minute
# optimisation trial = 36 control conditions of 2 minutes each.
N_DAYS = 5
CONDITIONS_PER_DAY = 36

# Subject label used in the paper -> variable prefix inside processed_data.mat
SUBJECTS = {"S1": "michael", "S2": "eley"}

# Column layout of the 29 features (see data/README.md)
STEP_COLS = list(range(0, 9))      # 4 per leg + step width
EMG_COLS = list(range(9, 25))      # 16 EMG channels (8 per leg)
CONTROL_COLS = list(range(25, 29))  # 4 exoskeleton control parameters

FEATURE_GROUPS = {
    "all": STEP_COLS + EMG_COLS + CONTROL_COLS,
    "step_emg": STEP_COLS + EMG_COLS,
    "step": STEP_COLS,
    "emg": EMG_COLS,
}

# Minimum achievable MSE given the noise in metabolic measurements
# (Krimsky & Ng, 2018). Used as a reference line in plots.
NOISE_FLOOR_MSE = 0.0025
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add src/config.py
# save this step under your name
git commit -m "feat(config): shared paths, subjects and feature groups"
```

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# upload the first commits to GitHub and make main track origin/main
git push -u origin main
```

> **Wait** until you (Moorty) have reviewed and merged Manav's **PR #1** - this code imports `src.data` and `src.evaluation`. See "Reviewing Manav's pull requests".

## Step 2 - PR #3: Neural network with nested hyper-parameter tuning

_The paper's main model, re-implemented in scikit-learn._

Start a fresh branch from the latest `main`.

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
git checkout main                 # go back to the shared main branch
git pull                          # download everything merged so far
git checkout -b feature/neural-network   # create your own working branch for this PR
```

### B1.1 - commit: `feat(models): one-hidden-layer tanh network with nested tuning`

Create **`src/models.py`** with exactly this content:

```python
"""Shallow neural-network regressor and its hyper-parameter tuning.

The reference study used MATLAB's ``feedforwardnet`` with one tanh hidden
layer, a linear output and Bayesian-regularisation training (``trainbr``).
Our equivalent in scikit-learn:

* ``MLPRegressor`` with one hidden layer and ``tanh`` activation
  (the output layer of MLPRegressor is always linear),
* an L2 weight penalty ``alpha`` in place of Bayesian regularisation
  (both shrink the weights to stop a tiny, noisy dataset being over-fitted),
* the quasi-Newton ``lbfgs`` solver, which converges well on small datasets,
* inputs AND target standardised using the training folds only.
"""
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE
from src.evaluation import cross_validate_model

NEURON_SWEEP = range(1, 11)          # hidden sizes plotted, as in the paper's Fig. 1
HIDDEN_UNIT_GRID = (1, 2, 3, 4, 6, 8)  # hidden sizes searched by the tuned model
ALPHA_GRID = (0.1, 1.0, 10.0)        # L2 strengths searched by the tuned model


def make_nn(hidden_units: int = 3, alpha: float = 1.0, seed: int = RANDOM_STATE):
    """Standardise -> 1-hidden-layer tanh MLP -> linear output (target also standardised)."""
    mlp = MLPRegressor(
        hidden_layer_sizes=(hidden_units,),
        activation="tanh",
        solver="lbfgs",
        alpha=alpha,
        max_iter=3000,
        random_state=seed,
    )
    net = Pipeline([("scale", StandardScaler()), ("mlp", mlp)])
    return TransformedTargetRegressor(regressor=net, transformer=StandardScaler())


def make_tuned_nn(seed: int = RANDOM_STATE, hidden_grid=HIDDEN_UNIT_GRID, alpha_grid=ALPHA_GRID):
    """NN whose hidden size and L2 strength are chosen by an inner 4-fold CV.

    When this object is itself cross-validated, the inner search only ever
    sees the outer training folds (nested CV), so the reported error is not
    biased by picking the best configuration on the test data.
    """
    grid = {
        "regressor__mlp__hidden_layer_sizes": [(h,) for h in hidden_grid],
        "regressor__mlp__alpha": list(alpha_grid),
    }
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=seed)
    return GridSearchCV(make_nn(seed=seed), grid, scoring="neg_mean_squared_error", cv=inner_cv, n_jobs=1)


def neuron_sweep(data, protocol: str, neurons=NEURON_SWEEP, alpha: float = 1.0, n_repeats: int = 3) -> pd.DataFrame:
    """Cross-validated MSE for each hidden-layer size (reproduces the paper's Fig. 1 curves)."""
    rows = []
    for h in neurons:
        res = cross_validate_model(make_nn(h, alpha), data, protocol, n_repeats=n_repeats)
        rows.append({"neurons": h, "mse": res.mse, "mse_std": res.mse_std})
    return pd.DataFrame(rows)
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add src/models.py
# save this step under your name
git commit -m "feat(models): one-hidden-layer tanh network with nested tuning"
```

### B1.2 - commit: `test: neural network unit tests`

Create **`tests/test_models.py`** with exactly this content:

```python
import numpy as np

from src.data import load_subject
from src.models import make_nn, make_tuned_nn, neuron_sweep


def _toy(n=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = np.tanh(X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.normal(size=n)
    return X, y


def test_nn_learns_simple_function():
    X, y = _toy()
    model = make_nn(hidden_units=4, alpha=0.1).fit(X[:100], y[:100])
    mse = np.mean((model.predict(X[100:]) - y[100:]) ** 2)
    assert mse < 0.1 * y.var()


def test_tuned_nn_picks_from_grid():
    X, y = _toy()
    search = make_tuned_nn(hidden_grid=(1, 3), alpha_grid=(1.0,)).fit(X, y)
    assert search.best_params_["regressor__mlp__hidden_layer_sizes"] in [(1,), (3,)]


def test_neuron_sweep_on_real_data():
    d = load_subject("S1").subset("step")
    sweep = neuron_sweep(d, "lodo", neurons=(1, 2), n_repeats=1)
    assert list(sweep.neurons) == [1, 2]
    assert (sweep.mse > 0).all()
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add tests/test_models.py
# save this step under your name
git commit -m "test: neural network unit tests"
```

### B1 - open the pull request

Claude runs the full test suite one last time: `python -m pytest -q` (it must say 'passed' with no failures).

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# upload your branch to GitHub
git push -u origin feature/neural-network
# open the pull request (prints the PR link)
gh pr create --base main --head feature/neural-network --title "Neural network with nested hyper-parameter tuning" --body "The paper's main model, re-implemented in scikit-learn. Part of Moorty's work (repository owner, neural network, feature selection, experiments, live demo)."
```

> **CHECKPOINT** - Claude: ask Moorty for the PR link printed above and tell them: "Send this link to Manav to review and merge." Then give Moorty a 4-6 bullet plain-English summary of what this PR does and why, so they can explain it in the Q&A.

## Step 3 - PR #4: Nested forward selection and PCA

_Dimensionality-reduction experiments from the paper, done without leakage._

Start a fresh branch from the latest `main`.

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
git checkout main                 # go back to the shared main branch
git pull                          # download everything merged so far
git checkout -b feature/feature-selection   # create your own working branch for this PR
```

### B2.1 - commit: `feat(selection): nested forward selection and PCA sweep`

Create **`src/feature_selection.py`** with exactly this content:

```python
"""Dimensionality reduction: forward stepwise selection and PCA, both nested inside CV.

The reference study ran selection/PCA on the full dataset before testing,
which leaks test information into the chosen features. Here every selector
is fitted on the training folds only and then applied to the held-out fold.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation import cross_validate_model, make_splitter
from src.models import make_nn

PCA_COMPONENTS = (1, 2, 3, 5, 8, 12, 16, 20, 25, 29)


def make_forward_selector_model(tol: float = 1e-4, hidden_units: int = 3, alpha: float = 1.0):
    """Standardise -> greedy forward selection (scored with a fast ridge model) -> NN.

    Features are added one at a time while the inner-CV MSE improves by more
    than ``tol``.
    """
    selector = SequentialFeatureSelector(
        RidgeCV(alphas=np.logspace(-3, 3, 13)),
        n_features_to_select="auto",
        tol=tol,
        direction="forward",
        scoring="neg_mean_squared_error",
        cv=5,
    )
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", selector),
        ("model", make_nn(hidden_units, alpha)),
    ])


def forward_selection(data, protocol: str, tol: float = 1e-4):
    """Run nested forward selection.

    Returns ``(mse, frequency)`` where ``frequency`` is the fraction of outer
    folds in which each feature was selected - a stable measure of importance
    that answers "which signals matter?" without trusting one greedy run.
    """
    splitter = make_splitter(protocol)
    X, y = data.X, data.y
    y_pred = np.empty_like(y)
    counts = np.zeros(X.shape[1])
    n_folds = 0
    groups = data.day if protocol == "lodo" else None
    for train, test in splitter.split(X, y, groups):
        model = clone(make_forward_selector_model(tol=tol)).fit(X[train], y[train])
        y_pred[test] = model.predict(X[test])
        counts += model.named_steps["select"].get_support()
        n_folds += 1
    frequency = pd.Series(counts / n_folds, index=data.feature_names, name="frequency")
    return float(mean_squared_error(y, y_pred)), frequency.sort_values(ascending=False)


def make_pca_model(n_components: int, hidden_units: int = 3, alpha: float = 1.0):
    """Standardise -> PCA(n_components) -> NN."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("model", make_nn(hidden_units, alpha)),
    ])


def pca_sweep(data, protocol: str, components=PCA_COMPONENTS, n_repeats: int = 1) -> pd.DataFrame:
    """Cross-validated MSE versus number of principal components.

    ``explained_variance`` is computed on the full dataset purely for
    reporting; the models themselves fit PCA on training folds only.
    """
    components = [k for k in components if k <= data.X.shape[1]]
    full_pca = PCA().fit(StandardScaler().fit_transform(data.X))
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    rows = []
    for k in components:
        res = cross_validate_model(make_pca_model(k), data, protocol, n_repeats=n_repeats)
        rows.append({"components": k, "mse": res.mse, "explained_variance": float(cumulative[k - 1])})
    return pd.DataFrame(rows)
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add src/feature_selection.py
# save this step under your name
git commit -m "feat(selection): nested forward selection and PCA sweep"
```

### B2.2 - commit: `test: feature selection and PCA tests`

Create **`tests/test_feature_selection.py`** with exactly this content:

```python
from src.data import load_subject
from src.feature_selection import forward_selection, pca_sweep


def test_forward_selection_frequencies():
    d = load_subject("S1").subset("step")
    mse, freq = forward_selection(d, "lodo")
    assert mse > 0
    assert len(freq) == 9
    assert freq.between(0, 1).all()


def test_pca_sweep_explained_variance():
    d = load_subject("S1").subset("step")
    pca = pca_sweep(d, "kfold", components=(2, 9))
    assert list(pca.components) == [2, 9]
    assert pca.explained_variance.iloc[-1] > 0.99
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add tests/test_feature_selection.py
# save this step under your name
git commit -m "test: feature selection and PCA tests"
```

### B2 - open the pull request

Claude runs the full test suite one last time: `python -m pytest -q` (it must say 'passed' with no failures).

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# upload your branch to GitHub
git push -u origin feature/feature-selection
# open the pull request (prints the PR link)
gh pr create --base main --head feature/feature-selection --title "Nested forward selection and PCA" --body "Dimensionality-reduction experiments from the paper, done without leakage. Part of Moorty's work (repository owner, neural network, feature selection, experiments, live demo)."
```

> **CHECKPOINT** - Claude: ask Moorty for the PR link printed above and tell them: "Send this link to Manav to review and merge." Then give Moorty a 4-6 bullet plain-English summary of what this PR does and why, so they can explain it in the Q&A.

> **Wait** until Manav's **PR #2** (baselines + EDA) is merged by you and your PRs #3 and #4 are merged by Manav - the runner imports `src.baselines`, `src.models`, `src.feature_selection`.

## Step 4 - PR #5: Experiment runner, results and live demo

_Produces every table and figure, and the notebook used in the live demo._

Start a fresh branch from the latest `main`.

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
git checkout main                 # go back to the shared main branch
git pull                          # download everything merged so far
git checkout -b feature/experiments-and-demo   # create your own working branch for this PR
```

### B3.1 - commit: `feat(viz): result plotting helpers`

Create **`src/viz.py`** with exactly this content:

```python
"""Plots of experiment results (read from the CSVs in results/tables)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CONDITIONS_PER_DAY, NOISE_FLOOR_MSE

FEATURE_SET_LABELS = {"all": "All features", "step_emg": "Step + EMG", "step": "Step only", "emg": "EMG only"}
PROTOCOL_STYLE = {"kfold": ("#2a78b5", "Random 5-fold (paper)"), "lodo": ("#d4622a", "Leave-one-day-out")}
MODEL_COLORS = {"constant": "#9e9e9e", "lasso": "#2a78b5", "ridge": "#7fb2dd", "nn": "#d4622a"}


def plot_neuron_sweep(sweep: pd.DataFrame, subject: str):
    """MSE vs hidden-layer size for each feature set and both CV protocols."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, (fs, label) in zip(axes.ravel(), FEATURE_SET_LABELS.items()):
        part = sweep[(sweep.subject == subject) & (sweep.feature_set == fs)]
        for protocol, (color, plabel) in PROTOCOL_STYLE.items():
            p = part[part.protocol == protocol]
            ax.plot(p.neurons, p.mse, "o-", color=color, label=plabel)
        ax.axhline(NOISE_FLOOR_MSE, color="k", ls=":", lw=1, label="Noise floor")
        ax.set_title(label)
        ax.set_ylim(bottom=0)
    for ax in axes[1]:
        ax.set_xlabel("Hidden neurons")
    for ax in axes[:, 0]:
        ax.set_ylabel("CV MSE")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{subject}: neural network size vs cross-validated MSE")
    fig.tight_layout()
    return fig


def plot_model_comparison(comparison: pd.DataFrame, subject: str):
    """Grouped bars: MSE of each model on each feature set, one panel per protocol."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    models = [m for m in MODEL_COLORS if m in comparison.model.unique()]
    width = 0.8 / len(models)
    x = np.arange(len(FEATURE_SET_LABELS))
    for ax, (protocol, (_, plabel)) in zip(axes, PROTOCOL_STYLE.items()):
        part = comparison[(comparison.subject == subject) & (comparison.protocol == protocol)]
        for i, m in enumerate(models):
            vals = [part[(part.model == m) & (part.feature_set == fs)].mse.mean() for fs in FEATURE_SET_LABELS]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, color=MODEL_COLORS[m], label=m)
        ax.axhline(NOISE_FLOOR_MSE, color="k", ls=":", lw=1)
        ax.set_xticks(x, FEATURE_SET_LABELS.values())
        ax.set_title(plabel)
    axes[0].set_ylabel("CV MSE (lower is better)")
    axes[0].legend()
    fig.suptitle(f"{subject}: model comparison")
    fig.tight_layout()
    return fig


def plot_lodo_predictions(predictions: pd.DataFrame, subject: str):
    """Measured vs predicted metabolic cost when each day is held out in turn."""
    part = predictions[predictions.subject == subject].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(part.y_true, "k.-", lw=1, label="Measured")
    for col, color in [("pred_lasso", MODEL_COLORS["lasso"]), ("pred_nn", MODEL_COLORS["nn"])]:
        ax.plot(part[col], "-", color=color, lw=1.4, label=col.replace("pred_", "Predicted "))
    for b in np.flatnonzero(np.diff(part.day)) + 0.5:
        ax.axvline(b, color="grey", lw=0.8, ls="--")
    ax.set_xlabel(f"Condition (dashed lines separate days of ~{CONDITIONS_PER_DAY})")
    ax.set_ylabel("Metabolic cost (norm.)")
    ax.set_title(f"{subject}: leave-one-day-out predictions (all features)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def plot_selection_frequency(frequency: pd.DataFrame, top: int = 15):
    """How often each feature was chosen by nested forward selection."""
    subjects = frequency.subject.unique()
    fig, axes = plt.subplots(1, len(subjects), figsize=(6 * len(subjects), 5))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, subjects):
        part = frequency[(frequency.subject == s) & (frequency.protocol == "lodo")]
        part = part.sort_values("frequency", ascending=False).head(top)
        ax.barh(part.feature, part.frequency, color=MODEL_COLORS["nn"])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of held-out days where selected")
        ax.set_title(f"{s}: forward-selection frequency (top {top})")
    fig.tight_layout()
    return fig


def plot_pca_sweep(pca: pd.DataFrame):
    """MSE vs number of principal components, with cumulative explained variance."""
    subjects = pca.subject.unique()
    fig, axes = plt.subplots(1, len(subjects), figsize=(6 * len(subjects), 4))
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, subjects):
        part = pca[pca.subject == s]
        for protocol, (color, plabel) in PROTOCOL_STYLE.items():
            p = part[part.protocol == protocol]
            ax.plot(p.components, p.mse, "o-", color=color, label=plabel)
        ax.set_xlabel("Principal components")
        ax.set_ylabel("CV MSE")
        ax.set_title(f"{s}: PCA + NN")
        ev = ax.twinx()
        p = part[part.protocol == "kfold"]
        ev.plot(p.components, p.explained_variance, "k:", label="Explained variance")
        ev.set_ylim(0, 1.05)
        ev.set_ylabel("Cumulative explained variance")
        ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    return fig
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add src/viz.py
# save this step under your name
git commit -m "feat(viz): result plotting helpers"
```

### B3.2 - commit: `feat: experiment runner script`

Create **`scripts/__init__.py`** as an empty file.

Create **`scripts/run_experiments.py`** with exactly this content:

```python
"""Run every experiment and write tables + figures to results/.

Usage (from the repository root):
    python -m scripts.run_experiments           # full run, ~5-10 minutes
    python -m scripts.run_experiments --quick   # smoke test, ~1 minute
"""
import argparse
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.baselines import BASELINES  # noqa: E402
from src.config import FEATURE_GROUPS, FIGURES_DIR, TABLES_DIR  # noqa: E402
from src.data import load_all  # noqa: E402
from src.evaluation import PROTOCOLS, cross_validate_model  # noqa: E402
from src.feature_selection import forward_selection, pca_sweep  # noqa: E402
from src.models import make_tuned_nn, neuron_sweep  # noqa: E402
from src import viz  # noqa: E402

warnings.filterwarnings("ignore")

# Table I of Krimsky & Ng (2018): random k-fold MSE, (lasso, nn)
PAPER_TABLE = {
    ("S1", "all"): (0.0089, 0.0089), ("S1", "step_emg"): (0.0089, 0.0104),
    ("S1", "step"): (0.0200, 0.0138), ("S1", "emg"): (0.0158, 0.0118),
    ("S2", "all"): (0.0176, 0.0301), ("S2", "step_emg"): (0.0168, 0.0313),
    ("S2", "step"): (0.0217, 0.0233), ("S2", "emg"): (0.0203, 0.0199),
}
PAPER_CONSTANT = {"S1": 0.0657, "S2": 0.0465}


def log(msg, start):
    print(f"[{time.time() - start:6.1f}s] {msg}", flush=True)


def run_model_comparison(datasets, n_repeats, nn_kwargs, start):
    rows, predictions = [], []
    for subject, data in datasets.items():
        for fs in FEATURE_GROUPS:
            d = data.subset(fs)
            for protocol in PROTOCOLS:
                models = {name: factory() for name, factory in BASELINES.items()}
                models["nn"] = make_tuned_nn(**nn_kwargs)
                preds = {}
                for name, model in models.items():
                    res = cross_validate_model(model, d, protocol, n_repeats=n_repeats)
                    rows.append({"subject": subject, "feature_set": fs, "protocol": protocol, "model": name,
                                 "mse": res.mse, "mse_std": res.mse_std, "rmse": res.rmse, "r2": res.r2})
                    preds[name] = res
                if fs == "all" and protocol == "lodo":
                    for i in range(d.n_samples):
                        predictions.append({"subject": subject, "day": int(d.day[i]), "y_true": d.y[i],
                                            "pred_lasso": preds["lasso"].y_pred[i], "pred_nn": preds["nn"].y_pred[i]})
                log(f"compare {subject} {fs:8s} {protocol:5s} "
                    + " ".join(f"{m}={r.mse:.4f}" for m, r in preds.items()), start)
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def add_paper_columns(comparison):
    def paper(row):
        if row.protocol != "kfold":
            return None
        if row.model == "constant":
            return PAPER_CONSTANT[row.subject]
        if row.model in ("lasso", "nn"):
            return PAPER_TABLE[(row.subject, row.feature_set)][0 if row.model == "lasso" else 1]
        return None
    comparison["paper_mse"] = comparison.apply(paper, axis=1)
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="tiny grids and 1 repeat, for smoke testing")
    args = parser.parse_args()

    n_repeats = 1 if args.quick else 3
    neurons = (1, 2, 4) if args.quick else range(1, 11)
    nn_kwargs = {"hidden_grid": (2, 4), "alpha_grid": (1.0,)} if args.quick else {}
    pca_components = (2, 8, 29) if args.quick else (1, 2, 3, 5, 8, 12, 16, 20, 25, 29)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    datasets = load_all()

    # 1. Baselines vs tuned neural network, 4 feature sets x 2 protocols
    comparison, predictions = run_model_comparison(datasets, n_repeats, nn_kwargs, start)
    comparison = add_paper_columns(comparison)
    comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    predictions.to_csv(TABLES_DIR / "lodo_predictions.csv", index=False)

    # 2. Neural-network size sweep (paper Fig. 1, plus LODO)
    sweeps = []
    for subject, data in datasets.items():
        for fs in FEATURE_GROUPS:
            for protocol in PROTOCOLS:
                s = neuron_sweep(data.subset(fs), protocol, neurons=neurons, n_repeats=n_repeats)
                sweeps.append(s.assign(subject=subject, feature_set=fs, protocol=protocol))
        log(f"neuron sweep {subject} done", start)
    sweep = pd.concat(sweeps, ignore_index=True)
    sweep.to_csv(TABLES_DIR / "neuron_sweep.csv", index=False)

    # 3. Nested forward selection and PCA on all features
    freq_rows, sel_rows, pca_rows = [], [], []
    for subject, data in datasets.items():
        for protocol in PROTOCOLS:
            mse, freq = forward_selection(data, protocol)
            sel_rows.append({"subject": subject, "protocol": protocol, "mse": mse,
                             "features_always_selected": int((freq == 1).sum())})
            freq_rows.append(freq.rename_axis("feature").reset_index().assign(subject=subject, protocol=protocol))
            pca_rows.append(pca_sweep(data, protocol, components=pca_components).assign(subject=subject, protocol=protocol))
            log(f"selection {subject} {protocol}: forward-selection MSE={mse:.4f}", start)
    frequency = pd.concat(freq_rows, ignore_index=True)
    pca = pd.concat(pca_rows, ignore_index=True)
    frequency.to_csv(TABLES_DIR / "selection_frequency.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(TABLES_DIR / "selection_summary.csv", index=False)
    pca.to_csv(TABLES_DIR / "pca_sweep.csv", index=False)

    # 4. Figures
    figures = {}
    for subject in datasets:
        figures[f"neuron_sweep_{subject}"] = viz.plot_neuron_sweep(sweep, subject)
        figures[f"model_comparison_{subject}"] = viz.plot_model_comparison(comparison, subject)
        figures[f"lodo_predictions_{subject}"] = viz.plot_lodo_predictions(predictions, subject)
    figures["selection_frequency"] = viz.plot_selection_frequency(frequency)
    figures["pca_sweep"] = viz.plot_pca_sweep(pca)
    for name, fig in figures.items():
        fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150)
        plt.close(fig)

    summary = comparison[comparison.feature_set == "all"].pivot_table(
        index=["subject", "model"], columns="protocol", values="mse").round(4)
    print("\nMSE with all features:\n", summary)
    log(f"done - tables in {TABLES_DIR}, figures in {FIGURES_DIR}", start)


if __name__ == "__main__":
    main()
```

Claude runs the tests first (they must pass): `python -m pytest -q`

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add scripts/__init__.py scripts/run_experiments.py
# save this step under your name
git commit -m "feat: experiment runner script"
```

### B3.3 - commit: `results: tables and figures from the full experiment run`

Run the full experiment suite (about 5 minutes on a laptop; it prints progress). Do **not** use `--quick` for the committed results.

```bash
python -m scripts.run_experiments
```

At the end it prints an "MSE with all features" table. Expected values (small differences in the 3rd-4th decimal are fine across machines / library versions):

```text
protocol           kfold    lodo
subject model
S1      constant  0.0666  0.0917
        lasso     0.0091  0.0205
        nn        0.0101  0.0223
        ridge     0.0086  0.0143
S2      constant  0.0362  0.0478
        lasso     0.0135  0.0309
        nn        0.0150  0.0315
        ridge     0.0146  0.0358
```

If a number differs by more than about 0.003, stop and tell Moorty before committing.

`results/tables/*.csv` - produced by the experiment run (see the command in this step).

`results/figures/{model_comparison,neuron_sweep,lodo_predictions}_S*.png, selection_frequency.png, pca_sweep.png` - produced by the experiment run (see the command in this step).

Commit only these result files. The `eda_*` and `baseline_*` figures belong to Manav's PR #2 and must not change. If they show up as modified, Moorty undoes that with `git checkout -- results/figures/eda_* results/figures/baseline_*`.

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add results/tables results/figures/model_comparison_S1.png results/figures/model_comparison_S2.png results/figures/neuron_sweep_S1.png results/figures/neuron_sweep_S2.png results/figures/lodo_predictions_S1.png results/figures/lodo_predictions_S2.png results/figures/selection_frequency.png results/figures/pca_sweep.png
# save this step under your name
git commit -m "results: tables and figures from the full experiment run"
```

### B3.4 - commit: `notebook: models, selection and results discussion`

Generate **`notebooks/03_models_and_results.ipynb`** with the builder script below. Save the script **outside the repository** (for example in your system temp folder) as `build_03_models.py` - it must not be committed. Then, from the repo root with the virtual environment active:

```bash
python <path-to>/build_03_models.py
jupyter nbconvert --to notebook --execute --inplace notebooks/03_models_and_results.ipynb
```

The executed notebook (with outputs and plots) is what gets committed. Check it contains no error outputs, then delete the builder script.

```python
import os

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# 03 - Neural network, feature selection and results\n"
       "This notebook reads the tables produced by `python -m scripts.run_experiments` "
       "(run it first; it takes about 10 minutes) and discusses the results."),
    code("import sys, pathlib, warnings\n"
         "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
         "warnings.filterwarnings('ignore')\n"
         "import matplotlib.pyplot as plt\n"
         "import pandas as pd\n"
         "from src.config import TABLES_DIR\n"
         "from src import viz\n"
         "comparison = pd.read_csv(TABLES_DIR / 'model_comparison.csv')\n"
         "sweep = pd.read_csv(TABLES_DIR / 'neuron_sweep.csv')\n"
         "predictions = pd.read_csv(TABLES_DIR / 'lodo_predictions.csv')\n"
         "frequency = pd.read_csv(TABLES_DIR / 'selection_frequency.csv')\n"
         "selection = pd.read_csv(TABLES_DIR / 'selection_summary.csv')\n"
         "pca = pd.read_csv(TABLES_DIR / 'pca_sweep.csv')"),
    md("## 1. Reproducing the paper (random 5-fold CV)\n"
       "`paper_mse` is Table I of Krimsky & Ng (2018). Our `nn` is tuned by *nested* CV (hidden size and L2 strength "
       "are picked on training folds only), so it is a slightly stricter number than the paper's best-of-sweep value."),
    code("kf = comparison[(comparison.protocol == 'kfold') & comparison.model.isin(['constant', 'lasso', 'nn'])]\n"
         "kf.pivot_table(index=['subject', 'feature_set'], columns='model', values=['mse', 'paper_mse']).round(4)"),
    md("## 2. Neural-network size (paper Fig. 1, now with leave-one-day-out)"),
    code("for s in ['S1', 'S2']:\n"
         "    viz.plot_neuron_sweep(sweep, s)\n"
         "plt.show()"),
    md("As in the paper, small networks (1-4 neurons) are enough; more neurons do not help on 180 samples."),
    md("## 3. Random k-fold vs leave-one-day-out"),
    code("summary = comparison.pivot_table(index=['subject', 'feature_set', 'model'], columns='protocol', values='mse')\n"
         "summary['lodo / kfold'] = summary.lodo / summary.kfold\n"
         "summary.round(4)"),
    code("for s in ['S1', 'S2']:\n"
         "    viz.plot_model_comparison(comparison, s)\n"
         "plt.show()"),
    code("for s in ['S1', 'S2']:\n"
         "    viz.plot_lodo_predictions(predictions, s)\n"
         "plt.show()"),
    md("## 4. Dimensionality reduction (nested inside CV)"),
    code("selection.round(4)"),
    code("viz.plot_selection_frequency(frequency)\n"
         "plt.show()"),
    code("viz.plot_pca_sweep(pca)\n"
         "plt.show()"),
    md("## Findings\n"
       "1. **Reproduction**: with the paper's random k-fold protocol we obtain errors of the same size as Table I "
       "(S1, all features: LASSO 0.0091 vs 0.0089 in the paper), far below the constant-prediction error (0.067).\n"
       "2. **Generalising to a new day is much harder**: holding out a whole day roughly doubles the error "
       "(S1 LASSO 0.0091 -> 0.0205, S2 0.0135 -> 0.0309). For S2 the best model explains only about 14% of the "
       "variance of an unseen day. Random k-fold lets the model see other conditions from the same day, so it "
       "over-states real-world accuracy.\n"
       "3. **Linear vs neural**: with all features, regularised linear models are as good as the nested-tuned NN "
       "(S1 Ridge 0.0086 / 0.0143 for k-fold / LODO). The NN only helps on the small feature sets "
       "(step-only, EMG-only), where some non-linearity is useful. Under LODO, larger networks get worse "
       "(see the neuron sweep), i.e. they over-fit day-specific patterns.\n"
       "4. **Feature sets**: step + EMG beats either group alone. The 4 control parameters help clearly under LODO "
       "(S2 LASSO 0.0448 -> 0.0309), probably because the optimiser's settings drift across days and so encode "
       "how far the subject has adapted.\n"
       "5. **Selection / PCA**: nested forward selection is worse than using all features (LODO MSE about 0.046), and "
       "the selected features change between held-out days - the same instability the paper reports. PCA helps a "
       "little under k-fold (S1: 12 components, 0.0094) but is erratic under LODO, because components are chosen "
       "for variance, not for predicting the target."),
]
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/03_models_and_results.ipynb")
print("wrote notebooks/03_models_and_results.ipynb")
```

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add notebooks/03_models_and_results.ipynb
# save this step under your name
git commit -m "notebook: models, selection and results discussion"
```

### B3.5 - commit: `notebook: live demo`

Generate **`notebooks/04_live_demo.ipynb`** with the builder script below. Save the script **outside the repository** (for example in your system temp folder) as `build_04_demo.py` - it must not be committed. Then, from the repo root with the virtual environment active:

```bash
python <path-to>/build_04_demo.py
jupyter nbconvert --to notebook --execute --inplace notebooks/04_live_demo.ipynb
```

The executed notebook (with outputs and plots) is what gets committed. Check it contains no error outputs, then delete the builder script.

```python
import os

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Live demo - Predicting metabolic cost during human-in-the-loop optimisation\n"
       "**Team 13** - Manav Dewangan (PES2UG24CS262), Lakshmi Narasimha Moorty Perepa (PES2UG24CS248)\n\n"
       "Metabolic cost is slow, noisy and needs a lab mask to measure. Can we predict it from wearable signals "
       "(gait + EMG) recorded while an exoskeleton controller is being optimised?\n\n"
       "Run the cells top to bottom. Change `SUBJECT` / `FEATURE_SET` below to answer panel questions live."),
    code("SUBJECT = 'S1'        # 'S1' or 'S2'\n"
         "FEATURE_SET = 'all'   # 'all', 'step_emg', 'step', 'emg'"),
    code("import sys, pathlib, time, warnings\n"
         "sys.path.insert(0, str(pathlib.Path.cwd().parent))\n"
         "warnings.filterwarnings('ignore')\n"
         "import matplotlib.pyplot as plt\n"
         "import numpy as np\n"
         "import pandas as pd\n"
         "from src.baselines import make_constant, make_lasso\n"
         "from src.config import NOISE_FLOOR_MSE, TABLES_DIR\n"
         "from src.data import load_all\n"
         "from src.eda import plot_metabolic_by_day\n"
         "from src.evaluation import cross_validate_model\n"
         "from src.models import make_nn"),
    md("## Step 1 - The data: 5 walking days x 36 controller settings per subject"),
    code("datasets = load_all()\n"
         "data = datasets[SUBJECT].subset(FEATURE_SET)\n"
         "print(f'{SUBJECT}: {data.n_samples} samples, {data.X.shape[1]} features ({FEATURE_SET})')\n"
         "plot_metabolic_by_day(datasets)\n"
         "plt.show()"),
    md("## Step 2 - Train live, testing on a day the model has never seen"),
    code("models = {\n"
         "    'constant': make_constant(),\n"
         "    'lasso': make_lasso(),\n"
         "    'neural net (3 tanh units)': make_nn(hidden_units=3, alpha=1.0),\n"
         "}\n"
         "results = {}\n"
         "for name, model in models.items():\n"
         "    t = time.time()\n"
         "    results[name] = {p: cross_validate_model(model, data, p) for p in ['kfold', 'lodo']}\n"
         "    print(f'{name:28s} random 5-fold MSE = {results[name][\"kfold\"].mse:.4f} | '\n"
         "          f'leave-one-day-out MSE = {results[name][\"lodo\"].mse:.4f}  ({time.time() - t:.1f}s)')\n"
         "print(f'noise floor of metabolic measurement = {NOISE_FLOOR_MSE}')"),
    code("fig, ax = plt.subplots(figsize=(11, 3.8))\n"
         "ax.plot(data.y, 'k.-', lw=1, label='Measured')\n"
         "ax.plot(results['lasso']['lodo'].y_pred, label='LASSO (day held out)')\n"
         "ax.plot(results['neural net (3 tanh units)']['lodo'].y_pred, label='NN (day held out)')\n"
         "for b in np.flatnonzero(np.diff(data.day)) + 0.5:\n"
         "    ax.axvline(b, color='grey', ls='--', lw=0.8)\n"
         "ax.set_xlabel('Condition (dashed = new day)'); ax.set_ylabel('Metabolic cost (norm.)')\n"
         "ax.legend(ncol=3); ax.set_title(f'{SUBJECT}: leave-one-day-out predictions')\n"
         "plt.show()"),
    md("## Step 3 - Full results (pre-computed by `python -m scripts.run_experiments`)"),
    code("comparison = pd.read_csv(TABLES_DIR / 'model_comparison.csv')\n"
         "table = comparison[comparison.feature_set == FEATURE_SET].pivot_table(\n"
         "    index=['subject', 'model'], columns='protocol', values=['mse', 'paper_mse'])\n"
         "table.round(4)"),
    md("## Take-aways\n"
       "* With the paper's random k-fold protocol we reproduce its accuracy (MSE about 0.009 for S1).\n"
       "* Holding out a whole day - the realistic use case - roughly doubles the error: day-to-day adaptation is the "
       "main obstacle, not model capacity.\n"
       "* Simple regularised linear models match or beat the shallow neural network on this small dataset."),
]
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
os.makedirs("notebooks", exist_ok=True)
nbf.write(nb, "notebooks/04_live_demo.ipynb")
print("wrote notebooks/04_live_demo.ipynb")
```

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add notebooks/04_live_demo.ipynb
# save this step under your name
git commit -m "notebook: live demo"
```

### B3 - open the pull request

Claude runs the full test suite one last time: `python -m pytest -q` (it must say 'passed' with no failures).

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# upload your branch to GitHub
git push -u origin feature/experiments-and-demo
# open the pull request (prints the PR link)
gh pr create --base main --head feature/experiments-and-demo --title "Experiment runner, results and live demo" --body "Produces every table and figure, and the notebook used in the live demo. Part of Moorty's work (repository owner, neural network, feature selection, experiments, live demo)."
```

> **CHECKPOINT** - Claude: ask Moorty for the PR link printed above and tell them: "Send this link to Manav to review and merge." Then give Moorty a 4-6 bullet plain-English summary of what this PR does and why, so they can explain it in the Q&A.

> **Do this after the demo (Fri 9 Oct).** Wait until Manav's **PR #6** (README + write-up skeleton) is merged.

## Step 5 - PR #7: Results in README and write-up

_Fills in the results half of the README and write-up._

Start a fresh branch from the latest `main`.

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
git checkout main                 # go back to the shared main branch
git pull                          # download everything merged so far
git checkout -b docs/results-and-report   # create your own working branch for this PR
```

### B4.1 - commit: `docs: add results section to README`

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
Cross-validated MSE of normalised metabolic cost using all 29 features
(`results/tables/model_comparison.csv`; noise floor of the measurement is about 0.0025).

| Subject | Model | Random 5-fold | Leave-one-day-out | Paper (5-fold) |
|---|---|---|---|---|
| S1 | Constant | 0.0666 | 0.0917 | 0.0657 |
| S1 | LASSO | 0.0091 | 0.0205 | 0.0089 |
| S1 | Ridge | 0.0086 | **0.0143** | - |
| S1 | Neural net (nested-tuned) | 0.0101 | 0.0223 | 0.0089 |
| S2 | Constant | 0.0362 | 0.0478 | 0.0465 |
| S2 | LASSO | 0.0135 | **0.0309** | 0.0176 |
| S2 | Ridge | 0.0146 | 0.0358 | - |
| S2 | Neural net (nested-tuned) | 0.0150 | 0.0315 | 0.0301 |

Key findings:

1. With the paper's random k-fold protocol we reproduce its accuracy (S1 LASSO 0.0091 vs 0.0089).
2. Predicting an **unseen day** roughly doubles the error - day-to-day adaptation of the subject is
   the main obstacle, and random k-fold over-states real-world accuracy.
3. With all features, regularised linear models are as accurate as the shallow neural network;
   the network only helps on the small step-only / EMG-only feature sets.
4. Step + EMG beats either group alone; control parameters help noticeably under leave-one-day-out.
5. Nested forward selection and PCA do not beat the full regularised models.

![Model comparison S1](results/figures/model_comparison_S1.png)
![Leave-one-day-out predictions S1](results/figures/lodo_predictions_S1.png)
<!-- RESULTS:END -->

## Reference

E. Krimsky and E. Ng, "Predicting Metabolic Cost During Human-in-the-Loop Optimization", CS229
project report, Stanford University, 2018. Original MATLAB code: https://github.com/ngeley/cs229project
````

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add README.md
# save this step under your name
git commit -m "docs: add results section to README"
```

### B4.2 - commit: `docs(report): implementation, results and conclusions`

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
The code is a Python package (\texttt{src/}) built on scikit-learn, with unit tests, a single
script that regenerates every table and figure, and four Jupyter notebooks (EDA, baselines,
results, live demo). The neural network mirrors the paper's: one hidden layer of tanh units and a
linear output, trained with the quasi-Newton L-BFGS solver; an L2 weight penalty replaces
MATLAB's Bayesian regularisation, and both inputs and target are standardised inside the pipeline.
The hidden size $\in\{1,2,3,4,6,8\}$ and penalty $\alpha\in\{0.1,1,10\}$ are chosen by an
\emph{inner} 4-fold CV (nested CV), so test folds never influence model selection. We also swept
1--10 neurons (paper Fig.~1), and ran forward stepwise selection and PCA nested inside the CV loop.

\section{Results}
Table~\ref{tab:res} reports cross-validated MSE with all features.
\begin{table}[h]
\centering\small
\begin{tabular}{llccc}
\toprule
Subj. & Model & k-fold & LODO & Paper (k-fold)\\
\midrule
S1 & Constant & 0.0666 & 0.0917 & 0.0657\\
   & LASSO    & 0.0091 & 0.0205 & 0.0089\\
   & Ridge    & 0.0086 & 0.0143 & --\\
   & NN       & 0.0101 & 0.0223 & 0.0089\\
\midrule
S2 & Constant & 0.0362 & 0.0478 & 0.0465\\
   & LASSO    & 0.0135 & 0.0309 & 0.0176\\
   & Ridge    & 0.0146 & 0.0358 & --\\
   & NN       & 0.0150 & 0.0315 & 0.0301\\
\bottomrule
\end{tabular}
\caption{MSE of normalised metabolic cost, all 29 features. Noise floor $\approx0.0025$.}
\label{tab:res}
\end{table}
\vspace{-0.3cm}
\begin{figure}[h]
\centering
\includegraphics[width=\columnwidth]{figures/model_comparison_S1.png}
\vspace{-0.6cm}
\caption{S1: MSE per model and feature set under both protocols.}
\label{fig:cmp}
\end{figure}
\vspace{-0.3cm}

\textbf{Reproduction.} Under random k-fold our errors match the paper (S1 LASSO 0.0091 vs 0.0089,
an RMSE of about 10\% of normal-walking cost), and our nested-tuned NN improves on the paper's S2
network (0.0150 vs 0.0301).
\textbf{Unseen days.} LODO roughly doubles the error (S1 LASSO 0.0091$\to$0.0205; S2
0.0135$\to$0.0309); for S2 the best model explains only about 14\% of an unseen day's variance.
\textbf{Linear vs neural.} With all features, regularised linear models match the NN (best LODO:
Ridge 0.0143 for S1); the NN helps only on the small step-only and EMG-only sets, and larger
networks degrade under LODO.
\textbf{Features.} Step+EMG beats either group alone (Fig.~\ref{fig:cmp}). Control parameters help
markedly under LODO (S2 LASSO 0.0448$\to$0.0309), likely because the optimiser's settings drift
across days and encode adaptation. Nested forward selection (LODO MSE $\approx0.046$) and PCA did
not beat the full regularised models, and the selected features changed between held-out days.

\section{Conclusions}
We reproduced the reference accuracy in Python with a leak-free pipeline, and showed that the
headline numbers depend on the evaluation protocol: predicting a new walking day is roughly twice
as hard, because the subject's adaptation shifts metabolic cost from day to day. On 180 samples,
simple regularised linear models are as accurate as a shallow neural network. Future work: more
subjects, day-level normalisation or adaptation features, and evaluating on fully unseen subjects.
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

Copy the figure into the report folder so Overleaf can find it:

```bash
mkdir -p report/figures
cp results/figures/model_comparison_S1.png report/figures/
```

Then the commit:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# stage exactly the files for this step (nothing else)
git add report/writeup.tex report/figures/model_comparison_S1.png
# save this step under your name
git commit -m "docs(report): implementation, results and conclusions"
```

### B4 - open the pull request

Claude runs the full test suite one last time: `python -m pytest -q` (it must say 'passed' with no failures).

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# upload your branch to GitHub
git push -u origin docs/results-and-report
# open the pull request (prints the PR link)
gh pr create --base main --head docs/results-and-report --title "Results in README and write-up" --body "Fills in the results half of the README and write-up. Part of Moorty's work (repository owner, neural network, feature selection, experiments, live demo)."
```

> **CHECKPOINT** - Claude: ask Moorty for the PR link printed above and tell them: "Send this link to Manav to review and merge." Then give Moorty a 4-6 bullet plain-English summary of what this PR does and why, so they can explain it in the Q&A.

## Reviewing Manav's pull requests

You review and merge Manav's PRs (this is visible evidence of collaboration in the repo):

- PR #1 `feature/data-and-cv` - Data loading and cross-validation protocols
- PR #2 `feature/baselines-and-eda` - Baseline models and exploratory data analysis
- PR #6 `docs/readme-and-report` - README and write-up skeleton

When Manav sends a PR link (replace `N` with the PR number):

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
git checkout main && git pull     # make sure your main is up to date
gh pr checkout N                    # download the PR's branch to test it locally
```

Claude then runs `python -m pytest -q` and reads through the changed files with Moorty,
explaining what they do. If the tests pass:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# approve the PR (shows up on GitHub as your review)
gh pr review N --approve --body "Reviewed: tests pass locally, LGTM."
# merge it with a merge commit, which keeps every individual commit and author
gh pr merge N --merge --delete-branch
# go back to main and download the merged result
git checkout main && git pull
```

Always use **`--merge`** (a merge commit), never squash or rebase - squashing would collapse
Manav's commits and hide individual contributions. If the tests fail, do not merge:
Moorty posts the failure with `gh pr review N --request-changes --body "<what failed>"`.
If GitHub reports a merge conflict, Manav runs `git pull origin main` on their branch,
resolves it, and pushes again.

## Slides you own (in the shared Google Slides deck)

Keep slides visual: one figure + 3-4 bullets each. Figures are in `results/figures/`.

7. **Neural network** - 1 hidden layer, tanh, linear output, L2 penalty (replaces Bayesian regularisation), L-BFGS; nested CV picks hidden size {1,2,3,4,6,8} and alpha {0.1,1,10}; figure `neuron_sweep_S1.png`.
8. **Reproduction vs paper** - results table (k-fold column vs paper); S1 LASSO 0.0091 vs 0.0089; our S2 NN 0.0150 vs paper 0.0301.
9. **Unseen days are harder** - figure `model_comparison_S1.png` + `lodo_predictions_S1.png`; error roughly doubles under LODO; S2 explains only ~14% of an unseen day's variance.
10. **Feature selection & PCA** - `selection_frequency.png`, `pca_sweep.png`; nested selection doesn't beat full regularised models; selected features vary by day (paper saw the same instability).
11. **Conclusions & future work** - protocol matters more than model; linear ~ NN on 180 samples; future: more subjects, day-level normalisation, subject-independent models.
12. **Live demo** - switch to `notebooks/04_live_demo.ipynb`.
13. **Contributions** - one line each (both of you agree on wording); link to the repo's commit graph / PR list.

## Live demo script (Thu 8 Oct)

1. Open `notebooks/04_live_demo.ipynb` in Jupyter/VS Code from the repo root's `.venv` (`jupyter notebook notebooks/04_live_demo.ipynb`).
2. **Kernel -> Restart & Run All** *before* the panel arrives (takes ~15-30 s), so plots are on screen.
3. Walk through: Step 1 data plot (Manav explains), Step 2 live training (Moorty explains), Step 3 results table (Moorty).
4. If asked "what about S2 / EMG only?": change `SUBJECT` / `FEATURE_SET` in the 2nd cell and re-run the cells below it (~15 s).
5. Backup: `python -m pytest -q` in a terminal shows the test-suite passing; `python -m scripts.run_experiments --quick` runs everything end-to-end in ~2 min.

## Q&A preparation (Moorty) - know these cold

- **Network architecture?** Input (standardised) -> 1 hidden layer of tanh units -> 1 linear output. Same as the paper's `feedforwardnet` with `trainbr`. Target also standardised (`TransformedTargetRegressor`), predictions converted back.
- **Bayesian regularisation vs our alpha?** MATLAB's trainbr adds a weight-decay penalty whose strength it estimates automatically; we use an explicit L2 penalty `alpha` and choose it by inner CV. Same purpose: stop a tiny, noisy dataset being over-fitted.
- **Why L-BFGS, not Adam/SGD?** Quasi-Newton full-batch optimiser - converges reliably and fast on ~150 samples; mini-batch SGD is for large datasets.
- **What is nested CV and why?** Outer loop estimates test error; inner loop (4-fold `GridSearchCV`) on the outer-training rows picks hidden size and alpha. Picking the best configuration by looking at the outer test error (as the paper's neuron plot effectively does) is optimistic.
- **Why is NN not better than LASSO?** 180 noisy samples; the relationship is close to linear once features are standardised; the NN's extra flexibility mostly fits noise. It helps only on the small feature sets (step-only/EMG-only). Under LODO, bigger networks get worse.
- **Why does LODO double the error?** The subject adapts over days, shifting metabolic cost (day means 1.15 -> 0.66 for S1). A model trained on other days must extrapolate to a new adaptation level.
- **Why do control parameters help under LODO?** CMA-ES moves the settings across days as it converges, so they partly encode "which stage of adaptation" the subject is in.
- **Forward selection / PCA results?** Done *inside* each fold. Selection is greedy and unstable - selected features change per held-out day (we report selection frequency); PCA picks directions of max variance, not max correlation with the target, so it can discard predictive signal. Neither beat the full regularised models.
- **How do I reproduce everything?** `pip install -r requirements.txt`, `pytest`, `python -m scripts.run_experiments` (~5 min) -> `results/tables/*.csv` and `results/figures/*.png`; notebooks 03/04 read those files.
- **Code walkthrough:** `make_nn(h, alpha)`, `make_tuned_nn()` (GridSearchCV), `neuron_sweep()`, `forward_selection()` returns (mse, frequency), `pca_sweep()`, `run_experiments.main()` loops subject x feature set x protocol x model.

## After PR #7: write-up PDF, faculty access, submission

1. **Overleaf:** upload `report/writeup.tex` and `report/figures/model_comparison_S1.png` (keep the
   `figures/` folder) to the shared Overleaf project and compile. It must fit on **2 pages**; if it
   spills over, shorten the Results discussion slightly (do not shrink the font below 10pt).
   The guidelines say "one-page write-up" in a heading but "two-page summary" in the text - ask
   the faculty if unsure; 2 pages is the safe reading.
2. Download the PDF. Claude copies it to `report/writeup.pdf` on a new branch, which Moorty creates and commits:

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# start a small branch from the latest main
git checkout main && git pull && git checkout -b docs/writeup-pdf
# (Claude copies the downloaded PDF to report/writeup.pdf at this point)
# stage and commit only the PDF
git add report/writeup.pdf
git commit -m "docs(report): compiled two-page write-up"
# upload and open the pull request
git push -u origin docs/writeup-pdf
gh pr create --base main --head docs/writeup-pdf --title "Compiled write-up PDF" --body "Final 2-page PDF compiled on Overleaf."
```

   Manav merges it. Moorty merges Manav's slides PR (`docs/slides`).
3. **Share with faculty / TAs** (ask them for their GitHub usernames):

**Moorty types these commands** (Claude: show them, explain them, but never run git or gh commands yourself; wait until Moorty says they worked):

```bash
# give a faculty member / TA access to the private repo (repeat per person)
gh api -X PUT repos/<your-github-username>/hilo-metabolic-cost/collaborators/<faculty-username>
```

4. Final sanity check: Moorty runs `git checkout main && git pull`, Claude runs `python -m pytest -q`.
   Open the README on GitHub and check the images render, and check the
   **Insights -> Contributors** page shows both of you.
5. Submit the write-up PDF (and anything else the course portal asks for) before **Sat 10 Oct, 11:59 PM**.

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
