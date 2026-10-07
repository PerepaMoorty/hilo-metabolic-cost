# START HERE - Team 13 mini-project plan (for Moorty)

## The files (all in `References/generated/`)

| File | For | What it does |
|---|---|---|
| `GUIDE_PersonB_Moorty.md` | **you** | Give it to Claude Code. Claude writes your code and runs the tests; **you type every git/gh command** (Claude shows each one with comments). |
| `GUIDE_PersonA_Manav.md` | **Manav** | Send it to him. Same deal: his Claude writes his code, he types his own git/gh commands. |
| `WORKFLOW_EXPLAINED.md` | **both** | Plain-language explanation of the workflow. Send it to Manav too. |
| `References/` | - | The original material. It is **not** copied into the repo, so the repo is our own Python reimplementation, not a copy of the MATLAB code. |

Both guides contain the complete code, commit by commit. I built and tested the whole project
before splitting it. I replayed the full git history (scaffold -> PR #1 ... PR #7) in a throwaway
test folder, and the tests passed after every merge (14 tests at the end). All four notebooks run
without errors, and the full experiment run takes about 5 minutes.

## What to do right now (Wed 7 Oct)

1. The repo folder `MiniProject/hilo-metabolic-cost/` already contains the files for your **first
   commit** plus a ready Python environment (`.venv`). No git commands have been run in it.
2. **You** run the first-commit commands (init, identity, `gh repo create`, first commit). They are
   in the guide's setup section and Step 1, with a comment on every line.
3. Then open Claude Code in `MiniProject` and say:
   *"Read References/generated/GUIDE_PersonB_Moorty.md and follow it step by step for me. The
   project folder and the first commit's files already exist."*
   Have ready: **Manav's GitHub username** (for the invite).
4. When it reaches the checkpoint, **send Manav**:
   - `GUIDE_PersonA_Manav.md` and `WORKFLOW_EXPLAINED.md`
   - the repo URL (`https://github.com/<you>/hilo-metabolic-cost`)
   - a reminder to accept the invite (`.../hilo-metabolic-cost/invitations`)
   - message to paste: *"Install Git, Python 3.11 and GitHub CLI (`gh`), run `gh auth login`, accept
     the repo invite, then open Claude Code in an empty folder and tell it: 'Read GUIDE_PersonA_Manav.md
     and follow it step by step for me.' Claude writes the code; you type the git commands it shows you."*
5. Create a shared **Google Slides** deck and an **Overleaf** project, and give Manav edit access to both.

## Order of work (each PR is merged by the other person)

```
You:   scaffold ──────────────┐            PR#3 nn ─ PR#4 selection ─┐        PR#5 experiments+demo
Manav:                         └ PR#1 data+CV ┬ PR#2 baselines+EDA ───┴─(both)─┘    PR#6 README+write-up
                                (you merge)    └─ you start PR#3 once PR#1 is merged
Thu 8:  Manav does a fresh-clone check -> slides -> live demo (notebooks/04_live_demo.ipynb)
Fri 9-Sat 10:  you: PR#7 results + write-up -> Overleaf PDF -> commit PDFs -> add faculty -> submit
```

Final contribution count: about 14 commits for you and 11 for Manav, plus each of you reviews and
merges the other's PRs. All PRs use merge commits, so every individual commit stays visible.

## What the project is (in case the panel asks you the big picture)

- **Same as the reference:** LASSO and a one-hidden-layer tanh neural network that predict
  normalised metabolic cost from step, EMG and control features, compared across 4 feature sets.
  Under the paper's random 5-fold CV we reproduce its numbers: S1 LASSO MSE **0.0091** (paper 0.0089).
- **New in ours:**
  - **Leave-one-day-out CV** (train on 4 days, test on an unseen day). This was the paper's
    suggested future work. The error **roughly doubles** (S1 0.0091 -> 0.0205), because the
    subject keeps adapting from day to day.
  - **Leak-free pipelines.** The reference scaled the data before splitting it. Under
    leave-one-day-out that made S2 look 21% better than it really is.
  - **Nested tuning** of the network, and **nested** feature selection and PCA.
  - A Ridge baseline, tests, an experiment runner and a demo notebook.
- **Conclusion:** the evaluation protocol matters more than the model. With 180 samples,
  regularised linear models are about as accurate as the neural network.

## Things to double-check with the faculty

- The guidelines heading says "One-Page Write-up" but the text says "two-page summary". The guides
  aim for **2 pages**.
- The GitHub usernames of the faculty/TAs to add to the private repo (step in your guide's last section).
