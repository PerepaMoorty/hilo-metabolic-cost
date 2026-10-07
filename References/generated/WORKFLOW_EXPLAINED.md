# How we'll build the project together (the simple version)

Read this before opening your guide. Your guide has every detail; this page only explains how the
pieces fit together.

---

## 1. The idea in one paragraph

There is **one shared project on GitHub** (Moorty creates it). Each of us works **on our own laptop**,
with **our own Claude Code** and **our own guide**. We each add our own part in small saved steps
called **commits**. When a part is finished, we send it to the other person to check. They approve it,
and it joins the main project. At the end GitHub shows exactly who did what, which is part of the grade.

---

## 2. Six words you need

| Word | What it means |
|---|---|
| **Repo** | The project folder that lives on GitHub. Each of us has a copy on our laptop. |
| **Commit** | Saving one small step, with a short message like "add data loader". It is stamped with **your name**. |
| **Branch** | Your own working lane, so your half-finished work doesn't disturb the other person. |
| **Push** | Uploading your commits from your laptop to GitHub. |
| **Pull request (PR)** | "Hey, my part is ready, please check it." It comes as a link you send to the other person. |
| **Merge** | The other person approves the PR, and your work joins the main project (`main`). |

---

## 3. Why the commits will show *your* name

Git signs every commit with the name and email set on the laptop where the commit is made. In
setup, each guide sets this once:

```
git config user.name  "Your Name"
git config user.email "the-email-of-YOUR-GitHub-account"
```

So:
- Commits made on **Moorty's laptop** are Moorty's.
- Commits made on **Manav's laptop** are Manav's.
- The email **must be the one on your GitHub account**, or GitHub can't link the commits to your
  profile.

**You type every git and gh command yourself.** Claude writes the code and runs the tests, then
shows you the exact commands, with a comment on each line saying what it does, and waits until you
say they worked. Claude never runs git for you.

---

## 4. The plan, round by round

Each round finishes when the PR links have been sent **and merged**. Tick these off together.

### Round 0 - Setup (Wed, today)
| Moorty | Manav |
|---|---|
| Creates the private GitHub repo and makes the first 2 commits (empty project skeleton). | Installs Git, Python 3.11 and GitHub CLI, then runs `gh auth login`. |
| Invites Manav to the repo. Creates a shared Google Slides deck and an Overleaf project. | Accepts the invite (email/GitHub notification). Opens Claude Code with `GUIDE_PersonA_Manav.md`. |

**Message:** Moorty sends Manav the repo link. Manav replies "invite accepted".

### Round 1 - Data (Manav)
| Moorty | Manav |
|---|---|
| Waits, then reviews and merges PR #1. | **PR #1**: adds the dataset, the code that loads it and the testing method (4 commits). Sends the PR link to Moorty. |

Everything else needs this part, so it goes first and should be quick.

### Round 2 - Both work at the same time
| Moorty | Manav |
|---|---|
| **PR #3**: neural network (2 commits). **PR #4**: feature selection + PCA (2 commits). Sends both links to Manav. | **PR #2**: simple baseline models + data exploration notebooks (4 commits). Sends the link to Moorty. |
| Merges Manav's PR #2. | Merges Moorty's PRs #3 and #4. |

### Round 3 - Results and documentation
| Moorty | Manav |
|---|---|
| **PR #5**: runs all experiments, saves results and makes the **live demo notebook** (5 commits). Needs PRs #2, #3 and #4 merged first. | **PR #6**: README (how to install and run) + first half of the write-up (2 commits). |
| Merges Manav's PR #6. | Merges Moorty's PR #5. |

### Round 4 - Get ready for the demo (Wed evening)
- Both fill in the shared Google Slides. Manav does slides 1-6 (problem, data, method); Moorty does 7-13 (models, results, demo).
- Practise the demo once: open `notebooks/04_live_demo.ipynb`, then Restart & Run All.
- Each of you reads the **"Q&A preparation"** section of your own guide.

### Round 5 - Demo day (Thu 8 Oct)
- **Morning, Manav:** downloads a fresh copy of the repo and checks that everything runs (the "Final checks" section of his guide). If anything fails, tell Moorty immediately.
- **Review:** slides, then live demo, then questions.

### Round 6 - Finish and submit (Fri 9 - Sat 10 Oct)
| Moorty | Manav |
|---|---|
| **PR #7**: results in the README + second half of the write-up. | Merges PR #7. |
| Compiles the write-up on Overleaf (2 pages) and commits the PDF in a small PR. | Downloads the slides as PDF and commits them in a small PR. |
| Adds the faculty/TAs to the GitHub repo. Submits before **Sat 11:59 PM**. | Proof-reads the write-up. |

---

## 5. Five golden rules

1. **Never work directly on `main`.** Use one branch per PR; your guide names them.
2. **Only touch your own files.** If the other person's file looks broken, tell them. Don't fix it yourself.
3. **Tests must pass before you commit or push.** Claude runs `python -m pytest -q` each time, before showing you the git commands.
4. **When merging, always use "Create a merge commit".** Not "Squash" or "Rebase", which would hide individual commits.
5. **Talk at every checkpoint.** When your Claude says CHECKPOINT, send the link or message it tells you to.

---

## 6. How to know it worked

On GitHub, go to the repo, then **Insights -> Contributors**. You should both appear, with about
14 commits for Moorty and 11 for Manav. The **Pull requests -> Closed** tab should show 7+ PRs,
each opened by one of us and merged by the other.
