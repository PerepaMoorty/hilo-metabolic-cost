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
