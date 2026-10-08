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
