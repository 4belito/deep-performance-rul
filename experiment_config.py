"""
Experiment configuration file. This file is imported in all notebooks to ensure that the same data and model configuration is used across all experiments.
"""

from src.models.degradation.gamma_flexible_nvcte import GammaDegradation as DegModel  # noqa: F401
from src.models.degradation.gamma_flexible_nvcte import GammaDegradationNLL as Loss  # noqa: F401

# DATA
DATA_NAME = "DS02"

# ESTIMATION DATA
FAILED_THRESHOLD = 0.1
Q_LOW = 0.01
Q_HIGH = 0.99

# FILTER
FILTER_DATA = True
FILT_NOISE_MEM = 0.01
FILT_SMOOTH_STRENGTH = 20.0


# Network Training
USE_LASTOBS_LOSS = True
SEED = 42
EMA_BETA = 0.9

# Network Model
PREDICTION_START_IDX = 0
HIDDEN_DIMS = [256, 256, 128, 64, 32]  # [16]  #
LEAKY_SLOPE = 0.05

# EXPERIMENT
N_REP = 10
UNCERTAINTY_LEVEL = 0.95
PRED_STAT = "mean"  # "mean" or "mode"

FILTERED_SUFFIX = f"_filtered_n{FILT_NOISE_MEM}_s{FILT_SMOOTH_STRENGTH}" if FILTER_DATA else ""
PFNET_NAME = (
    "net"
    + "x".join([str(h) for h in HIDDEN_DIMS])
    + f"leaky{LEAKY_SLOPE}_emaloss{EMA_BETA}_init{PREDICTION_START_IDX}"  # {FILTERED_SUFFIX}
)

ESTIMATION_NAME = f"estimation_thr{FAILED_THRESHOLD}_q{Q_LOW}-{Q_HIGH}" + FILTERED_SUFFIX
