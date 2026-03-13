"""
Experiment configuration file. This file is imported in all notebooks to ensure that the same data and model configuration is used across all experiments.
"""

from pathlib import Path

from src.models.degradation.gamma import GammaDegradation as DegModel  # noqa: F401
from src.models.degradation.gamma import GammaDegradationNLL as Loss  # noqa: F401

SEED = 42
N_REP = 3


# DATA
DATA_NAME = "DS06"

DATA_DIR = Path("experiments") / DATA_NAME


# OPERATION CONDITION NORMALIZATION
EOL_MARG = 0.01
NOM_MARG = 0.01
MIN_CORR = 0.6
MIN_RANGE = 0.6
OPCOND_DIR = DATA_DIR / f"opcond_q{EOL_MARG}-{1-NOM_MARG}_corr{MIN_CORR}_range{MIN_RANGE}"


# ESTIMATION DATA
# filter
FILTER_DATA = False  # True  #
FILT_NOISE_MEM = 0.01
FILT_SMOOTH_STRENGTH = 20.0

FAILED_THRESHOLD = 0.2

FILTERED_SUFFIX = f"_filtered_n{FILT_NOISE_MEM}_s{FILT_SMOOTH_STRENGTH}" if FILTER_DATA else ""
THRESHOLD_SUFFIX = f"_thr{FAILED_THRESHOLD}" if FAILED_THRESHOLD < 1 else ""
ESTIMATION_NAME = "estimation" + THRESHOLD_SUFFIX + FILTERED_SUFFIX
ESTIMATION_DIR = OPCOND_DIR / ESTIMATION_NAME


## DEGRADATION MODEL
DEGR_MODEL_DIR = ESTIMATION_DIR / DegModel.name()


# NETWORK CONTROLLER TRAINING
# Network model parameters
HIDDEN_DIMS = [256, 256, 128, 64, 32]  # [64, 64]  #     [16]  #
LEAKY_SLOPE = 0.05
# loss parameters
WEIGHT_DECAY = 0  # 1e-4  #\
GRADIENT_CLIP = None
LOSS_TAIL_SIZE = 5
CURRENT_OBS = True
# early stopping score
AVERSION = 0.5  # 0  #
LOSS_WINDOW = 10

# PF parameters
N_PARTICLES = 2000
PREDICTION_START_IDX = 0
PRED_STAT = "mean"  # "mean" or "mode"

WEIGHT_DECAY_SUFFIX = f"_wdecay{WEIGHT_DECAY}" if WEIGHT_DECAY > 0 else ""
GRADIENT_CLIP_SUFFIX = f"_gradclip{GRADIENT_CLIP}" if GRADIENT_CLIP is not None else ""
AVERSION_SUFFIX = f"_aversion{AVERSION}" if AVERSION > 0 else ""
LOSS_WINDOW_SUFFIX = f"_losswin{LOSS_WINDOW}" if LOSS_WINDOW > 1 else ""
PREDICTION_START_IDX_SUFFIX = f"_pred{PREDICTION_START_IDX}" if PREDICTION_START_IDX > 0 else ""
PRED_STAT_SUFFIX = f"_{PRED_STAT}" if PRED_STAT != "mean" else ""
LOSS_TAIL_SIZE_SUFFIX = f"_losstail{LOSS_TAIL_SIZE}" if LOSS_TAIL_SIZE != 1 else ""
CURRENT_OBS_SUFFIX = "_currentobs" if CURRENT_OBS else ""

PFNET_NAME = f"net{'x'.join(map(str, HIDDEN_DIMS))}" f"_leaky{LEAKY_SLOPE}" f"_npar{N_PARTICLES}_cv"
PFNET_NAME += (
    WEIGHT_DECAY_SUFFIX
    + GRADIENT_CLIP_SUFFIX
    + AVERSION_SUFFIX
    + LOSS_WINDOW_SUFFIX
    + PREDICTION_START_IDX_SUFFIX
    + PRED_STAT_SUFFIX
    + LOSS_TAIL_SIZE_SUFFIX
    + CURRENT_OBS_SUFFIX
)

PFNET_DIR = DEGR_MODEL_DIR / PFNET_NAME


# RESULTS
UNCERTAINTY_LEVEL = 0.95

PRED_DIR = PFNET_DIR / f"pred_ulevel{UNCERTAINTY_LEVEL}"
