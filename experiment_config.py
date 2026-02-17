"""
Experiment configuration file. This file is imported in all notebooks to ensure that the same data and model configuration is used across all experiments.
"""

from src.models.degradation.normal import NormalDegradation as DegModel  # noqa: F401
from src.models.degradation.normal import NormalDegradationNLL as Loss  # noqa: F401

DATA_NAME = "DS03"
USE_FILTERED_DATA = True
SEED = 42


PREDICTION_START_IDX = 0
HIDDEN_DIMS = [256, 256, 128, 64, 32]
LEAKY_SLOPE = 0.05


N_REP = 10
UNCERTAINTY_LEVEL = 0.95


PFNET_NAME = "net" + "x".join([str(h) for h in HIDDEN_DIMS]) + f"leaky{LEAKY_SLOPE}"
FILTERED_SUFFIX = "_filtered" if USE_FILTERED_DATA else ""
