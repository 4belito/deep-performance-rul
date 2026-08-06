"""
Experiment configuration file. This file is imported in all notebooks to ensure that the
same data and model configuration is used across all experiments.
"""

import json
import os
from pathlib import Path
from typing import Any

from src.models.degradation.gamma import GammaDegradation as DegModel  # noqa: F401
from src.models.degradation.gamma import GammaDegradationNLL as Loss  # noqa: F401

SEED = 42

# ---- Fixed configuration (identical across every dataset) ----
EOL_MARG = 0.001
NOM_MARG = 0.001
MIN_CORR = 0.6
MIN_RANGE = 0.6
FAILED_THRESHOLD = 0.2
UNCERTAINTY_LEVEL = 0.95

_EXPERIMENTS_ROOT = (
    Path("/lustre") / Path.home().name / "deep-performance-rul" / "experiments"
)


def data_name_from_env() -> str:
    """Dataset name from the DATA_NAME env var (explicit per-run choice)."""
    name = os.environ.get("DATA_NAME")
    if not name:
        raise RuntimeError(
            "DATA_NAME environment variable is not set. Set it before launching, "
            "e.g. `export DATA_NAME=DS01`, so the dataset is chosen explicitly."
        )
    return name


def dataset_paths(
    data_name: str,
    fields: list[str] | None = None,
) -> tuple[Path, ...]:
    """Return dataset-dependent paths for a given dataset name.

    ``fields`` optionally selects a subset of ``DatasetPaths`` field names:
    returns the full ``DatasetPaths`` when None, a single ``Path`` for one field
    (no unitary tuple), or a tuple of ``Path`` for several.
    """
    data_dir = _EXPERIMENTS_ROOT / data_name
    opcond_dir = data_dir / (
        f"opcond_q{EOL_MARG}-{1 - NOM_MARG}_corr{MIN_CORR}_range{MIN_RANGE}"
    )
    threshold_suffix = f"_thr{FAILED_THRESHOLD}" if FAILED_THRESHOLD < 1 else ""
    estimation_dir = opcond_dir / ("estimation" + threshold_suffix)
    degr_model_dir = estimation_dir / DegModel.name()
    paths = {
        "data": data_dir,
        "opcond": opcond_dir,
        "estimation": estimation_dir,
        "degr_model": degr_model_dir,
    }
    if fields is None:
        return tuple(paths.values())
    return tuple(paths[name] for name in fields if name in paths)


def dataset_path(data_name: str, field: str) -> Path:
    """Return a single dataset-dependent path for a given dataset name and field."""
    return dataset_paths(data_name, fields=[field])[0]


# NETWORK CONTROLLER PATHS
# ARGS_ID is chosen per notebook (never globally) so training and test never drift:
# each notebook picks PFNET_ARGS[ARGS_ID] and derives its paths via pfnet_paths().
def pfnet_paths(
    args_id: int,
    data_name: str,
    uncertainty_level: float = UNCERTAINTY_LEVEL,
) -> tuple[Path, Path]:
    """Return (pfnet_dir, pred_dir) for a dataset + hyperparameter-set id."""
    pfnet_dir = dataset_path(data_name, "degr_model") / f"net_arg{args_id}"
    pred_dir = pfnet_dir / f"pred_ulevel{uncertainty_level}"
    return pfnet_dir, pred_dir


def load_baseline_gains(
    args_id: int,
    data_name: str,
    perform_name: str,
    uncertainty_level: float = UNCERTAINTY_LEVEL,
) -> dict[str, Any]:
    """Per-metric constant PF gains tuned by Optuna, or PF defaults if absent.

    Reads ``net_arg{args_id}/{perform_name}/optuna_best_gains.json`` and returns a
    ``{"NOISE", "PRIOR", "LIK"}`` dict (None entries fall back to ParticleFilter
    defaults). Used by the no-network baseline (``use_net=False``).
    """
    pfnet_dir, _ = pfnet_paths(args_id, data_name, uncertainty_level)
    path = pfnet_dir / perform_name / "optuna_best_gains.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"NOISE": None, "PRIOR": None, "LIK": None}


def count_baseline_trials(
    args_id: int,
    data_name: str,
    perform_name: str,
    uncertainty_level: float = UNCERTAINTY_LEVEL,
) -> int:
    """Completed Optuna trials for a per-metric baseline study (0 if none)."""
    import optuna  # lazy: keep experiment_config light for non-optuna notebooks

    pfnet_dir, _ = pfnet_paths(args_id, data_name, uncertainty_level)
    db = pfnet_dir / perform_name / "optuna_pf_gains.db"
    if not db.exists():
        return 0
    study = optuna.load_study(
        study_name=f"pf_gains_{perform_name}_{data_name}",
        storage=f"sqlite:///{db.as_posix()}",
    )
    return len(
        study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    )


def build_pfnet_hparams(
    *,
    deg_model: str,
    hidden_dims: list[int],
    leaky_slope: float,
    dropout: float,
    lr: float,
    n_particles: int,
    max_life: int,
    loss_tail_size: int,
    aversion: float,
    loss_window: int,
    min_epochs: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    stop_mode: str,
    eval_repetitions: int = 10,
    eval_seed_stride: int = 100_000,
) -> dict[str, Any]:
    """Return only model/training hyperparameters (no run-specific context)."""
    return {
        "deg_model": deg_model,
        "hidden_dims": hidden_dims,
        "leaky_slope": leaky_slope,
        "dropout": dropout,
        "lr": lr,
        "n_particles": n_particles,
        "max_life": max_life,
        "loss_tail_size": loss_tail_size,
        "aversion": aversion,
        "loss_window": loss_window,
        "min_epochs": min_epochs,
        "max_epochs": max_epochs,
        "patience": patience,
        "min_delta": min_delta,
        "stop_mode": stop_mode,
        "eval_repetitions": eval_repetitions,
        "eval_seed_stride": eval_seed_stride,
    }


def save_pfnet_hparams(hparams: dict[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=2)

    return output_path
