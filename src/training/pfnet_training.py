import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from src.models.particle_filter.core import ParticleFilter

if TYPE_CHECKING:
    from src.models.networks.pf_mlp import ParticleFilterMLP


def train_pf_controller_cv(
    *,
    net: "ParticleFilterMLP",
    optimizer: torch.optim.Optimizer,
    units: list[int],
    dev_data: dict[int, torch.Tensor],
    dev_degmodels: dict[int, Any],
    n_particles: int,
    max_life: int,
    loss_tail_steps: int,
    n_epochs: int,
    checkpoint_best_path: Path,
    checkpoint_last_path: Path,
    start_epoch: int = 0,
    best_score: float = float("inf"),
    scores: list[float] | None = None,
    train_epoch_losses: list[float] | None = None,
    eval_epoch_losses: list[float] | None = None,
    seed: int = 0,
    eval_repetitions: int = 10,
    eval_seed_stride: int = 100_000,
    eval_initial_interval_epochs: int = 10,
    eval_final_interval_epochs: int = 1,
    match_train_eval_prior: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train PF controller with leave-one-unit-out CV and checkpointing.

    Evaluation is stochastic but reproducible:
    - each eval unit uses a fixed list of per-repetition rollout seeds
    - seeds are ``seed + unit * eval_seed_stride + repetition`` (same every epoch)
    - eval RNG is isolated from training RNG via ``torch.random.fork_rng``

    Notes:
    - best-checkpoint selection is based only on raw evaluation score
    - no early stopping is applied (full fixed-budget training)
    - eval interval changes linearly from the initial to the final interval
    - ``match_train_eval_prior``: when True the training prior keeps every unit
      except the fitted ``train_unit`` (n-1 base models), matching the evaluation
      prior; when False (default) the held-out ``eval_unit`` is also removed (n-2)
    """
    checkpoint_best_path = Path(checkpoint_best_path)
    checkpoint_last_path = Path(checkpoint_last_path)
    checkpoint_best_path.parent.mkdir(parents=True, exist_ok=True)

    if eval_repetitions < 1:
        raise ValueError("eval_repetitions must be >= 1")
    if eval_initial_interval_epochs < 1:
        raise ValueError("eval_initial_interval_epochs must be >= 1")
    if eval_final_interval_epochs < 1:
        raise ValueError("eval_final_interval_epochs must be >= 1")

    scores = [] if scores is None else list(scores)
    train_epoch_losses = [] if train_epoch_losses is None else list(train_epoch_losses)
    eval_epoch_losses = [] if eval_epoch_losses is None else list(eval_epoch_losses)

    # Use a dedicated RNG for training shuffles so evaluation randomness can stay fixed.
    train_rng = np.random.default_rng(int(seed))

    # Fixed per-repetition eval seeds, identical at every evaluation epoch.
    eval_seed_table: dict[int, list[int]] = {
        int(eval_unit): [
            int(seed) + int(eval_unit) * int(eval_seed_stride) + rep
            for rep in range(eval_repetitions)
        ]
        for eval_unit in sorted(units)
    }

    best_checkpoint = None
    epoch = start_epoch
    total_epochs = max(1, n_epochs - start_epoch)
    next_eval_epoch = start_epoch

    def mean_tail_nll(
        pf: ParticleFilter,
        t_data: torch.Tensor,
        s_data: torch.Tensor,
    ) -> torch.Tensor:
        """Average NLL over the trajectory using optional tail-loss slicing."""
        step_losses: list[torch.Tensor] = []
        for k in range(len(t_data)):
            mixture_dist = pf.step(
                t_obs=t_data[[k]],
                s_obs=s_data[[k]],
            )
            start_loss = -loss_tail_steps if loss_tail_steps else k
            last_dist = mixture_dist.distribution(s=s_data[start_loss:])
            step_nll = -last_dist.log_prob(t_data[start_loss:]).mean()
            step_losses.append(step_nll)
        return torch.stack(step_losses).mean()

    for epoch in range(start_epoch, n_epochs):
        eval_fold_losses = []
        train_fold_losses = []

        progress = (epoch - start_epoch) / max(1, total_epochs - 1)
        current_eval_interval = int(
            round(
                eval_initial_interval_epochs
                + (eval_final_interval_epochs - eval_initial_interval_epochs) * progress
            )
        )
        current_eval_interval = max(1, current_eval_interval)
        do_eval_epoch = epoch >= next_eval_epoch or epoch == n_epochs - 1

        # Keep train fold ordering stochastic (but reproducible across full runs).
        for eval_unit in train_rng.permutation(units):
            train_unit_losses = []

            # --------------------------
            # TRAINING
            # --------------------------
            for train_unit in train_rng.permutation(units):
                if train_unit == eval_unit:
                    continue

                if match_train_eval_prior:
                    train_offline_units = [u for u in units if u != train_unit]
                else:
                    train_offline_units = [
                        u for u in units if u not in (train_unit, eval_unit)
                    ]
                train_offline_degmodels = [
                    dev_degmodels[u] for u in train_offline_units
                ]

                train_t_data = dev_data[train_unit][:, 0]
                train_s_data = dev_data[train_unit][:, 1]

                optimizer.zero_grad()

                train_pf = ParticleFilter(
                    base_models=train_offline_degmodels,
                    net=net,
                    n_particles=n_particles,
                    max_life=max_life,
                ).train()

                train_unit_loss = mean_tail_nll(train_pf, train_t_data, train_s_data)
                train_unit_loss.backward()
                optimizer.step()

                train_unit_losses.append(train_unit_loss.item())

            train_fold_loss = float(np.mean(train_unit_losses))
            train_fold_losses.append(train_fold_loss)

            if do_eval_epoch:
                # --------------------------
                # EVALUATION
                # --------------------------
                eval_offline_units = [u for u in units if u != eval_unit]
                eval_offline_degmodels = [dev_degmodels[u] for u in eval_offline_units]

                eval_t_data = dev_data[eval_unit][:, 0]
                eval_s_data = dev_data[eval_unit][:, 1]

                eval_rep_losses = []
                for eval_seed in eval_seed_table[int(eval_unit)]:
                    # Isolate eval RNG from training so reseeding has no side effect.
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(eval_seed)
                        eval_pf = ParticleFilter(
                            base_models=eval_offline_degmodels,
                            net=net,
                            n_particles=n_particles,
                            max_life=max_life,
                        ).eval()
                        eval_loss = mean_tail_nll(eval_pf, eval_t_data, eval_s_data)
                    eval_rep_losses.append(float(eval_loss.item()))

                eval_fold_loss = float(np.mean(eval_rep_losses))
                eval_fold_losses.append(eval_fold_loss)

        # --------------------------
        # epoch statistics
        # --------------------------
        train_epoch_loss = float(np.mean(train_fold_losses))
        eval_epoch_loss = (
            float(np.mean(eval_fold_losses)) if do_eval_epoch else float("nan")
        )

        train_epoch_losses.append(train_epoch_loss)
        eval_epoch_losses.append(eval_epoch_loss)

        # --------------------------
        # SELECTION SCORE (raw eval only)
        # --------------------------
        score = float(eval_epoch_loss)
        scores.append(score)

        if verbose:
            if do_eval_epoch:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train={train_epoch_loss:.3f} | "
                    f"eval={eval_epoch_loss:.3f} | "
                    f"score={score:.3f} "
                    f"(eval_every~{current_eval_interval})"
                )
            else:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train={train_epoch_loss:.3f} | "
                    f"eval=skipped (eval_every~{current_eval_interval})"
                )

        # --------------------------
        # CHECKPOINT SELECTION
        # --------------------------
        if do_eval_epoch:
            next_eval_epoch = epoch + current_eval_interval

            if score < best_score:
                best_score = score

                best_checkpoint = {
                    "epoch": epoch,
                    "model_state": net.state_dict(),
                    "best_score": best_score,
                }

                torch.save(best_checkpoint, checkpoint_best_path)

                if verbose:
                    print(f"  + saved (score={score:.3f})")

        # Persist the LAST checkpoint every epoch so resume points and epoch
        # counts reflect true progress, not just the last best epoch.
        last_checkpoint = {
            "epoch": epoch,
            "model_state": net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_score": best_score,
            "scores": scores,
            "train_losses": train_epoch_losses,
            "eval_losses": eval_epoch_losses,
            "rng_torch": torch.random.get_rng_state(),
            "rng_numpy": np.random.get_state(),
        }

        torch.save(last_checkpoint, checkpoint_last_path)

        # Tiny heartbeat written EVERY epoch (independent of best) so an external
        # monitor can read the current epoch and detect whether the run is alive.
        heartbeat = {
            "epoch": epoch,
            "n_epochs": n_epochs,
            "best_epoch": best_checkpoint["epoch"] if best_checkpoint else -1,
            "best_score": best_score if best_score != float("inf") else None,
            "updated": time.time(),
        }
        with open(checkpoint_last_path.parent / "progress.json", "w") as f:
            json.dump(heartbeat, f)

    return {
        "best_score": best_score,
        "scores": scores,
        "train_epoch_losses": train_epoch_losses,
        "eval_epoch_losses": eval_epoch_losses,
        "epoch": epoch,
        "best_checkpoint": best_checkpoint,
    }
