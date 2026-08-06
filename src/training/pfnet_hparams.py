from typing import Any


def network(
    hidden_dims: list[int] | None = None,
    leaky_slope: float = 0.05,
    dropout: float = 0.1,
    activation: str = "leaky",
) -> dict[str, Any]:
    return {
        "HIDDEN_DIMS": [128, 64, 32] if hidden_dims is None else hidden_dims,
        "LEAKY_SLOPE": leaky_slope,
        "DROPOUT": dropout,
        "ACTIVATION": activation,
    }


def particle_filter(
    n_particles: int = 2000,
    max_life: int = 100,
) -> dict[str, Any]:
    return {
        "N_PARTICLES": n_particles,
        "MAX_LIFE": max_life,
    }


def training(
    lr: float = 5e-4,
    n_epochs: int = 300,
    loss_tail_steps: int = 5,
    match_train_eval_prior: bool = False,
) -> dict[str, Any]:
    return {
        "LR": lr,
        "N_EPOCHS": n_epochs,
        "LOSS_TAIL_STEPS": loss_tail_steps,
        "MATCH_TRAIN_EVAL_PRIOR": match_train_eval_prior,
    }


def evaluation(
    repetitions: int = 10,
    initial_interval_epochs: int = 12,
    final_interval_epochs: int = 1,
    seed_stride: int = 100_000,
) -> dict[str, Any]:
    return {
        "REPETITIONS": repetitions,
        "INITIAL_INTERVAL_EPOCHS": initial_interval_epochs,
        "FINAL_INTERVAL_EPOCHS": final_interval_epochs,
        "SEED_STRIDE": seed_stride,
    }


def gains(
    noise: "float | list[float] | None" = None,
    prior: "float | list[float] | None" = None,
    lik: "float | None" = None,
) -> dict[str, Any]:
    """Fixed PF gains for the no-network baseline (use_net=False).

    Each entry is None (-> PF default), a scalar (broadcast over state dims), or a
    length-state_dim vector (per-dim). Ignored when a network is used.
    """
    return {"NOISE": noise, "PRIOR": prior, "LIK": lik}


# sentinel so an explicit net=None means "no network" (baseline), while an
# omitted net falls back to the default funnel network
_DEFAULT_NET: Any = object()


def make_args(
    net: dict[str, Any] | None = _DEFAULT_NET,
    pf: dict[str, Any] | None = None,
    train: dict[str, Any] | None = None,
    eval_: dict[str, Any] | None = None,
    gains_: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Assemble one arg set; omitted categories fall back to args1 defaults.

    ``net=None`` marks a no-network baseline (NETWORK is None, use_net False);
    omit ``net`` to get the default network.
    """
    return {
        "NETWORK": network() if net is _DEFAULT_NET else net,
        "PARTICLE_FILTER": particle_filter() if pf is None else pf,
        "TRAINING": training() if train is None else train,
        "EVALUATION": evaluation() if eval_ is None else eval_,
        "GAINS": gains() if gains_ is None else gains_,
    }


PFNET_ARGS: dict[int, dict[str, dict[str, Any] | None]] = {
    # args0: ablation baseline (bootstrap PF). net=None -> no network, so the 13
    # dynamic gains are replaced by fixed constants inside ParticleFilter
    # (noise scale 1.0, prior weight 0.0, likelihood weight 1.0).
    0: make_args(net=None),
    # args1: baseline funnel network.
    1: make_args(),
    # args2: smaller network to reduce overfitting of the (t, s) -> gains surface.
    2: make_args(net=network(hidden_dims=[32, 32])),
    # args3: args1 network, but the training prior matches the eval prior (n-1).
    3: make_args(train=training(match_train_eval_prior=True)),
    # args4: one extra layer at similar width; intermediate capacity above [32, 32].
    4: make_args(net=network(hidden_dims=[32, 32, 32])),
    # args5: args2 architecture but a smooth (tanh) activation instead of the
    # piecewise-linear LeakyReLU, better suited to a small smooth-surface network.
    5: make_args(net=network(hidden_dims=[32, 32], activation="tanh")),
    # args6: args5 with more capacity (wider + deeper) to exploit the smooth tanh
    # surface without the LeakyReLU kinks; tests whether extra width helps.
    6: make_args(net=network(hidden_dims=[16, 16], activation="tanh")),
    # args7: even smaller network to test the limits of underfitting.
    7: make_args(net=network(hidden_dims=[16, 16])),
    # args10: tuned bootstrap-PF baseline. net=None -> no network; constant gains
    # from the 3-parameter Optuna search (5o-pf_controller_optuna).
    10: make_args(net=None, gains_=gains(noise=1.7317, prior=0.0496, lik=0.266)),
}
