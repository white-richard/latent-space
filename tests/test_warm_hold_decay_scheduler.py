import math

import pytest
import torch

from latent_space.schedulers.warm_hold_decay_scheduler import WHDScheduler


def make_optimizer(lr=1e-3):
    model = torch.nn.Linear(4, 2)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def make_sched(
    opt,
    *,
    n_iters=1000,
    frac_warmup=0.1,
    init_div=100.0,
    final_factor=0.1,
    decay_type="1-sqrt",
    frac_decay=0.2,
    last_epoch=-1,
    auto_trigger=False,
):
    return WHDScheduler(
        opt,
        n_iterations=n_iters,
        frac_decay=frac_decay,
        final_lr_factor=final_factor,
        frac_warmup=frac_warmup,
        init_div_factor=init_div,
        decay_type=decay_type,
        last_epoch=last_epoch,
        auto_trigger_cooldown=auto_trigger,
    )


# ----------------------------
# Pure multiplier behavior tests
# ----------------------------


@pytest.mark.parametrize("decay_type", ["linear", "cosine", "square", "mirror_cosine", "1-sqrt"])
def test_warmup_shape_and_bounds(decay_type):
    """
    Warmup should:
      - start near 1/init_div_factor at step=0
      - be non-decreasing through warmup
      - end at ~1.0 at step=n_warmup
    We test _multiplier(step) directly to avoid LambdaLR indexing quirks.
    """
    base_lr = 1e-3
    n_iters = 100
    frac_warmup = 0.1
    init_div = 100.0

    opt = make_optimizer(lr=base_lr)
    sched = make_sched(
        opt,
        n_iters=n_iters,
        frac_warmup=frac_warmup,
        init_div=init_div,
        final_factor=0.1,
        decay_type=decay_type,
        frac_decay=None,  # irrelevant for warmup
    )

    n_warmup = int(frac_warmup * n_iters)
    assert n_warmup > 0

    # step=0 warmup start
    m0 = sched._multiplier(0)
    assert abs(m0 - (1.0 / init_div)) < 1e-12

    # warmup should be non-decreasing
    vals = [sched._multiplier(s) for s in range(0, n_warmup)]
    assert all(v2 >= v1 - 1e-12 for v1, v2 in zip(vals, vals[1:]))

    # at step==n_warmup warmup is done => hold (1.0)
    assert abs(sched._multiplier(n_warmup) - 1.0) < 1e-12


def test_hold_is_constant_until_cooldown():
    base_lr = 1e-3
    n_iters = 200
    frac_warmup = 0.1

    opt = make_optimizer(lr=base_lr)
    sched = make_sched(
        opt,
        n_iters=n_iters,
        frac_warmup=frac_warmup,
        final_factor=0.1,
        decay_type="1-sqrt",
        frac_decay=0.2,
    )

    n_warmup = int(frac_warmup * n_iters)

    # pick a few steps after warmup, before cooldown
    for s in [n_warmup, n_warmup + 1, n_warmup + 50]:
        assert abs(sched._multiplier(s) - 1.0) < 1e-12


def test_auto_trigger_cooldown_when_enabled():
    base_lr = 1e-3
    n_iters = 500
    frac_decay = 0.2

    opt = make_optimizer(lr=base_lr)
    sched = make_sched(
        opt,
        n_iters=n_iters,
        frac_warmup=0.0,
        final_factor=0.1,
        decay_type="linear",
        frac_decay=frac_decay,
        auto_trigger=True,
    )

    auto_step = sched.auto_trigger_step
    assert auto_step is not None
    assert auto_step == int((1 - frac_decay) * n_iters)
    assert sched.cooldown_start_step is None

    sched._multiplier(auto_step - 1)
    assert sched.cooldown_start_step is None

    sched._multiplier(auto_step)
    assert sched.cooldown_start_step == auto_step
    assert sched.cooldown_end_step is not None
    assert sched.cooldown_end_step > auto_step
    assert abs(sched._multiplier(sched.cooldown_end_step) - sched.final_lr_factor) < 1e-12


def test_auto_trigger_not_started_when_disabled():
    base_lr = 1e-3
    n_iters = 500
    frac_decay = 0.3

    opt = make_optimizer(lr=base_lr)
    sched = make_sched(
        opt,
        n_iters=n_iters,
        frac_warmup=0.0,
        final_factor=0.1,
        decay_type="linear",
        frac_decay=frac_decay,
        auto_trigger=False,
    )

    step = int((1 - frac_decay) * n_iters)
    sched._multiplier(step)
    assert sched.cooldown_start_step is None

    sched.trigger_cooldown(step=step)
    assert sched.cooldown_start_step == step


def test_auto_trigger_requires_frac_decay():
    opt = make_optimizer()
    with pytest.raises(ValueError, match="auto_trigger_cooldown requires frac_decay to be set"):
        make_sched(
            opt,
            n_iters=100,
            frac_warmup=0.0,
            final_factor=0.1,
            decay_type="linear",
            frac_decay=None,
            auto_trigger=True,
        )


@pytest.mark.parametrize("decay_type", ["linear", "cosine", "square", "mirror_cosine", "1-sqrt"])
def test_cooldown_trigger_explicit_step_decays_over_frac_decay(decay_type):
    """
    If frac_decay is set, triggering cooldown at step S should set:
      cooldown_start_step = S
      cooldown_end_step = min(S + ceil(frac_decay * S), n_iterations)
    and multiplier should:
      - be 1.0 at step S (start of cooldown)
      - be final_lr_factor at step >= cooldown_end_step
    """
    base_lr = 1e-3
    n_iters = 1000
    frac_warmup = 0.0
    final_factor = 0.1
    frac_decay = 0.2

    opt = make_optimizer(lr=base_lr)
    sched = make_sched(
        opt,
        n_iters=n_iters,
        frac_warmup=frac_warmup,
        final_factor=final_factor,
        decay_type=decay_type,
        frac_decay=frac_decay,
    )

    S = 800
    sched.trigger_cooldown(step=S)

    assert sched.cooldown_start_step == S
    expected_len = max(1, int(math.ceil(frac_decay * S)))
    expected_end = min(S + expected_len, n_iters)
    assert sched.cooldown_end_step == expected_end

    # At start of cooldown, curve begins at 1.0
    assert abs(sched._multiplier(S) - 1.0) < 1e-12

    # Just before end should be > final
    if expected_end - 1 > S:
        assert sched._multiplier(expected_end - 1) > final_factor

    # At/after end should equal final_factor
    assert abs(sched._multiplier(expected_end) - final_factor) < 1e-12
    assert abs(sched._multiplier(expected_end + 10) - final_factor) < 1e-12


def test_frac_decay_none_decays_to_end_of_training():
    base_lr = 1e-3
    n_iters = 1000
    final_factor = 0.1

    opt = make_optimizer(lr=base_lr)
    sched = make_sched(
        opt,
        n_iters=n_iters,
        frac_warmup=0.0,
        final_factor=final_factor,
        decay_type="1-sqrt",
        frac_decay=None,  # decay to end of training
    )

    S = 400
    sched.trigger_cooldown(step=S)
    assert sched.cooldown_start_step == S
    assert sched.cooldown_end_step == n_iters

    assert abs(sched._multiplier(S) - 1.0) < 1e-12
    assert abs(sched._multiplier(n_iters) - final_factor) < 1e-12


def test_step_ge_n_iterations_clamps_to_final():
    base_lr = 1e-3
    n_iters = 10
    final_factor = 0.1

    opt = make_optimizer(lr=base_lr)
    sched = make_sched(
        opt,
        n_iters=n_iters,
        frac_warmup=0.0,
        final_factor=final_factor,
        decay_type="1-sqrt",
        frac_decay=None,
    )

    assert abs(sched._multiplier(n_iters) - final_factor) < 1e-12
    assert abs(sched._multiplier(n_iters + 1) - final_factor) < 1e-12


# ----------------------------
# Decay type coverage
# ----------------------------


@pytest.mark.parametrize("decay_type", ["linear", "cosine", "square", "mirror_cosine", "1-sqrt"])
def test_decay_is_nonincreasing_during_cooldown(decay_type):
    """
    During cooldown, multiplier should be non-increasing (monotone down).
    (Mirror cosine can overshoot in some constructions; your implementation is intended to remain in [0,1]
     but we still only assert non-increasing for your current formula.)
    """
    base_lr = 1e-3
    n_iters = 200
    final_factor = 0.1

    opt = make_optimizer(lr=base_lr)
    sched = make_sched(
        opt,
        n_iters=n_iters,
        frac_warmup=0.0,
        final_factor=final_factor,
        decay_type=decay_type,
        frac_decay=None,
    )

    S = 50
    sched.trigger_cooldown(step=S)

    vals = [sched._multiplier(s) for s in range(S, n_iters + 1)]
    assert all(v2 <= v1 + 1e-12 for v1, v2 in zip(vals, vals[1:]))
    assert abs(vals[-1] - final_factor) < 1e-12


@pytest.mark.xfail(
    reason="Current exp implementation depends on final_lr_factor and is double-scaled; fix exp unit curve."
)
def test_exp_decay_reaches_final_factor_at_end():
    """
    This is the regression test that will fail until 'exp' is fixed to be compatible
    with unit-curve + scaling (or handled as a special case).
    """
    base_lr = 1e-3
    n_iters = 100
    final_factor = 0.1

    opt = make_optimizer(lr=base_lr)
    sched = make_sched(
        opt,
        n_iters=n_iters,
        frac_warmup=0.0,
        final_factor=final_factor,
        decay_type="exp",
        frac_decay=None,
    )

    S = 10
    sched.trigger_cooldown(step=S)
    assert abs(sched._multiplier(n_iters) - final_factor) < 1e-12


# ----------------------------
# Checkpointing / resume behavior
# ----------------------------


def test_state_dict_roundtrip_preserves_cooldown_state():
    base_lr = 1e-3

    opt1 = make_optimizer(lr=base_lr)
    sched1 = make_sched(
        opt1,
        n_iters=1000,
        frac_warmup=0.0,
        final_factor=0.1,
        decay_type="linear",
        frac_decay=0.2,
    )
    sched1.trigger_cooldown(step=200)

    sd = sched1.state_dict()

    opt2 = make_optimizer(lr=base_lr)
    sched2 = make_sched(
        opt2,
        n_iters=1000,
        frac_warmup=0.0,
        final_factor=0.1,
        decay_type="linear",
        frac_decay=0.2,
    )
    sched2.load_state_dict(sd)

    assert sched2.cooldown_start_step == sched1.cooldown_start_step
    assert sched2.cooldown_end_step == sched1.cooldown_end_step
    assert sched2.last_epoch == sched1.last_epoch


def test_resume_with_last_epoch_does_not_require_initial_lr_preexisting():
    """
    Should not raise KeyError because WHDScheduler sets param_group['initial_lr'] automatically.
    """
    base_lr = 1e-3
    opt = make_optimizer(lr=base_lr)

    sched = make_sched(
        opt,
        n_iters=100,
        frac_warmup=0.0,
        final_factor=0.1,
        decay_type="linear",
        frac_decay=None,
        last_epoch=10,  # resume
    )

    assert "initial_lr" in opt.param_groups[0]
    assert isinstance(sched, WHDScheduler)


def test_start_cooldown_immediately_sets_some_start_and_end():
    base_lr = 1e-3
    opt = make_optimizer(lr=base_lr)

    sched = WHDScheduler(
        opt,
        n_iterations=1000,
        frac_decay=0.2,
        final_lr_factor=0.1,
        frac_warmup=0.0,
        init_div_factor=100.0,
        decay_type="linear",
        last_epoch=20,
        start_cooldown_immediately=True,
    )

    assert sched.cooldown_start_step is not None
    assert sched.cooldown_end_step is not None
    assert sched.cooldown_end_step <= sched.n_iterations


# ----------------------------
# Integration smoke test (scheduler.step)
# ----------------------------


def test_integration_step_runs_and_lr_changes_when_cooldown_triggered():
    """
    Only a smoke test: ensures scheduler.step() runs and LR eventually decreases after trigger.
    Does NOT assert the exact first-step value (PyTorch indexing/order dependent).
    """
    base_lr = 1e-3
    n_iters = 200

    opt = make_optimizer(lr=base_lr)
    sched = make_sched(
        opt,
        n_iters=n_iters,
        frac_warmup=0.1,
        final_factor=0.1,
        decay_type="1-sqrt",
        frac_decay=None,
    )

    lrs = []
    trigger_at = 80

    for step in range(n_iters):
        if step == trigger_at:
            # explicit start to remove last_epoch ambiguity
            sched.trigger_cooldown(step=step)

        sched.step()
        lrs.append(opt.param_groups[0]["lr"])

    # Should have at least one LR drop after trigger
    assert min(lrs[trigger_at + 1 :]) < max(lrs[: trigger_at + 1])
    # End should be at final factor * base_lr (allow tolerance)
    assert abs(lrs[-1] - base_lr * 0.1) < 1e-9
