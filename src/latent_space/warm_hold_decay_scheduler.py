import math
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class WHDScheduler(LambdaLR):
    """
    Warmup -> Hold (constant) -> Optional decay to the end (triggered manually).

    - If cooldown NOT triggered: after warmup, multiplier stays 1.0 until n_iterations.
    - If cooldown triggered at step S: decay starts at S and ends at n_iterations
      with final_lr_factor.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        n_iterations: int,  # total number of iterations for the FULL run
        frac_decay: float | None = 0.2,  # fraction of current iterations to decay over
        final_lr_factor: float = 0.1,  # multiplier at the very end
        frac_warmup: float = 0.1,
        init_div_factor: float = 100.0,
        decay_type: str = "1-sqrt",  # ['linear','exp','cosine','mirror_cosine','square','1-sqrt']
        last_epoch: int = -1,
        start_cooldown_immediately: bool = False,
    ):
        self.n_iterations = n_iterations
        self.final_lr_factor = final_lr_factor
        self.frac_warmup = frac_warmup
        self.init_div_factor = init_div_factor
        self.decay_type = decay_type
        self.frac_decay = frac_decay

        if self.frac_decay is not None and not (0.0 < self.frac_decay <= 1.0):
            raise ValueError("frac_decay must be in (0, 1] or None")

        self.n_warmup = int(self.frac_warmup * self.n_iterations)
        self.cooldown_start_step: int | None = None  # when decay begins
        self.cooldown_end_step: int | None = None  # when decay ends

        def lr_lambda(step: int) -> float:
            return self._multiplier(step)

        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])

        super().__init__(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

        if start_cooldown_immediately:
            # start cooldown at the *next* step that will be scheduled
            self.trigger_cooldown(step=self.last_epoch + 1)

    def trigger_cooldown(self, step: int | None = None) -> None:
        """
        Begin cooldown so decay starts immediately from `step` (defaults to next scheduled step).
        """
        if step is None:
            step = self.last_epoch + 1

        # don't start cooldown before warmup finishes
        step = max(step, self.n_warmup)
        self.cooldown_start_step = step

        if self.frac_decay is None:
            end = self.n_iterations
        else:
            decay_len = max(1, int(math.ceil(self.frac_decay * step)))
            end = step + decay_len

        self.cooldown_end_step = min(end, self.n_iterations)

    def _multiplier(self, step: int) -> float:
        # Past the planned end: clamp to final factor
        if step >= self.n_iterations:
            return self.final_lr_factor

        # Warmup
        if self.n_warmup > 0 and step < self.n_warmup:
            x = step / self.n_warmup
            return x + (1 - x) / self.init_div_factor

        # Hold forever unless cooldown triggered
        if self.cooldown_start_step is None or step < self.cooldown_start_step:
            return 1.0

        # If cooldown end is reached, stick at final factor
        assert self.cooldown_end_step is not None
        if step >= self.cooldown_end_step:
            return self.final_lr_factor

        # Cooldown progress
        decay_len = max(1, self.cooldown_end_step - self.cooldown_start_step)
        t = (step - self.cooldown_start_step) / decay_len
        t = min(max(t, 0.0), 1.0)

        return self._scaled_decay_value(t)

    def _scaled_decay_value(self, t: float) -> float:
        """Scale decay curve so it goes from 1 -> final_lr_factor."""
        raw = self._decay_unit(t)  # 1 -> 0
        return self.final_lr_factor + (1.0 - self.final_lr_factor) * raw

    def _decay_unit(self, t: float) -> float:
        """Returns a curve that starts at 1 when t=0 and ends at 0 when t=1."""
        if self.decay_type == "linear":
            return 1 - t
        if self.decay_type == "exp":
            k = 5.0  # shape; larger = faster drop early
            return (math.exp(-k * t) - math.exp(-k)) / (1 - math.exp(-k))
        if self.decay_type == "cosine":
            return (1 + math.cos(math.pi * t)) * 0.5
        if self.decay_type == "mirror_cosine":
            cosine_value = (1 + math.cos(math.pi * t)) * 0.5
            linear_value = 1 - t
            return linear_value * 2 - cosine_value
        if self.decay_type == "square":
            return 1 - t**2
        if self.decay_type == "1-sqrt":
            return 1 - math.sqrt(t)
        raise ValueError(f"Unknown decay_type: {self.decay_type}")

    def state_dict(self) -> dict[str, Any]:
        sd = super().state_dict()
        sd["cooldown_start_step"] = self.cooldown_start_step
        sd["cooldown_end_step"] = self.cooldown_end_step
        return sd

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.cooldown_start_step = state_dict.pop("cooldown_start_step", None)
        self.cooldown_end_step = state_dict.pop("cooldown_end_step", None)
        super().load_state_dict(state_dict)


if __name__ == "__main__":
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # ---- setup ----
    os.makedirs("temp", exist_ok=True)

    base_lr = 1e-3
    n_iters = 10_000
    frac_warmup = 0.1
    final_lr_factor = 0.1

    trigger_step_for_plots = 8_000  # when to start cooldown in the plots
    model = torch.nn.Linear(10, 1)

    def make_optimizer(params):
        return torch.optim.AdamW(params, lr=base_lr)

    def make_scheduler(optimizer, decay_type: str, start_cooldown_immediately: bool = False):
        return WHDScheduler(
            optimizer,
            n_iterations=n_iters,
            frac_warmup=frac_warmup,
            final_lr_factor=final_lr_factor,
            decay_type=decay_type,
            init_div_factor=100.0,
            start_cooldown_immediately=start_cooldown_immediately,
        )

    def simulate_lrs(decay_type: str, trigger_at: int | None):
        optimizer = make_optimizer(model.parameters())
        scheduler = make_scheduler(optimizer, decay_type)

        lrs = np.empty(n_iters, dtype=np.float64)

        for step in range(n_iters):
            if trigger_at is not None and step == trigger_at:
                scheduler.trigger_cooldown()

            scheduler.step()
            lrs[step] = optimizer.param_groups[0]["lr"]

        multipliers = lrs / base_lr
        return multipliers, lrs

    optimizer = make_optimizer(model.parameters())

    # ---- single schedule plot (with cooldown trigger) ----
    multipliers, lrs = simulate_lrs("1-sqrt", trigger_at=trigger_step_for_plots)
    steps = np.arange(n_iters)

    plt.figure(figsize=(12, 6))
    plt.plot(steps, multipliers, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("LR Multiplier (lr / base_lr)")
    plt.title(f"Warmup-Hold + Triggered Cooldown (1-sqrt), trigger@{trigger_step_for_plots}")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, linestyle="--", alpha=0.5, label="Hold (1.0)")
    plt.axhline(y=final_lr_factor, linestyle="--", alpha=0.5, label="Final factor")
    plt.axvline(x=trigger_step_for_plots, linestyle="--", alpha=0.5, label="Cooldown trigger")
    plt.legend()
    plt.tight_layout()
    plt.savefig("temp/whd_triggered_schedule.png", dpi=150)
    print("Saved: temp/whd_triggered_schedule.png")

    # ---- compare decay types (all triggered at same step) ----
    decay_types = ["cosine", "linear", "1-sqrt", "square", "exp", "mirror_cosine"]
    plt.figure(figsize=(14, 8))

    for decay_type in decay_types:
        multipliers, _ = simulate_lrs(decay_type, trigger_at=trigger_step_for_plots)
        plt.plot(steps, multipliers, label=decay_type, linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("LR Multiplier (lr / base_lr)")
    plt.title(f"Decay Type Comparison (trigger@{trigger_step_for_plots})")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, linestyle="--", alpha=0.5)
    plt.axhline(y=final_lr_factor, linestyle="--", alpha=0.5)
    plt.axvline(x=trigger_step_for_plots, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("temp/whd_triggered_comparison.png", dpi=150)
    print("Saved: temp/whd_triggered_comparison.png")

    # ---- print LR progression + trigger cooldown mid-run (toy demo) ----
    # Reset optimizer lr and build a fresh scheduler
    optimizer.param_groups[0]["lr"] = base_lr
    optimizer.param_groups[0].pop("initial_lr", None)  # optional

    scheduler = make_scheduler(optimizer, "1-sqrt")

    print(f"\nInitial LR: {optimizer.param_groups[0]['lr']:.6f}")
    for step in range(100):
        if step == 80:
            scheduler.trigger_cooldown()
            print(f"Triggered cooldown at step {step}")

        scheduler.step()
        print(f"Step {step + 1}, LR: {optimizer.param_groups[0]['lr']:.6f}")
