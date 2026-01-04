import math
from typing import Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def wsd_schedule(
    n_iterations: int,
    final_lr_factor: float = 0.1,
    frac_warmup: float = 0.1,
    init_div_factor: float = 100,
    fract_decay: float = 0.1,
    decay_type: str = "sqrt",
) -> Callable[[int], float]:
    """Warmup, hold, and decay schedule.
    Args:
        n_iterations: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end; suggested to be swept on
        frac_warmup: fraction of total iterations used for warmup (0 <= frac_warmup < 1)
        init_div_factor: initial division factor for warmup. Simply, initial LR = max_lr / init_div_factor
        fract_decay: fraction of iterations used for decay
        decay_type: type of decay schedule - one of ['linear', 'exp', 'cosine',
            'mirror_cosine', 'square', 'sqrt']. Default: 'sqrt'
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate

    References:
        Alexander H. et al. (2024). "Scaling Laws and Compute-Optimal
        Training Beyond Fixed Training Durations."
        Source: https://github.com/epfml/schedules-and-scaling
    """
    valid_decay_types = ["linear", "exp", "cosine", "mirror_cosine", "square", "sqrt"]
    if decay_type not in valid_decay_types:
        raise ValueError(
            f"decay_type '{decay_type}' is not valid. Must be one of {valid_decay_types}"
        )

    if not (0 <= frac_warmup < 1):
        raise ValueError("frac_warmup must be in [0, 1)")

    n_warmup = int(frac_warmup * n_iterations)

    n_anneal_steps = int(fract_decay * n_iterations)
    n_hold = n_iterations - n_anneal_steps  # stable hold phase

    def schedule(step):
        if step < n_warmup:
            return (step / n_warmup) + (1 - step / n_warmup) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < n_iterations:
            if decay_type == "linear":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
            elif decay_type == "exp":
                return final_lr_factor ** ((step - n_hold) / n_anneal_steps)
            elif decay_type == "cosine":
                return (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
            elif decay_type == "mirror_cosine":
                cosine_value = (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
                linear_value = final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
                return linear_value * 2 - cosine_value
            elif decay_type == "square":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - ((step - n_hold) / n_anneal_steps) ** 2
                )

            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - math.sqrt((step - n_hold) / n_anneal_steps)
                )

        else:
            return final_lr_factor

    return schedule


def get_wsd_scheduler(
    optimizer: Optimizer,
    n_iterations: int,
    final_lr_factor: float = 0.01,
    frac_warmup: float = 0.1,
    init_div_factor: float = 100,
    fract_decay: float = 0.1,
    decay_type: str = "sqrt",
    last_epoch: int = -1,
) -> LambdaLR:
    """PyTorch LambdaLR scheduler with warmup, stable, and decay phases.

    This is a convenience wrapper around wsd_schedule that returns a PyTorch
    scheduler object ready to use in training loops.

    Args:
        optimizer: PyTorch optimizer to schedule
        n_iterations: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end; suggested to be swept on
        frac_warmup: fraction of total iterations used for warmup (0 <= frac_warmup < 1)
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
        decay_type: type of decay schedule - one of ['linear', 'exp', 'cosine',
            'mirror_cosine', 'square', 'sqrt']. Default: 'sqrt'
        last_epoch: the index of the last epoch (for resuming training)

    Returns:
        scheduler: PyTorch LambdaLR scheduler

    Example:
        >>> import torch
        >>> model = torch.nn.Linear(10, 1)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = get_wsd_scheduler(optimizer, n_iterations=10000, frac_warmup=0.1)
        >>> for step in range(10000):
        ...     loss = train_step()
        ...     optimizer.step()
        ...     scheduler.step()

    References:
        Alexander H. et al. (2024). "Scaling Laws and Compute-Optimal
        Training Beyond Fixed Training Durations."
        Source: https://github.com/epfml/schedules-and-scaling
    """
    schedule_fn = wsd_schedule(
        n_iterations=n_iterations,
        final_lr_factor=final_lr_factor,
        frac_warmup=frac_warmup,
        init_div_factor=init_div_factor,
        fract_decay=fract_decay,
        decay_type=decay_type,
    )

    return LambdaLR(optimizer, lr_lambda=schedule_fn, last_epoch=last_epoch)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    n_iters = 10000
    schedule_fn = wsd_schedule(
        n_iterations=n_iters,
        final_lr_factor=0.1,
        frac_warmup=0.1,
        init_div_factor=100,
        fract_decay=0.1,
        decay_type="sqrt",
    )

    # Generate learning rate multipliers
    steps = np.arange(n_iters)
    lr_multipliers = [schedule_fn(step) for step in steps]

    plt.figure(figsize=(12, 6))
    plt.plot(steps, lr_multipliers, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("LR Multiplier")
    plt.title("Warmup-Stable-Decay Schedule (sqrt decay)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Max LR")
    plt.axhline(y=0.01, color="g", linestyle="--", alpha=0.5, label="Final LR")
    plt.legend()
    plt.tight_layout()
    plt.savefig("temp/wsd_schedule.png", dpi=150)
    print(f"Schedule plot saved to wsd_schedule.png")

    # Compare different decay types
    decay_types = ["cosine", "linear", "sqrt", "square", "exp", "mirror_cosine"]
    plt.figure(figsize=(14, 8))

    for decay_type in decay_types:
        schedule_fn = wsd_schedule(
            n_iterations=n_iters,
            final_lr_factor=0.1,
            frac_warmup=0.1,
            fract_decay=0.1,
            decay_type=decay_type,
        )
        lr_multipliers = [schedule_fn(step) for step in steps]
        plt.plot(steps, lr_multipliers, label=decay_type, linewidth=2)

    plt.xlabel("Iteration")
    plt.ylabel("LR Multiplier")
    plt.title("Comparison of Decay Types")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("temp/wsd_schedule_comparison.png", dpi=150)
    print(f"Comparison plot saved to wsd_schedule_comparison.png")

    import torch

    print("\n--- PyTorch Example ---")
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_wsd_scheduler(
        optimizer, n_iterations=10000, frac_warmup=0.1, fract_decay=0.1, decay_type="sqrt"
    )

    print(f"Initial LR: {optimizer.param_groups[0]['lr']:.6f}")

    for step in range(5):
        scheduler.step()
        print(f"Step {step+1}, LR: {optimizer.param_groups[0]['lr']:.6f}")
