from timm.scheduler import CosineLRScheduler


def create_cosine_scheduler(optimizer, epochs:int, lr_min:float, warmup_t:int, warmup_lr_init:float=1e-6, per_epoch:bool=True):
    """
    Create a cosine learning rate scheduler with warmup.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose learning rate will be scheduled.
        epochs (int): Total number of training epochs.
        lr_min (float): Minimum learning rate at the end of the schedule.
        warmup_t (int): Number of warmup epochs.
        warmup_lr_init (float, optional): Initial learning rate for warmup. Defaults to 1e-6.
        per_epoch (bool, optional): If True, the scheduler steps per epoch; if False, per iteration. Defaults to True.

    Example:
        scheduler = create_cosine_scheduler(optimizer, epochs=100, lr_min=1e-5, warmup_t=10)

        scheduler.step(epoch)  # Call this at the end of each epoch if per_epoch is True
        or
        scheduler.step_update(iteration)  # Call this at each iteration if per_epoch is False

    Returns:
        CosineLRScheduler: Configured cosine learning rate scheduler.
    """
    return CosineLRScheduler(
        optimizer,
        t_initial=epochs,
        lr_min=lr_min,
        warmup_lr_init=warmup_lr_init,
        warmup_t=warmup_t,
        cycle_limit=1,
        t_in_epochs=per_epoch,
    )
