import torch
import torch.nn.functional as F


class StochasticSkipCrossEntropy(torch.nn.Module):
    def __init__(
        self,
        p_on_correct: float = 0.5,
        weight=None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = 'mean',
        normalize_over_included: bool = True,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        if not (0.0 <= p_on_correct <= 1.0):
            raise ValueError("p_on_correct must be in [0, 1].")
        if reduction not in ('none', 'mean', 'sum'):
            raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
        self.p_on_correct = float(p_on_correct)
        self.register_buffer('weight', weight if weight is None else torch.as_tensor(weight, dtype=torch.float))
        self.ignore_index = ignore_index
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction
        self.normalize_over_included = bool(normalize_over_included)
        self._ext_generator = generator  # may be None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, C, ...) raw, unnormalized scores
        targets: (N, ...) class indices
        """
        # Compute per-sample CE (flatten any trailing dims)
        orig_shape = targets.shape
        N = targets.numel()
        if logits.ndim > 2:
            # reshape to (N, C)
            C = logits.size(1)
            ce_logits = logits.permute(0, *range(2, logits.ndim), 1).contiguous().view(N, C)
        else:
            ce_logits = logits
        flat_targets = targets.view(-1)

        per_example = F.cross_entropy(
            ce_logits,
            flat_targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.label_smoothing
        )  # shape: (N,)

        # Determine which predictions are correct (ignoring 'ignore_index')
        with torch.no_grad():
            preds = ce_logits.argmax(dim=1)
            valid = (flat_targets != self.ignore_index)
            correct = (preds == flat_targets) & valid
            incorrect = (~correct) & valid

            # On correct predictions: include with probability p_on_correct
            if self._ext_generator is not None:
                r = torch.rand(correct.sum().item(), generator=self._ext_generator, device=ce_logits.device)
                include_correct = torch.zeros_like(correct, dtype=torch.bool)
                include_correct[correct] = (r < self.p_on_correct)
            else:
                include_correct = torch.zeros_like(correct, dtype=torch.bool)
                include_correct[correct] = (torch.rand(correct.sum().item(), device=ce_logits.device) < self.p_on_correct)

            include_mask = incorrect | include_correct  # bool mask over N elements

        # Zero out skipped items
        masked_losses = torch.where(include_mask, per_example, torch.zeros_like(per_example))

        if self.reduction == 'none':
            return masked_losses.view(orig_shape)
        elif self.reduction == 'sum':
            return masked_losses.sum()
        else:  # 'mean'
            if self.normalize_over_included:
                denom = include_mask.sum()
                # If no samples included (all correct & skipped), return 0 on the right device/dtype
                if denom.item() == 0:
                    return masked_losses.sum()  # zero, correct dtype/device
                return masked_losses.sum() / denom
            else:
                # Average over the full batch (zeros count), matching standard CE's 'mean' shape
                return masked_losses.mean()