"""ConfigAdapter Protocol and DataclassAdapter implementation."""

from __future__ import annotations

import copy
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConfigAdapter(Protocol):
    """Protocol for adapting experiment configs to support seed injection."""

    def inject_seed(self, config: Any, seed: int) -> Any:
        """Return a copy of config with the seed set."""
        ...

    def to_native(self, config: Any) -> Any:
        """Convert config to the native form expected by train_fn."""
        ...


def _set_dotted(obj: Any, path: str, value: Any) -> None:
    """Set a dotted-path attribute on a (possibly nested) object.

    Raises AttributeError if any intermediate attribute is missing.
    """
    parts = path.split(".")
    target = obj
    for attr in parts[:-1]:
        target = getattr(target, attr)
    setattr(target, parts[-1], value)


class DataclassAdapter:
    """Adapter for dataclass-based configs.

    Parameters
    ----------
    seed_field:
        Dotted path to the seed attribute, e.g. ``"experiment.seed"``.

    """

    def __init__(self, seed_field: str = "experiment.seed") -> None:
        self.seed_field = seed_field

    def inject_seed(self, config: Any, seed: int) -> Any:
        seeded = copy.deepcopy(config)
        _set_dotted(seeded, self.seed_field, seed)
        return seeded

    def to_native(self, config: Any) -> Any:
        return config
