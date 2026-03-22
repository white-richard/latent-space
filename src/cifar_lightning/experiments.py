#!/usr/bin/env python3
"""CLI entrypoint for the cifar_lightning package.

This module is intended to be executed with:

    python -m cifar_lightning.experiments [ARGS...]

It delegates argument parsing and the training run to the `train` module.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

# Import the training helpers from the sibling module.
# We import the functions we need so this module is a thin wrapper.
from . import train as train_module

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run a training experiment.

    Args:
        argv: Optional sequence of command-line arguments (excluding program name).
              If None, sys.argv[1:] is used.

    Returns:
        exit code (0 on success).

    """
    # Use provided argv or CLI args if None
    argv_list = list(argv) if argv is not None else None

    # Let the train module build the argparse.Namespace from argv
    parsed = train_module.parse_args(argv_list)

    # Convert parsed Namespace into a simple dataclass-style config expected by train()
    cfg = train_module.namespace_to_dataclass(parsed)

    # Run training
    metrics = train_module.train(cfg)

    # Print metrics in a concise form for the caller
    print("Test metrics:", metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
