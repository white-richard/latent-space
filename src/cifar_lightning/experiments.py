from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from . import train as train_module

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    argv_list = list(argv) if argv is not None else None

    parsed = train_module.parse_args(argv_list)
    cfg = train_module.namespace_to_dataclass(parsed)

    metrics = train_module.train(cfg)

    print("Test metrics:", metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
