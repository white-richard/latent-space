"""A lightweight shim package to provide a minimal subset of the `dinov3` API
used by this repository when the real `dinov3` package is not installed.

This module intentionally exposes only a small, well-defined surface that the
local vision transformer implementation expects (primarily `dinov3.utils`).
If you have the official `dinov3` package available, prefer to install it and
remove this shim.

Notes:
- This shim is a compatibility convenience for development and testing. It is
  not a full reimplementation of the upstream package.
- The shim exposes the most commonly-used helpers from `dinov3.utils` on the
  package level for convenience, in addition to keeping the real module
  available under `dinov3.utils`.

"""

from __future__ import annotations

from typing import NoReturn

__all__ = [
    "__version__",
    "cat_keep_shapes",
    "is_shim",
    "named_apply",
    "named_replace",
    "uncat_with_shapes",
    "utils",
]

# Version identifier for the shim package.
__version__ = "0.0.0-shim"

# Import the local utils implementation if available; otherwise expose a clear
# fallback module error at import-time when the helper is not present.
try:
    # Relative import to the shim module `dinov3.utils` implemented alongside this file.
    from . import utils  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    # Provide a lightweight fallback object so attribute access errors are clear.
    class _MissingUtilsModule:
        def __getattr__(self, name):
            msg = (
                "The local `dinov3.utils` shim module could not be imported. "
                "If you intended to use the real `dinov3` package, please install it, "
                "or ensure the project contains `dinov3/utils.py`."
            )
            raise ModuleNotFoundError(
                msg,
            )

    utils = _MissingUtilsModule()  # type: ignore

# Re-export commonly-used helpers at package level for convenience.
# These names are expected by various modules in the repository.
try:
    cat_keep_shapes = utils.cat_keep_shapes  # type: ignore
    uncat_with_shapes = utils.uncat_with_shapes  # type: ignore
    named_replace = utils.named_replace  # type: ignore
    named_apply = utils.named_apply  # type: ignore
except Exception:
    # If any of the expected attributes are missing, create callables that raise
    # clear errors at runtime to help debugging.
    def _missing_attr(name: str):
        def _fn(*_args, **_kwargs) -> NoReturn:
            msg = (
                f"dinov3 shim missing required attribute '{name}'. "
                "Install the real `dinov3` package or add the missing shim helpers."
            )
            raise AttributeError(
                msg,
            )

        return _fn

    cat_keep_shapes = _missing_attr("cat_keep_shapes")
    uncat_with_shapes = _missing_attr("uncat_with_shapes")
    named_replace = _missing_attr("named_replace")
    named_apply = _missing_attr("named_apply")


# Helper to indicate that this is a shim package.
def is_shim() -> bool:
    """Return True if this package is the lightweight shim (always True)."""
    return True
