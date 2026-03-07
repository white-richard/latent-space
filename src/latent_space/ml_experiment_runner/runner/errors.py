"""Custom exceptions and warnings for the ML experiment runner."""

from __future__ import annotations


class AllSeedsFailedError(Exception):
    """Raised when every seed in an experiment raises an exception."""


class InsufficientSeedsError(Exception):
    """Raised when the number of successful seeds is below min_successful_seeds."""


class MetricShapeMismatchWarning(UserWarning):
    """Emitted when sequence metrics have different lengths across seeds."""


class UnsupportedMetricTypeWarning(UserWarning):
    """Emitted when a leaf metric field is not float, int, or list."""


class ExportError(Exception):
    """Wraps I/O or formatting failures during result export."""
