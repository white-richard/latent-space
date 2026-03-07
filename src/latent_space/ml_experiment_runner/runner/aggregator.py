"""MetricsAggregator: aggregate metric dataclasses across multiple seeds."""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Union

import numpy as np

from .errors import MetricShapeMismatchWarning, UnsupportedMetricTypeWarning

# Recursive type alias
AggregatedMetrics = dict[str, Union["AggregatedLeaf", "AggregatedMetrics"]]


@dataclasses.dataclass
class AggregatedLeaf:
    mean: float | list[float]
    std: float | list[float]
    min: float | list[float]
    max: float | list[float]
    n_seeds: int


def _walk(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Recursively extract {dotted_path: value} from a dataclass."""
    result: dict[str, Any] = {}
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return result
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        key = f"{prefix}.{f.name}" if prefix else f.name
        if dataclasses.is_dataclass(val) and not isinstance(val, type):
            result.update(_walk(val, key))
        else:
            result[key] = val
    return result


def _build_nested(flat: dict[str, AggregatedLeaf]) -> AggregatedMetrics:
    """Convert {dotted_path: AggregatedLeaf} to a nested dict."""
    out: AggregatedMetrics = {}
    for path, leaf in flat.items():
        parts = path.split(".")
        node: dict[str, Any] = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = leaf
    return out


class MetricsAggregator:
    @staticmethod
    def aggregate(results: list[Any]) -> AggregatedMetrics:
        """Aggregate a list of metrics dataclasses into AggregatedMetrics."""
        if not results:
            return {}

        # Discover leaf paths from the first result
        leaf_paths = list(_walk(results[0]).keys())

        flat_aggregated: dict[str, AggregatedLeaf] = {}

        for path in leaf_paths:
            values = []
            for r in results:
                walked = _walk(r)
                if path in walked:
                    values.append(walked[path])

            if not values:
                continue

            # Detect type
            all_scalar = all(
                isinstance(v, (float, int)) and not isinstance(v, bool) for v in values
            )
            all_list = all(isinstance(v, list) for v in values)

            if all_scalar:
                arr = np.array(values, dtype=float)
                flat_aggregated[path] = AggregatedLeaf(
                    mean=float(np.mean(arr)),
                    std=float(np.std(arr)),
                    min=float(np.min(arr)),
                    max=float(np.max(arr)),
                    n_seeds=len(values),
                )
            elif all_list:
                lengths = [len(v) for v in values]
                if len(set(lengths)) > 1:
                    warnings.warn(
                        f"Sequence lengths differ for metric '{path}': {lengths}. "
                        f"Truncating to shortest ({min(lengths)}).",
                        MetricShapeMismatchWarning,
                        stacklevel=2,
                    )
                min_len = min(lengths)
                arr = np.array([v[:min_len] for v in values], dtype=float)
                flat_aggregated[path] = AggregatedLeaf(
                    mean=np.mean(arr, axis=0).tolist(),
                    std=np.std(arr, axis=0).tolist(),
                    min=np.min(arr, axis=0).tolist(),
                    max=np.max(arr, axis=0).tolist(),
                    n_seeds=len(values),
                )
            else:
                warnings.warn(
                    f"Metric '{path}' has unsupported type(s) "
                    f"({[type(v).__name__ for v in values]}). Skipping.",
                    UnsupportedMetricTypeWarning,
                    stacklevel=2,
                )

        return _build_nested(flat_aggregated)
