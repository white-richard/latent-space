"""Tests for MetricsAggregator."""

from __future__ import annotations

import dataclasses
import warnings

import pytest

from latent_space.ml_experiment_runner.runner.aggregator import (
    AggregatedLeaf,
    MetricsAggregator,
)
from latent_space.ml_experiment_runner.runner.errors import (
    MetricShapeMismatchWarning,
    UnsupportedMetricTypeWarning,
)


@dataclasses.dataclass
class SimpleMetrics:
    accuracy: float
    loss: float


@dataclasses.dataclass
class NestedMetrics:
    val_acc: float
    train_losses: list[float]


@dataclasses.dataclass
class Outer:
    inner: SimpleMetrics
    top_loss: float


def test_scalar_mean_std():
    results = [SimpleMetrics(accuracy=0.8, loss=0.5), SimpleMetrics(accuracy=0.9, loss=0.3)]
    agg = MetricsAggregator.aggregate(results)

    acc_leaf = agg["accuracy"]
    assert isinstance(acc_leaf, AggregatedLeaf)
    assert acc_leaf.mean == pytest.approx(0.85)
    assert acc_leaf.std == pytest.approx(0.05)
    assert acc_leaf.min == pytest.approx(0.8)
    assert acc_leaf.max == pytest.approx(0.9)
    assert acc_leaf.n_seeds == 2

    loss_leaf = agg["loss"]
    assert isinstance(loss_leaf, AggregatedLeaf)
    assert loss_leaf.mean == pytest.approx(0.4)


def test_nested_dataclass():
    results = [
        Outer(inner=SimpleMetrics(accuracy=0.7, loss=0.6), top_loss=1.0),
        Outer(inner=SimpleMetrics(accuracy=0.9, loss=0.4), top_loss=0.8),
    ]
    agg = MetricsAggregator.aggregate(results)

    assert "inner" in agg
    assert isinstance(agg["inner"], dict)
    assert agg["inner"]["accuracy"].mean == pytest.approx(0.8)
    assert agg["top_loss"].mean == pytest.approx(0.9)


def test_sequence_aggregation():
    results = [
        NestedMetrics(val_acc=0.9, train_losses=[0.5, 0.4, 0.3]),
        NestedMetrics(val_acc=0.8, train_losses=[0.6, 0.5, 0.4]),
    ]
    agg = MetricsAggregator.aggregate(results)

    seq_leaf = agg["train_losses"]
    assert isinstance(seq_leaf, AggregatedLeaf)
    assert isinstance(seq_leaf.mean, list)
    assert len(seq_leaf.mean) == 3
    assert seq_leaf.mean[0] == pytest.approx(0.55)
    assert seq_leaf.n_seeds == 2


def test_sequence_length_mismatch_warning():
    results = [
        NestedMetrics(val_acc=0.9, train_losses=[0.5, 0.4, 0.3]),
        NestedMetrics(val_acc=0.8, train_losses=[0.6, 0.5]),
    ]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agg = MetricsAggregator.aggregate(results)

    shape_warnings = [x for x in w if issubclass(x.category, MetricShapeMismatchWarning)]
    assert len(shape_warnings) >= 1

    seq_leaf = agg["train_losses"]
    assert isinstance(seq_leaf.mean, list)
    assert len(seq_leaf.mean) == 2  # truncated to shortest


def test_unsupported_type_skipped():
    @dataclasses.dataclass
    class BadMetrics:
        score: float
        extra: str  # unsupported

    results = [BadMetrics(score=0.5, extra="hello"), BadMetrics(score=0.7, extra="world")]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agg = MetricsAggregator.aggregate(results)

    unsupported = [x for x in w if issubclass(x.category, UnsupportedMetricTypeWarning)]
    assert len(unsupported) >= 1
    assert "score" in agg
    assert "extra" not in agg


def test_empty_results():
    agg = MetricsAggregator.aggregate([])
    assert agg == {}
