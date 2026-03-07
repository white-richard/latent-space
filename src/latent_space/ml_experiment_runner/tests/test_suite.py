"""Tests for ExperimentSuite and ExperimentRunner."""

from __future__ import annotations

import dataclasses

import pytest

from latent_space.ml_experiment_runner.runner.adapter import DataclassAdapter
from latent_space.ml_experiment_runner.runner.config import RunnerConfig
from latent_space.ml_experiment_runner.runner.errors import (
    AllSeedsFailedError,
    InsufficientSeedsError,
)
from latent_space.ml_experiment_runner.runner.runner import Experiment, ExperimentRunner
from latent_space.ml_experiment_runner.runner.suite import ExperimentSuite

# ---------------------------------------------------------------------------
# Minimal config / metrics dataclasses
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ExperimentConfig:
    seed: int = 0
    lr: float = 0.01


@dataclasses.dataclass
class Config:
    experiment: ExperimentConfig = dataclasses.field(default_factory=ExperimentConfig)


@dataclasses.dataclass
class Metrics:
    accuracy: float
    loss: float


# ---------------------------------------------------------------------------
# Train functions
# ---------------------------------------------------------------------------


def trivial_train(config: Config) -> Metrics:
    """Returns fixed metrics regardless of config."""
    return Metrics(accuracy=0.9, loss=0.1)


def seed_aware_train(config: Config) -> Metrics:
    """Accuracy depends on seed to verify injection works."""
    return Metrics(accuracy=float(config.experiment.seed) / 100.0, loss=0.1)


def always_failing_train(config: Config) -> Metrics:
    raise RuntimeError("Intentional failure")


def sometimes_failing_train(call_counter: list[int]):
    """Fails on first call, succeeds on subsequent calls."""

    def _train(config: Config) -> Metrics:
        call_counter.append(1)
        if len(call_counter) == 1:
            raise RuntimeError("First call fails")
        return Metrics(accuracy=0.8, loss=0.2)

    return _train


# ---------------------------------------------------------------------------
# ExperimentRunner tests
# ---------------------------------------------------------------------------


def test_runner_basic():
    exp = Experiment(
        name="test",
        config=Config(),
        train_fn=trivial_train,
        seeds=[1, 2, 3],
        adapter=DataclassAdapter("experiment.seed"),
    )
    runner = ExperimentRunner()
    agg = runner.run(exp)

    assert "accuracy" in agg
    assert agg["accuracy"].mean == pytest.approx(0.9)
    assert agg["accuracy"].n_seeds == 3


def test_runner_seed_injection():
    exp = Experiment(
        name="seed_test",
        config=Config(),
        train_fn=seed_aware_train,
        seeds=[10, 20, 30],
        adapter=DataclassAdapter("experiment.seed"),
    )
    agg = ExperimentRunner().run(exp)
    # mean accuracy = (0.10 + 0.20 + 0.30) / 3 = 0.2
    assert agg["accuracy"].mean == pytest.approx(0.2)


def test_runner_all_seeds_fail():
    exp = Experiment(
        name="fail_test",
        config=Config(),
        train_fn=always_failing_train,
        seeds=[1, 2],
        adapter=DataclassAdapter("experiment.seed"),
    )
    with pytest.raises(AllSeedsFailedError):
        ExperimentRunner().run(exp)


def test_runner_insufficient_seeds():
    counter: list[int] = []
    exp = Experiment(
        name="partial_fail",
        config=Config(),
        train_fn=sometimes_failing_train(counter),
        seeds=[1, 2],
        adapter=DataclassAdapter("experiment.seed"),
    )
    cfg = RunnerConfig(min_successful_seeds=2)
    with pytest.raises(InsufficientSeedsError):
        ExperimentRunner(cfg).run(exp)


# ---------------------------------------------------------------------------
# ExperimentSuite tests
# ---------------------------------------------------------------------------


def test_suite_result_keys():
    experiments = [
        Experiment(
            name="exp_A",
            config=Config(),
            train_fn=trivial_train,
            seeds=[1, 2],
            adapter=DataclassAdapter("experiment.seed"),
        ),
        Experiment(
            name="exp_B",
            config=Config(),
            train_fn=trivial_train,
            seeds=[1, 2],
            adapter=DataclassAdapter("experiment.seed"),
        ),
    ]
    suite = ExperimentSuite(experiments)
    result = suite.run()

    assert set(result.results.keys()) == {"exp_A", "exp_B"}
    assert result.experiments == ["exp_A", "exp_B"]


def test_suite_output_markdown(tmp_path):
    experiments = [
        Experiment(
            name="exp_A",
            config=Config(),
            train_fn=trivial_train,
            seeds=[1],
            adapter=DataclassAdapter("experiment.seed"),
        ),
    ]
    cfg = RunnerConfig(
        output_dir=tmp_path,
        export_formats=["markdown"],
        suite_name="my_suite",
    )
    suite = ExperimentSuite(experiments, runner_config=cfg)
    result = suite.run()

    # Should have written a timestamped subdirectory
    subdirs = list(tmp_path.iterdir())
    assert len(subdirs) == 1
    md_file = subdirs[0] / "my_suite.md"
    assert md_file.exists()
