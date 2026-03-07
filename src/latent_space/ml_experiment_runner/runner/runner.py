"""Experiment dataclass and ExperimentRunner."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .aggregator import AggregatedMetrics, MetricsAggregator
from .config import RunnerConfig
from .errors import AllSeedsFailedError, InsufficientSeedsError

if TYPE_CHECKING:
    from .adapter import ConfigAdapter

logger = logging.getLogger(__name__)

TrainFunction = Callable[[Any], Any]


@dataclass
class Experiment:
    name: str
    config: Any
    train_fn: TrainFunction
    seeds: list[int]
    adapter: ConfigAdapter
    tags: dict[str, str] = field(default_factory=dict)


class ExperimentRunner:
    def __init__(self, runner_config: RunnerConfig | None = None) -> None:
        self.runner_config = runner_config or RunnerConfig()

    def run(self, experiment: Experiment) -> AggregatedMetrics:
        results: list[Any] = []

        for seed in experiment.seeds:
            try:
                seeded = experiment.adapter.inject_seed(experiment.config, seed)
                native = experiment.adapter.to_native(seeded)
                result = experiment.train_fn(native)
                results.append(result)
            except Exception:
                logger.exception(
                    "Seed %d failed for experiment '%s'. Skipping.",
                    seed,
                    experiment.name,
                )

        if len(results) == 0:
            raise AllSeedsFailedError(f"All seeds failed for experiment '{experiment.name}'.")

        if len(results) < self.runner_config.min_successful_seeds:
            raise InsufficientSeedsError(
                f"Experiment '{experiment.name}' succeeded on {len(results)} seed(s), "
                f"but min_successful_seeds={self.runner_config.min_successful_seeds}."
            )

        return MetricsAggregator.aggregate(results)
