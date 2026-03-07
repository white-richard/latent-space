"""ExperimentSuite — runs a collection of Experiment objects."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .config import RunnerConfig
from .runner import Experiment, ExperimentRunner

if TYPE_CHECKING:
    from .aggregator import AggregatedMetrics

logger = logging.getLogger(__name__)


@dataclass
class SuiteResult:
    experiments: list[str]
    results: dict[str, AggregatedMetrics]
    run_timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentSuite:
    def __init__(
        self,
        experiments: list[Experiment],
        runner_config: RunnerConfig | None = None,
    ) -> None:
        self.experiments = experiments
        self.runner_config = runner_config or RunnerConfig()

    def run(self) -> SuiteResult:
        runner = ExperimentRunner(self.runner_config)
        results: dict[str, AggregatedMetrics] = {}
        timestamp = datetime.now()

        for experiment in self.experiments:
            logger.info("Running experiment: %s", experiment.name)
            results[experiment.name] = runner.run(experiment)

        suite_result = SuiteResult(
            experiments=[e.name for e in self.experiments],
            results=results,
            run_timestamp=timestamp,
        )

        cfg = self.runner_config
        if cfg.output_dir is not None and cfg.export_formats:
            from .exporter import ResultExporter

            ts = timestamp.strftime("%Y%m%d_%H%M%S")
            output_path = cfg.output_dir / ts
            output_path.mkdir(parents=True, exist_ok=True)
            ResultExporter(cfg).export(suite_result, output_path)

        return suite_result
