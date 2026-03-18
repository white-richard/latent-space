"""RunnerConfig dataclass — all settings for ExperimentRunner / ExperimentSuite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class RunnerConfig:
    min_successful_seeds: int = 1
    sequence_export_mode: Literal["full", "summary", "final"] = "summary"
    metric_directions: dict[str, Literal["higher_is_better", "lower_is_better"]] = field(
        default_factory=dict,
    )
    default_metric_direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"
    latex_decimal_places: int = 4
    markdown_full_sequences: bool = False
    output_dir: Path | None = None
    export_formats: list[Literal["excel", "markdown", "latex"]] = field(default_factory=list)
    suite_name: str = "experiment_suite"
