from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from latent_space.utils.markdown_results import (
    MarkdownTableLogger,
    flatten_dict,
    render_markdown_table,
    to_cell,
    to_plain_dict,
)

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for Obsidian-compatible experiment reports."""

    project_name: str = "Project"
    report_filename: str = "experiment.md"
    config_filename: str = "config.md"
    dataset: str | None = None
    tags: list[str] = field(default_factory=list)
    hypothesis: str = ""
    date_format: str = "%Y-%m-%d"
    metrics_link: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    plots: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    conclusion: str | None = None
    status_tags: dict[str, str] = field(
        default_factory=lambda: {
            "training-complete": "#training-complete",
            "failed": "#failed",
            "evaluating": "#evaluating",
        }
    )


@dataclass(frozen=True)
class VariantSummary:
    label: str
    config_path: Path
    results_path: Path
    output_dir: Path
    variant_slug: str
    result: Any
    error: Exception | None = None


_DEFAULT_REPORT_CONFIG: ReportConfig | None = None


def set_default_report_config(config: ReportConfig | None) -> None:
    """Set a module-level default report configuration."""
    global _DEFAULT_REPORT_CONFIG
    _DEFAULT_REPORT_CONFIG = config


def get_default_report_config() -> ReportConfig:
    """Return the module-level default report configuration."""
    return _DEFAULT_REPORT_CONFIG or ReportConfig()


def load_report_config(path: Path) -> ReportConfig:
    """Load a report configuration from a JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ReportConfig(
        project_name=data.get("project_name", ReportConfig().project_name),
        report_filename=data.get("report_filename", ReportConfig().report_filename),
        config_filename=data.get("config_filename", ReportConfig().config_filename),
        dataset=data.get("dataset"),
        tags=list(data.get("tags", [])),
        hypothesis=data.get("hypothesis", ReportConfig().hypothesis),
        date_format=data.get("date_format", ReportConfig().date_format),
        metrics_link=data.get("metrics_link"),
        parameters=dict(data.get("parameters", {})),
        plots=list(data.get("plots", [])),
        observations=list(data.get("observations", [])),
        conclusion=data.get("conclusion"),
        status_tags=dict(data.get("status_tags", ReportConfig().status_tags)),
    )


def _get_nested_value(obj: Any, dotted_path: str) -> Any:
    current = obj
    for part in dotted_path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
        else:
            if not hasattr(current, part):
                return None
            current = getattr(current, part)
    return current


def _first_present(obj: Any, paths: Sequence[str]) -> Any:
    for path in paths:
        value = _get_nested_value(obj, path)
        if value is not None:
            return value
    return None


def resolve_run_id(config: Any) -> str:
    run_id = _first_present(
        config,
        [
            "experiment.run_id",
            "experiment.experiment_name",
            "run_id",
            "experiment_name",
            "model.model_name",
            "model.name",
        ],
    )
    return str(run_id) if run_id is not None else "run"


def resolve_project_name(config: Any, report_config: ReportConfig) -> str:
    if (
        report_config.project_name
        and report_config.project_name != ReportConfig().project_name
    ):
        return report_config.project_name
    project_name = _first_present(config, ["experiment.project_name", "project_name"])
    return str(project_name) if project_name is not None else report_config.project_name


def resolve_dataset(config: Any, report_config: ReportConfig) -> str | None:
    if report_config.dataset:
        return report_config.dataset
    dataset = _first_present(config, ["data.dataset", "dataset", "data.name"])
    return str(dataset) if dataset is not None else None


def resolve_tags(config: Any, report_config: ReportConfig) -> list[str]:
    tags = list(report_config.tags)
    extra = _first_present(config, ["experiment.tags", "tags"])
    if isinstance(extra, (list, tuple)):
        tags.extend([str(t) for t in extra])
    return tags


def resolve_parameters(config: Any, report_config: ReportConfig) -> dict[str, Any]:
    parameters = dict(report_config.parameters)
    model_name = _first_present(
        config, ["model.model_name", "model.name", "model.arch"]
    )
    learning_rate = _first_present(config, ["training.lr", "optimizer.lr", "lr"])
    batch_size = _first_present(config, ["data.batch_size", "batch_size"])
    hardware = _first_present(config, ["experiment.hardware", "hardware"])

    parameters.setdefault("Model", model_name or "Unknown")
    if learning_rate is not None:
        parameters.setdefault("Learning Rate", learning_rate)
    else:
        parameters.setdefault("Learning Rate", "Unknown")
    if batch_size is not None:
        parameters.setdefault("Batch Size", batch_size)
    else:
        parameters.setdefault("Batch Size", "Unknown")
    parameters.setdefault("Hardware", hardware or "Unknown")
    return parameters


def _extract_row_label_payload(
    result: Any,
) -> tuple[Sequence[Any], str, dict[str, Any]] | None:
    if not isinstance(result, dict):
        return None
    row_labels = result.get("__row_labels")
    if row_labels is None:
        return None
    row_label_header = result.get("__row_label_header", "name")
    columns = {
        k: v
        for k, v in result.items()
        if not (isinstance(k, str) and k.startswith("__row_label"))
    }
    return row_labels, row_label_header, columns


def normalize_metrics(result: Any) -> Any:
    if isinstance(result, Mapping):
        if "__row_labels" in result:
            return result
        if "metrics" in result:
            metrics_value = result.get("metrics")
            return (
                metrics_value
                if isinstance(metrics_value, Mapping)
                else {"metrics": metrics_value}
            )
        return result
    return {"metrics": result}


def metrics_to_markdown_table(metrics: Any, *, float_precision: int = 6) -> str:
    if metrics is None:
        return ""
    row_payload = _extract_row_label_payload(metrics)
    if row_payload is not None:
        row_labels, row_label_header, columns = row_payload
        rows: list[dict[str, Any]] = []
        for idx in range(len(row_labels)):
            row: dict[str, Any] = {row_label_header: row_labels[idx]}
            for key, seq in columns.items():
                row[key] = seq[idx]
            rows.append(row)
        headers = [row_label_header, *columns.keys()]
        return render_markdown_table(
            headers, rows, float_precision=float_precision
        ).strip()

    if isinstance(metrics, Mapping):
        flattened = flatten_dict(metrics)
        return render_markdown_table(
            flattened.keys(), [flattened], float_precision=float_precision
        ).strip()

    return render_markdown_table(
        ["metrics"], [{"metrics": metrics}], float_precision=float_precision
    ).strip()


def log_metrics_table(path: Path, metrics: Any, *, float_precision: int = 6) -> None:
    logger = MarkdownTableLogger(path, float_precision=float_precision)
    normalized = normalize_metrics(metrics)
    row_payload = _extract_row_label_payload(normalized)
    if row_payload:
        row_labels, row_label_header, columns = row_payload
        logger.append_columns(
            columns, row_labels=row_labels, row_label_header=row_label_header
        )
    else:
        logger.append(
            normalized if isinstance(normalized, Mapping) else {"metrics": normalized}
        )


def resolve_plot_paths(result: Any, report_config: ReportConfig) -> list[str]:
    plots = list(report_config.plots)
    if isinstance(result, Mapping):
        for key in ("plots", "plot_paths", "plot_files"):
            value = result.get(key)
            if isinstance(value, (list, tuple)):
                plots.extend([str(v) for v in value])
    return [p for p in plots if p]


def resolve_status_tag(
    result: Any, *, error: Exception | None, report_config: ReportConfig
) -> str:
    if error is not None:
        return report_config.status_tags.get("failed", "#failed")
    if isinstance(result, Mapping):
        status = result.get("status") or result.get("state")
        if isinstance(status, str):
            status_key = status.strip().lower()
            if "eval" in status_key:
                return report_config.status_tags.get("evaluating", "#evaluating")
    return report_config.status_tags.get("training-complete", "#training-complete")


class ObsidianReportWriter:
    """Generate Obsidian-compatible experiment reports."""

    def __init__(self, config: ReportConfig | None = None) -> None:
        self.config = config or get_default_report_config()

    def build_report(
        self,
        *,
        config: Any,
        result: Any,
        status_error: Exception | None = None,
        config_link: str | None = None,
    ) -> str:
        run_id = resolve_run_id(config)
        project_name = resolve_project_name(config, self.config)
        status_tag = resolve_status_tag(
            result, error=status_error, report_config=self.config
        )
        date_str = datetime.now().strftime(self.config.date_format)
        hypothesis = self.config.hypothesis or (
            '*What change are we testing? (e.g., "Adding Gradient '
            'Checkpointing will reduce VRAM usage by 60% without affecting convergence.")*'
        )

        parameters = resolve_parameters(config, self.config)
        plots = resolve_plot_paths(result, self.config)
        metrics_link = self.config.metrics_link
        metrics_block = metrics_to_markdown_table(normalize_metrics(result))

        lines: list[str] = [
            f"# Experiment: {run_id}",
            f"**Project:** [[{project_name}]]",
            f"**Date:** {date_str}",
            f"status: {status_tag}",
            "",
            "## Hypothesis",
            hypothesis,
        ]

        # lines.extend([f"- **{key}:** {value}" for key, value in parameters.items()])
        # if config_link:
        #     lines.append(f"- **Full Config:** {config_link}")

        lines.extend(["", "## Results & Plots"])

        for plot in plots:
            lines.append(f"![[{plot}]]")

        if metrics_link:
            lines.append(f"- **Final Metrics:** {metrics_link}")
        else:
            lines.append("- **Final Metrics:**")
            lines.append(metrics_block or "_No metrics logged yet._")

        lines.extend(["", "## Observations & Conclusions"])

        observations = self.config.observations or [
            "[Note on convergence speed]",
            "[Unforeseen issues]",
        ]
        for obs in observations:
            lines.append(f"- {obs}")

        conclusion = self.config.conclusion or "[Was the hypothesis proven?]"
        lines.append(f"- **Conclusion:** {conclusion}")
        lines.append("")

        return "\n".join(lines)

    def write_report(
        self,
        output_dir: Path,
        *,
        config: Any,
        result: Any,
        status_error: Exception | None = None,
    ) -> Path:
        report_path = Path(output_dir) / self.config.report_filename
        config_path = write_config_report(
            output_dir,
            config,
            filename=self.config.config_filename,
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            self.build_report(
                config=config,
                result=result,
                status_error=status_error,
                config_link=f"![[{config_path.name}]]",
            ),
            encoding="utf-8",
        )
        return report_path


def write_config_report(
    output_dir: Path, config: Any, *, filename: str = "config.md"
) -> Path:
    """Write a markdown table of flattened config hyperparameters."""
    config_path = Path(output_dir) / filename
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = to_plain_dict(config)
    flattened = flatten_dict(config_dict)
    rows = [{"parameter": key, "value": value} for key, value in flattened.items()]
    table = render_markdown_table(["parameter", "value"], rows).strip()
    lines = ["# Config", "", table or "_No config values logged._", ""]
    config_path.write_text("\n".join(lines), encoding="utf-8")
    return config_path


def build_variant_section(
    *,
    summary: VariantSummary,
    report_config: ReportConfig,
    experiment_root: Path,
    project_name: str,
    experiment_name: str,
) -> list[str]:
    status_tag = resolve_status_tag(
        summary.result, error=summary.error, report_config=report_config
    )
    status_value = status_tag.lstrip("#")
    variants_dir = experiment_root / "variants"
    config_path = summary.config_path
    try:
        config_link = config_path.relative_to(variants_dir)
    except ValueError:
        config_link = config_path
    config_link = config_link.as_posix()
    date_str = datetime.now().strftime(report_config.date_format)

    lines = [
        "---",
        "type: experiment_variant",
        f'project: "[[{project_name}]]"',
        f"experiment: {experiment_name}",
        f"variant: {summary.variant_slug}",
        f"date: {date_str}",
        f"status: {status_value}",
        "---",
        "",
        f"# {experiment_name} — {summary.label}",
        "",
        f"config:: [config]({config_link})",
    ]

    metrics_payload = normalize_metrics(summary.result)
    if isinstance(metrics_payload, Mapping):
        flat = flatten_dict(metrics_payload)
        for key, value in flat.items():
            cleaned_key = key.removeprefix("metrics.")
            cleaned_key = cleaned_key.replace("/", "_")
            lines.append(f"{cleaned_key}:: {to_cell(value)}")
    else:
        lines.append(f"metrics:: {to_cell(metrics_payload)}")

    plots = []
    plots_dir = summary.output_dir / "plots"
    if plots_dir.exists():
        plots = sorted(p for p in plots_dir.rglob("*.png"))
    if plots:
        lines.append("")
        lines.append("### Plots")
        for plot in plots:
            try:
                rel_plot = plot.relative_to(experiment_root)
            except ValueError:
                rel_plot = plot
            lines.append(f"![]({rel_plot.as_posix()})")

    return lines


def write_experiment_report(
    experiment_root: Path,
    *,
    title: str,
    project_name: str,
    report_config: ReportConfig,
    variants: list[VariantSummary],
    base_config: Any,
    git_sha: str | None = None,
) -> Path:
    """Write a single experiment summary report with all variants included."""
    start_str = datetime.now().isoformat(timespec="seconds")
    dataset = resolve_dataset(base_config, report_config)
    tags = resolve_tags(base_config, report_config)
    hypothesis = report_config.hypothesis or (
        '*What change are we testing? (e.g., "Adding Gradient '
        'Checkpointing will reduce VRAM usage by 60% without affecting convergence.")*'
    )
    observations = report_config.observations or [
        "convergence::",
        "issues::",
    ]
    conclusion = report_config.conclusion or "conclusion::"

    tags_text = f"[{', '.join(tags)}]" if tags else "[]"
    frontmatter = [
        "---",
        "type: experiment",
        f'project: "[[{project_name}]]"',
    ]
    if dataset:
        frontmatter.append(f"dataset: {dataset}")
    frontmatter.extend(
        [
            f"experiment: {title}",
            f"start: {start_str}",
            f"git_sha: {git_sha or 'unknown'}",
            f"tags: {tags_text}",
            "---",
            "",
        ]
    )

    lines: list[str] = frontmatter + [
        f"# Experiment: {title}",
        "",
        "## Hypothesis",
        hypothesis,
        "",
        "## Parameters & Setup",
        "- See variant config files below.",
        "",
        "## Variants",
        "",
    ]

    variants_dir = experiment_root / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    for summary in variants:
        variant_path = variants_dir / f"{summary.variant_slug}.md"
        variant_lines = build_variant_section(
            summary=summary,
            report_config=report_config,
            experiment_root=experiment_root,
            project_name=project_name,
            experiment_name=title,
        )
        variant_path.write_text(
            "\n".join(variant_lines).strip() + "\n", encoding="utf-8"
        )
        lines.append(f"- [[variants/{summary.variant_slug}]]")

    lines.extend(["", "## Observations & Conclusions"])
    for obs in observations:
        obs_line = obs if obs.strip().startswith("-") else f"- {obs}"
        lines.append(obs_line)
    conclusion_line = (
        conclusion if conclusion.strip().startswith("-") else f"- {conclusion}"
    )
    lines.append(conclusion_line)

    report_path = Path(experiment_root) / "experiment.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return report_path


def write_variant_summary(
    experiment_root: Path, records: list[tuple[str, Path]]
) -> None:
    """Deprecated: use write_experiment_report instead."""
    _ = experiment_root, records
