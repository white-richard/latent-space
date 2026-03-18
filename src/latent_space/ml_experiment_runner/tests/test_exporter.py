"""Tests for ResultExporter."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from latent_space.ml_experiment_runner.runner.aggregator import AggregatedLeaf
from latent_space.ml_experiment_runner.runner.config import RunnerConfig
from latent_space.ml_experiment_runner.runner.exporter import ResultExporter
from latent_space.ml_experiment_runner.runner.suite import SuiteResult

if TYPE_CHECKING:
    from pathlib import Path


def _make_suite_result() -> SuiteResult:
    results = {
        "exp_A": {
            "accuracy": AggregatedLeaf(mean=0.90, std=0.01, min=0.88, max=0.92, n_seeds=3),
            "loss": AggregatedLeaf(mean=0.25, std=0.02, min=0.22, max=0.28, n_seeds=3),
        },
        "exp_B": {
            "accuracy": AggregatedLeaf(mean=0.85, std=0.02, min=0.82, max=0.88, n_seeds=3),
            "loss": AggregatedLeaf(mean=0.30, std=0.03, min=0.27, max=0.34, n_seeds=3),
        },
    }
    return SuiteResult(
        experiments=["exp_A", "exp_B"],
        results=results,
        run_timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )


def _make_suite_result_with_sequences() -> SuiteResult:
    results = {
        "exp_A": {
            "train_loss": AggregatedLeaf(
                mean=[0.5, 0.4, 0.3],
                std=[0.05, 0.04, 0.03],
                min=[0.45, 0.36, 0.27],
                max=[0.55, 0.44, 0.33],
                n_seeds=2,
            ),
        },
        "exp_B": {
            "train_loss": AggregatedLeaf(
                mean=[0.6, 0.5, 0.4],
                std=[0.06, 0.05, 0.04],
                min=[0.54, 0.45, 0.36],
                max=[0.66, 0.55, 0.44],
                n_seeds=2,
            ),
        },
    }
    return SuiteResult(
        experiments=["exp_A", "exp_B"],
        results=results,
        run_timestamp=datetime(2026, 1, 1, 12, 0, 0),
    )


def test_markdown_output(tmp_path: Path) -> None:
    suite_result = _make_suite_result()
    cfg = RunnerConfig(suite_name="test_suite", export_formats=["markdown"])
    exporter = ResultExporter(cfg)
    exporter.export(suite_result, tmp_path)

    md_file = tmp_path / "test_suite.md"
    assert md_file.exists()
    content = md_file.read_text()

    assert "accuracy" in content
    assert "exp_A" in content
    assert "exp_B" in content
    assert "±" in content


def test_markdown_best_value_present(tmp_path: Path) -> None:
    suite_result = _make_suite_result()
    cfg = RunnerConfig(
        suite_name="test_suite",
        export_formats=["markdown"],
        metric_directions={"accuracy": "higher_is_better", "loss": "lower_is_better"},
    )
    exporter = ResultExporter(cfg)
    exporter.export(suite_result, tmp_path)

    content = (tmp_path / "test_suite.md").read_text()
    # exp_A has higher accuracy (0.90 > 0.85) and lower loss (0.25 < 0.30)
    assert "0.9000" in content or "0.90" in content


def test_latex_contains_textbf(tmp_path: Path) -> None:
    suite_result = _make_suite_result()
    cfg = RunnerConfig(
        suite_name="test_suite",
        export_formats=["latex"],
        metric_directions={"accuracy": "higher_is_better"},
    )
    exporter = ResultExporter(cfg)
    exporter.export(suite_result, tmp_path)

    tex_file = tmp_path / "test_suite.tex"
    assert tex_file.exists()
    content = tex_file.read_text()
    assert r"\textbf" in content
    assert r"\toprule" in content
    assert r"\midrule" in content
    assert r"\bottomrule" in content


def test_latex_decimal_places(tmp_path: Path) -> None:
    suite_result = _make_suite_result()
    cfg = RunnerConfig(suite_name="test_suite", export_formats=["latex"], latex_decimal_places=2)
    exporter = ResultExporter(cfg)
    exporter.export(suite_result, tmp_path)

    content = (tmp_path / "test_suite.tex").read_text()
    # 0.90 formatted to 2 decimal places
    assert "0.90" in content


def test_excel_sheet_names(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")

    suite_result = _make_suite_result()
    cfg = RunnerConfig(suite_name="test_suite", export_formats=["excel"])
    exporter = ResultExporter(cfg)
    exporter.export(suite_result, tmp_path)

    xlsx_file = tmp_path / "test_suite.xlsx"
    assert xlsx_file.exists()

    wb = openpyxl.load_workbook(xlsx_file)
    assert "Scalar Metrics" in wb.sheetnames


def test_excel_with_sequences(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")

    suite_result = _make_suite_result_with_sequences()
    cfg = RunnerConfig(suite_name="test_suite", export_formats=["excel"])
    exporter = ResultExporter(cfg)
    exporter.export(suite_result, tmp_path)

    wb = openpyxl.load_workbook(tmp_path / "test_suite.xlsx")
    # Should have a sheet for the sequence metric (name truncated to 31 chars)
    assert any("train_loss" in name for name in wb.sheetnames)
