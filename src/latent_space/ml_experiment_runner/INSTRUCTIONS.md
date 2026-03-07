# ML Experiment Runner

A structured harness for multi-seed ML experiments with typed metric aggregation and comparison export (Markdown / LaTeX / Excel).

---

## Quick Start

```python
import dataclasses
from latent_space.ml_experiment_runner import (
    DataclassAdapter, Experiment, ExperimentSuite, RunnerConfig,
)

@dataclasses.dataclass
class Cfg:
    seed: int = 0
    lr: float = 1e-3

@dataclasses.dataclass
class Metrics:
    accuracy: float
    loss: float

def train(cfg: Cfg) -> Metrics:
    ...  # your training loop

suite = ExperimentSuite(
    experiments=[
        Experiment("baseline",    config=Cfg(lr=1e-3), train_fn=train,
                   seeds=[1,2,3], adapter=DataclassAdapter("seed")),
        Experiment("higher_lr",   config=Cfg(lr=1e-2), train_fn=train,
                   seeds=[1,2,3], adapter=DataclassAdapter("seed")),
    ],
    runner_config=RunnerConfig(
        suite_name="lr_sweep",
        export_formats=["markdown", "latex"],
        output_dir=Path("results"),
        metric_directions={"accuracy": "higher_is_better", "loss": "lower_is_better"},
    ),
)
result = suite.run()
# result.results["baseline"]["accuracy"].mean  →  float
```

Output files are written to `output_dir/<YYYYMMDD_HHMMSS>/`.

---

## Directory Layout

```
ml_experiment_runner/
├── runner/
│   ├── adapter.py      # ConfigAdapter Protocol, DataclassAdapter
│   ├── aggregator.py   # MetricsAggregator → AggregatedMetrics / AggregatedLeaf
│   ├── config.py       # RunnerConfig dataclass
│   ├── errors.py       # AllSeedsFailedError, InsufficientSeedsError, warnings
│   ├── exporter.py     # ResultExporter (Markdown / LaTeX / Excel)
│   ├── runner.py       # Experiment dataclass, ExperimentRunner
│   └── suite.py        # ExperimentSuite, SuiteResult
├── tests/
├── examples/
│   ├── dataclass_repo/ # DataclassAdapter with nested dataclass config
│   ├── argparse_repo/  # Custom adapter → argparse.Namespace
│   └── yaml_repo/      # Custom adapter for plain dict / YAML configs
└── INSTRUCTIONS.md
```

---

## Key Concepts

### `Experiment`

```python
@dataclass
class Experiment:
    name: str
    config: Any          # your config object (dataclass, dict, …)
    train_fn: Callable   # takes native config, returns a metrics dataclass
    seeds: list[int]
    adapter: ConfigAdapter
    tags: dict[str, str] = {}
```

`train_fn` must return a **dataclass** whose fields are `float`, `int`, or `list[float]`. Nested dataclasses are supported.

### `ConfigAdapter`

Controls how a seed is injected and how the config is handed to `train_fn`.

```python
class ConfigAdapter(Protocol):
    def inject_seed(self, config, seed: int) -> Any: ...
    def to_native(self, config) -> Any: ...
```

**Built-in:** `DataclassAdapter(seed_field="experiment.seed")` deep-copies the config and sets the seed at a dotted attribute path.

**Custom:** implement the two methods on any class (no base class needed — it's a `runtime_checkable` Protocol).

### `RunnerConfig`

| Field                      | Default              | Description                                            |
| -------------------------- | -------------------- | ------------------------------------------------------ |
| `min_successful_seeds`     | `1`                  | Raise `InsufficientSeedsError` if fewer seeds succeed  |
| `sequence_export_mode`     | `"summary"`          | `"full"` / `"summary"` / `"final"` for list metrics    |
| `metric_directions`        | `{}`                 | Per-metric `"higher_is_better"` or `"lower_is_better"` |
| `default_metric_direction` | `"higher_is_better"` | Fallback direction                                     |
| `latex_decimal_places`     | `4`                  | Decimal places in LaTeX tables                         |
| `markdown_full_sequences`  | `False`              | Force full sequence tables in Markdown                 |
| `output_dir`               | `None`               | Root for timestamped output subdirectories             |
| `export_formats`           | `[]`                 | Any of `"markdown"`, `"latex"`, `"excel"`              |
| `suite_name`               | `"experiment_suite"` | Used as filename stem and table caption                |

### Aggregated metrics

`MetricsAggregator.aggregate(results)` returns a nested dict mirroring your metrics dataclass tree, with `AggregatedLeaf` at every leaf:

```python
@dataclass
class AggregatedLeaf:
    mean: float | list[float]
    std:  float | list[float]
    min:  float | list[float]
    max:  float | list[float]
    n_seeds: int
```

Scalar fields → scalar stats. `list` fields → per-step stats (truncated to shortest with a `MetricShapeMismatchWarning` if lengths differ). Other types emit `UnsupportedMetricTypeWarning` and are skipped.

---

## Error Handling

| Exception / Warning            | When                                          |
| ------------------------------ | --------------------------------------------- |
| `AllSeedsFailedError`          | Every seed raises an exception                |
| `InsufficientSeedsError`       | Successful seeds < `min_successful_seeds`     |
| `MetricShapeMismatchWarning`   | Sequence fields differ in length across seeds |
| `UnsupportedMetricTypeWarning` | A leaf is not `float`, `int`, or `list`       |
| `ExportError`                  | I/O or formatting failure during export       |

Failed seeds are logged at `ERROR` level and skipped; the run continues on remaining seeds.

---

## Export Formats

### Markdown

Scalar table (rows = metrics, cols = experiments, cells = `mean ± std`) plus sequence sections (summary / full / final depending on `sequence_export_mode`). Reuses `latent_space.utils.markdown_results.render_markdown_table`.

### LaTeX

`booktabs`-style (`\toprule / \midrule / \bottomrule`). Best values wrapped in `\textbf{}`. One `table` environment per scalar table and per sequence metric.

### Excel

Requires `pip install latent_space[excel]` (adds `openpyxl`).

- Sheet **"Scalar Metrics"**: same layout as Markdown, best cell highlighted yellow.
- One additional sheet per sequence metric (name truncated to 31 chars): columns are `step`, then `(mean, std)` per experiment.

---

## Running the Tests

```bash
pytest src/latent_space/ml_experiment_runner/tests/
```
