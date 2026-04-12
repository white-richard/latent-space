import concurrent.futures
import sys
import warnings

import mlflow
import numpy as np
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec

_TIMEOUT = 30  # seconds


def _call_with_timeout(fn, *args, timeout=_TIMEOUT, **kwargs) -> any:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            msg = f"MLflow call '{fn.__name__}' timed out after {timeout}s"
            raise TimeoutError(
                msg,
            )


class _TeeStream:
    """Writes to both the original stream and a log file."""

    def __init__(self, original, log_file) -> None:
        self._original = original
        self._file = log_file

    def write(self, msg) -> None:
        self._original.write(msg)
        self._file.write(msg)
        self._file.flush()

    def flush(self) -> None:
        self._original.flush()
        self._file.flush()

    def __getattr__(self, attr):
        return getattr(self._original, attr)


def setup(*, experiment_name, uri: str = "http://100.121.43.41:5050") -> None:
    mlflow.set_tracking_uri(uri)
    _call_with_timeout(mlflow.set_experiment, experiment_name)
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)

    # --- stdout/stderr capture ---
    log_file = open("terminal_output.log", "w", buffering=1)
    sys.stdout = _TeeStream(sys.__stdout__, log_file)
    sys.stderr = _TeeStream(sys.__stderr__, log_file)

    def _flush_and_log() -> None:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        if mlflow.active_run():
            mlflow.log_artifact("terminal_output.log")

    _original_exit = mlflow.ActiveRun.__exit__

    def _patched_exit(self, *args):
        _flush_and_log()
        return _original_exit(self, *args)

    mlflow.ActiveRun.__exit__ = _patched_exit
    # --- end stdout/stderr capture ---

    try:
        import pynvml
    except ImportError:
        warnings.warn(
            "nvidia-ml-py is not installed. GPU metrics will not be logged by MLflow.",
            stacklevel=2,
        )


def test_connection() -> None:
    mlflow.get_tracking_uri()
    _call_with_timeout(mlflow.get_experiment_by_name, "my-first-experiment")


def log_model(model, input_example, name="model") -> None:
    input_example_cpu = input_example.detach().cpu()
    dtype = np.dtype(str(input_example_cpu.dtype).replace("torch.", ""))
    shape = tuple(input_example_cpu.shape)
    signature = ModelSignature(inputs=Schema([TensorSpec(dtype, shape)]))

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = input_example.device

    input_example_device = input_example.detach().to(model_device)
    mlflow.pytorch.log_model(
        model,
        name=name,
        serialization_format="pt2",
        input_example=input_example_device,
        signature=signature,
    )


# # Wrap the training code in a MLflow run
# with mlflow.start_run() as run:

# # Log training parameters
# mlflow.log_params(params)

# mlflow.log_metrics(
#     {"batch_loss": batch_loss, "batch_accuracy": batch_acc},
#     step=epoch * len(train_loader) + batch_idx,
# )
# mlflow.pytorch.log_model(model, name=f"checkpoint_{epoch}")
# # View results
# # mlflow server --port 5000
# # Load the final model
# model = mlflow.pytorch.load_model("runs:/<run_id>/final_model")
# # Resume the previous run to log test metrics
# with mlflow.start_run(run_id=run.info.run_id) as run:
