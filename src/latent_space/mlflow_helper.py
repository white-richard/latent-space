import atexit
import concurrent.futures
import pathlib
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


class Logger:
    def __init__(self, filename) -> None:
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


_terminal_log_path = None


def setup(*, experiment_name, uri: str = "http://100.121.43.41:5050") -> None:
    global _terminal_log_path
    mlflow.set_tracking_uri(uri)
    _call_with_timeout(mlflow.set_experiment, experiment_name)
    mlflow.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)

    # --- stdout/stderr capture ---
    log_path = pathlib.Path("terminal_output.log")
    log_path = log_path.resolve()
    logger = Logger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    _terminal_log_path = log_path
    atexit.register(end_run)
    # --- end stdout/stderr capture ---

    try:
        import pynvml
    except ImportError:
        warnings.warn(
            "nvidia-ml-py is not installed. GPU metrics will not be logged by MLflow.",
            stacklevel=2,
        )


def end_run(terminal_log_path: str | None = None) -> None:
    path_to_log = terminal_log_path or _terminal_log_path
    if mlflow.active_run():
        if path_to_log is not None:
            mlflow.log_artifact(path_to_log)
        mlflow.end_run()


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
