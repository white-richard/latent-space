import atexit
import concurrent.futures
import pathlib
import sys

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


class _Tee:
    def __init__(self, terminal, log_file) -> None:
        self._terminal = terminal
        self._log_file = log_file

    def write(self, data) -> None:
        self._terminal.write(data)
        self._log_file.write(data)
        self._log_file.flush()

    def flush(self) -> None:
        self._terminal.flush()
        self._log_file.flush()


_log_file = None
_log_path = None
_mlflow_enabled = False


def is_enabled() -> bool:
    return _mlflow_enabled


def setup(*, experiment_name, uri: str = "http://172.20.199.236:5050") -> None:
    global _log_file, _log_path, _mlflow_enabled

    _log_path = pathlib.Path("terminal_output.log").resolve()
    _log_file = open(_log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)
    atexit.register(end_run)

    try:
        mlflow.set_tracking_uri(uri)
        _call_with_timeout(mlflow.set_experiment, experiment_name)
        mlflow.enable_system_metrics_logging()
        mlflow.config.set_system_metrics_sampling_interval(1)
        _mlflow_enabled = True
    except Exception as e:
        print(
            f"Warning: MLflow connection failed ({type(e).__name__}: {e}). "
            "Continuing without MLflow tracking.",
            file=sys.stderr,
        )
        mlflow.set_tracking_uri("")
        _mlflow_enabled = False
        return

    try:
        import pynvml  # noqa: F401
    except ImportError:
        import warnings

        warnings.warn(
            "nvidia-ml-py is not installed. GPU metrics will not be logged by MLflow.",
            stacklevel=2,
        )


def end_run() -> None:
    global _log_file, _log_path
    try:
        if _log_file is not None:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            _log_file.close()
            _log_file = None
            if mlflow.active_run():
                content = _log_path.read_text()
                try:
                    mlflow.log_text(content, "terminal_output.log")
                except Exception as e:
                    print(f"Warning: artifact upload failed ({type(e).__name__}: {e})", file=sys.stderr)
                    try:
                        tail = content[-4000:] if len(content) > 4000 else content
                        mlflow.set_tag("terminal_output_tail", tail)
                    except Exception:
                        pass
    finally:
        if mlflow.active_run():
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
