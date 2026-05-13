import atexit
import concurrent.futures
import sys

import mlflow
import numpy as np
from mlflow.models import ModelSignature
from mlflow.types import Schema, TensorSpec

_TIMEOUT = 30  # seconds


def _call_with_timeout(fn, *args, timeout=_TIMEOUT, **kwargs) -> any:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        result = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        msg = f"MLflow call '{fn.__name__}' timed out after {timeout}s"
        raise TimeoutError(msg)
    executor.shutdown(wait=False)
    return result


_mlflow_enabled = False


def is_enabled() -> bool:
    return _mlflow_enabled


def setup(*, experiment_name, uri: str = "http://172.20.199.236:5050") -> None:
    global _mlflow_enabled

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


def safe_log_metric(key: str, value: float, step: int | None = None) -> None:
    if not mlflow.active_run():
        return
    try:
        mlflow.log_metric(key, value, step=step)
    except Exception as e:
        print(f"Warning: MLflow log_metric failed ({type(e).__name__}): {e}", file=sys.stderr)


def safe_log_metrics(metrics: dict, step: int | None = None) -> None:
    if not mlflow.active_run():
        return
    try:
        mlflow.log_metrics(metrics, step=step)
    except Exception as e:
        print(f"Warning: MLflow log_metrics failed ({type(e).__name__}): {e}", file=sys.stderr)


def end_run() -> None:
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
