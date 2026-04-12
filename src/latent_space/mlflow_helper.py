import concurrent.futures
import warnings

import mlflow

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


def setup(*, experiment_name, uri: str = "http://100.121.43.41:5050") -> None:
    mlflow.set_tracking_uri(uri)
    _call_with_timeout(mlflow.set_experiment, experiment_name)
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)

    # warn if nvidia-ml-py isn't installed in pip
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
    mlflow.pytorch.log_model(
        model,
        name=name,
        serialization_format="pt2",
        input_example=input_example,
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
