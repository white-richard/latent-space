import atexit
import concurrent.futures
import sys

import mlflow

_TIMEOUT = 20  # seconds


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
        client = mlflow.MlflowClient()
        exp = _call_with_timeout(client.get_experiment_by_name, experiment_name)
        if exp is None:
            _call_with_timeout(
                client.create_experiment,
                experiment_name,
                artifact_location="mlflow-artifacts:/",
            )
        mlflow.set_experiment(experiment_name)
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
