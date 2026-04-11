import mlflow


def setup(*, experiment_name, uri: str = "https://100.121.43.41:5050") -> None:
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)


def test_connection() -> None:
    mlflow.get_tracking_uri()
    mlflow.get_experiment_by_name("my-first-experiment")


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
