
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("food11-lab4")

client = MlflowClient()

with mlflow.start_run(run_name="import-food11-lab3") as run:
    print("Run ID:", run.info.run_id)

    mlflow.log_artifacts(
        "/mlflow-data/imported-food11",
        artifact_path="model"
    )

    model_uri = f"runs:/{run.info.run_id}/model"

    print("Model URI:", model_uri)

    registered = mlflow.register_model(
        model_uri=model_uri,
        name="food11"
    )

    client.set_registered_model_alias(
        name="food11",
        alias="champion",
        version=registered.version
    )

    print("Registered version:", registered.version)
    print("Alias: champion")
