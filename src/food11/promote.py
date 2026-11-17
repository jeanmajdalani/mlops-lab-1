
import argparse
import math
import os
from pathlib import Path

from mlflow.tracking import MlflowClient

MODEL_NAME = "food11"

DATASETS = {
    "mini": Path("data/food11_processed_mini"),
    "processed": Path("data/food11_processed"),
}

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_test_size(dataset_name):
    dataset_path = DATASETS[dataset_name] / "evaluation"

    if not dataset_path.is_dir():
        raise ValueError(
            f"Evaluation dataset not found: {dataset_path}"
        )

    n = sum(
        1
        for path in dataset_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() in EXTENSIONS
    )

    if n == 0:
        raise ValueError("Evaluation dataset is empty")

    return n


def get_accuracy(run):
    accuracy = run.data.metrics.get("test_accuracy")

    if accuracy is None:
        raise ValueError(
            f"Missing test_accuracy in run {run.info.run_id}"
        )

    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError("Invalid test_accuracy value")

    return accuracy


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        choices=["mini", "processed"],
        default=None,
    )

    args = parser.parse_args()

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    client = MlflowClient(tracking_uri=tracking_uri)

    # Get candidate and current champion
    candidate = client.get_model_version(
        MODEL_NAME,
        str(args.version),
    )

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion",
    )

    if str(candidate.version) == str(champion.version):
        raise ValueError(
            "Candidate is already the champion"
        )

    # Retrieve the training runs
    candidate_run = client.get_run(candidate.run_id)
    champion_run = client.get_run(champion.run_id)

    candidate_acc = get_accuracy(candidate_run)
    champion_acc = get_accuracy(champion_run)

    # Compare using the same evaluation dataset
    candidate_dataset = candidate_run.data.params.get(
        "dataset"
    )
    champion_dataset = champion_run.data.params.get(
        "dataset"
    )

    if (
        candidate_dataset
        and champion_dataset
        and candidate_dataset != champion_dataset
    ):
        raise ValueError(
            "Candidate and champion used different datasets"
        )

    dataset_name = (
        args.dataset
        or champion_dataset
        or candidate_dataset
    )

    if dataset_name not in DATASETS:
        raise ValueError(
            "Specify --dataset mini or --dataset processed"
        )

    for recorded_dataset in (
        candidate_dataset,
        champion_dataset,
    ):
        if (
            recorded_dataset
            and recorded_dataset != dataset_name
        ):
            raise ValueError(
                "Selected dataset does not match the training run"
            )

    n = get_test_size(dataset_name)

    # Calculate quality gate threshold
    se = math.sqrt(
        champion_acc * (1 - champion_acc) / n
    )

    threshold = champion_acc + 2 * se

    if candidate_acc >= threshold:
        decision = "promoted"

        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            args.version,
        )

    else:
        decision = "challenger"

        client.set_registered_model_alias(
            MODEL_NAME,
            "challenger",
            args.version,
        )

    # Save decision as model version tags
    tags = {
        "quality_gate_decision": decision,
        "quality_gate_candidate_accuracy": str(candidate_acc),
        "quality_gate_champion_accuracy": str(champion_acc),
        "quality_gate_threshold": str(threshold),
        "quality_gate_test_size": str(n),
        "quality_gate_champion_baseline_version": str(
            champion.version
        ),
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            MODEL_NAME,
            str(args.version),
            key,
            value,
        )

    print("\n===== MLFLOW QUALITY GATE =====")
    print(f"Candidate version: {args.version}")
    print(f"Champion version: {champion.version}")
    print(f"Candidate accuracy: {candidate_acc:.4f}")
    print(f"Champion accuracy: {champion_acc:.4f}")
    print(f"Test samples: {n}")
    print(f"Standard error: {se:.6f}")
    print(f"Required accuracy: {threshold:.4f}")
    print(f"Decision: {decision.upper()}")
    print("==============================\n")


if __name__ == "__main__":
    main()
