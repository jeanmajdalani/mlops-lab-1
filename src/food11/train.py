import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mini", "processed"],
        default="mini",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    return parser.parse_args()


def get_dataset_path(dataset_name):
    if dataset_name == "mini":
        return Path("data/food11_processed_mini")

    return Path("data/food11_processed")


def create_dataloaders(dataset_path, batch_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(
        dataset_path / "training",
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        dataset_path / "validation",
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        dataset_path / "evaluation",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def create_model():
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    number_of_features = model.fc.in_features

    model.fc = nn.Linear(
        number_of_features,
        11,
    )

    return model


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    average_loss = total_loss / len(loader.dataset)

    return average_loss


def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    args = parse_args()

    mlflow.set_tracking_uri(
        "http://127.0.0.1:5000"
    )

    mlflow.set_experiment(
        "food11"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    dataset_path = get_dataset_path(
        args.dataset
    )

    print(f"Dataset: {dataset_path}")

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_path,
        args.batch_size,
    )

    model = create_model().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    with mlflow.start_run():

        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "model": "resnet18",
            }
        )

        for epoch in range(args.epochs):

            train_loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
            )

            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                criterion,
                device,
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Accuracy: {val_accuracy:.4f}"
            )

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_accuracy",
                val_accuracy,
                step=epoch,
            )

        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )

        print(
            f"Final Test Accuracy: "
            f"{test_accuracy:.4f}"
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy,
        )

        mlflow.pytorch.log_model(
    model,
    name="model",
    serialization_format="pickle",
)

if __name__ == "__main__":
    main()