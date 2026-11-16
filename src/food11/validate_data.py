
import argparse
import hashlib
import sys
from pathlib import Path

from PIL import Image

SPLITS = ["training", "validation", "evaluation"]
CLASSES = [str(i) for i in range(11)]
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(dataset_path, min_per_class=50, tau=0.05):
    dataset_path = Path(dataset_path)

    problems = {
        "Class folders": [],
        "Image format and size": [],
        "Minimum images per class": [],
        "No duplicates between splits": [],
        "Class distribution": [],
    }

    counts = {}
    seen_hashes = {}

    for split in SPLITS:
        split_path = dataset_path / split

        if not split_path.is_dir():
            problems["Class folders"].append(
                f"Missing split: {split}"
            )
            counts[split] = {}
            continue

        actual_classes = {
            p.name for p in split_path.iterdir()
            if p.is_dir()
        }

        if actual_classes != set(CLASSES):
            problems["Class folders"].append(
                f"{split}: incorrect class folders"
            )

        counts[split] = {}

        for class_name in CLASSES:
            class_path = split_path / class_name

            if not class_path.is_dir():
                counts[split][class_name] = 0
                continue

            images = [
                p for p in class_path.iterdir()
                if p.is_file()
                and p.suffix.lower() in EXTENSIONS
            ]

            counts[split][class_name] = len(images)

            if len(images) < min_per_class:
                problems["Minimum images per class"].append(
                    f"{split}/{class_name}: "
                    f"{len(images)} < {min_per_class}"
                )

            for image_path in images:
                try:
                    with Image.open(image_path) as img:
                        img.load()

                        if img.mode != "RGB":
                            problems["Image format and size"].append(
                                f"{image_path}: mode={img.mode}"
                            )

                        if img.size != (128, 128):
                            problems["Image format and size"].append(
                                f"{image_path}: size={img.size}"
                            )

                except (OSError, ValueError, Image.DecompressionBombError) as exc:
                 problems["Image format and size"].append(
                  f"{image_path}: {exc}"
                )
                    

                # Detect identical files between splits
                file_hash = hashlib.md5(
                    image_path.read_bytes()
                ).hexdigest()

                if file_hash in seen_hashes:
                    old_split, old_path = seen_hashes[file_hash]

                    if old_split != split:
                        problems[
                            "No duplicates between splits"
                        ].append(
                            f"{old_path} and {image_path}"
                        )
                else:
                    seen_hashes[file_hash] = (
                        split,
                        image_path,
                    )

    # Compare training and validation distributions
    train_counts = counts.get("training", {})
    val_counts = counts.get("validation", {})

    train_total = sum(train_counts.values())
    val_total = sum(val_counts.values())

    if train_total == 0 or val_total == 0:
        problems["Class distribution"].append(
            "Empty training or validation split"
        )
    else:
        for class_name in CLASSES:
            train_share = (
                train_counts.get(class_name, 0) / train_total
            )

            val_share = (
                val_counts.get(class_name, 0) / val_total
            )

            difference = abs(train_share - val_share)

            if difference > tau:
                problems["Class distribution"].append(
                    f"Class {class_name}: "
                    f"difference={difference:.4f}"
                )

    # Print validation report
    print("\n===== DATA VALIDATION REPORT =====")

    all_passed = True

    for check_name, errors in problems.items():
        if errors:
            all_passed = False
            print(f"[FAIL] {check_name}")

            for error in errors[:5]:
                print(f"  - {error}")

            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")
        else:
            print(f"[PASS] {check_name}")

    print("==================================")

    if all_passed:
        print("Dataset validation PASSED")
        return 0

    print("Dataset validation FAILED")
    return 1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset_path", type=str)

    parser.add_argument(
        "--min-per-class",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    return validate_dataset(
        args.dataset_path,
        args.min_per_class,
        args.tau,
    )


if __name__ == "__main__":
    sys.exit(main())
