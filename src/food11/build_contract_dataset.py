
from hashlib import sha256
from pathlib import Path

from PIL import Image

SOURCE = Path("data/food11_processed")
DEST = Path("ci_data/food11_contract_mini")

SPLITS = ["training", "validation", "evaluation"]
CLASSES = [str(i) for i in range(11)]
EXTENSIONS = {".jpg", ".jpeg", ".png"}

PER_CLASS = 50


def main():
    if DEST.exists():
        raise SystemExit(
            f"Destination already exists: {DEST}"
        )

    unique_images = {}
    seen_hashes = set()

    # Collect distinct images after preprocessing
    for class_name in CLASSES:
        unique_images[class_name] = []

        for split in SPLITS:
            folder = SOURCE / split / class_name

            if not folder.is_dir():
                raise SystemExit(f"Missing folder: {folder}")

            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in EXTENSIONS:
                    continue

                with Image.open(path) as img:
                    processed = img.convert("RGB").resize(
                        (128, 128)
                    )
                    digest = sha256(
                        processed.tobytes()
                    ).hexdigest()

                if digest in seen_hashes:
                    continue

                seen_hashes.add(digest)
                unique_images[class_name].append(
                    (split, path)
                )

        count = len(unique_images[class_name])
        required = PER_CLASS * len(SPLITS)

        print(
            f"Class {class_name}: "
            f"{count} unique images"
        )

        if count < required:
            raise SystemExit(
                f"Class {class_name} needs "
                f"{required} unique images, "
                f"but only {count} are available."
            )

    # Build balanced and non-overlapping splits
    for class_name in CLASSES:
        available = unique_images[class_name].copy()

        for split in SPLITS:
            # Prefer files from their original split
            selected = [
                item for item in available
                if item[0] == split
            ][:PER_CLASS]

            # Fill remaining slots with unused images
            if len(selected) < PER_CLASS:
                remaining = [
                    item for item in available
                    if item not in selected
                ]
                selected.extend(
                    remaining[:PER_CLASS - len(selected)]
                )

            output_dir = DEST / split / class_name
            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for i, item in enumerate(selected):
                _, source_path = item

                with Image.open(source_path) as img:
                    processed = img.convert("RGB").resize(
                        (128, 128)
                    )
                    processed.save(
                        output_dir / f"{class_name}_{i:04d}.jpg",
                        format="JPEG",
                        quality=95,
                    )

                available.remove(item)

    print("Contract dataset created successfully!")
    print(f"Location: {DEST}")


if __name__ == "__main__":
    main()
