from pathlib import Path
import shutil
import random


RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

SPLITS = ["training", "validation", "evaluation"]


def get_class_name(filename: str) -> str:
    return filename.split("_")[0]


def build_processed_dataset():
    print("Creating full processed dataset...")

    for split in SPLITS:
        source_dir = RAW_DIR / split
        destination_split = PROCESSED_DIR / split

        destination_split.mkdir(parents=True, exist_ok=True)

        for image_path in source_dir.glob("*.jpg"):
            class_name = get_class_name(image_path.name)

            class_dir = destination_split / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy2(
                image_path,
                class_dir / image_path.name
            )

    print("Full processed dataset created.")


def build_mini_dataset(images_per_class=50):
    print("Creating mini dataset...")

    random.seed(42)

    for split in SPLITS:
        source_split = PROCESSED_DIR / split
        destination_split = MINI_DIR / split

        for class_dir in source_split.iterdir():
            if not class_dir.is_dir():
                continue

            destination_class = destination_split / class_dir.name
            destination_class.mkdir(parents=True, exist_ok=True)

            images = list(class_dir.glob("*.jpg"))

            selected_images = random.sample(
                images,
                min(images_per_class, len(images))
            )

            for image_path in selected_images:
                shutil.copy2(
                    image_path,
                    destination_class / image_path.name
                )

    print("Mini dataset created.")


def main():
    build_processed_dataset()
    build_mini_dataset(images_per_class=50)

    print("Done.")


if __name__ == "__main__":
    main()