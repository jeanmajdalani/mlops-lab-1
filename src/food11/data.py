
from PIL import Image

CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]


def category_from_filename(filename: str) -> str:
    class_id = int(filename.split("_")[0])
    return CLASSES[class_id]


def resize_image(img: Image.Image, size: int = 128) -> Image.Image:
    return img.convert("RGB").resize((size, size))
