
import os
from contextlib import asynccontextmanager
from io import BytesIO

import mlflow
import numpy as np
import torch

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms


# =====================================
# 1. MLFLOW CONFIGURATION
# =====================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/food11@champion",
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# =====================================
# 2. FOOD11 CLASS NAMES
# =====================================

# Same alphabetical order as ImageFolder
# in Lab 2

CLASS_NAMES = [
    "0",
    "1",
    "10",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


# =====================================
# 3. IMAGE PREPROCESSING
# =====================================

# Same preprocessing used during
# validation and testing in Lab 2

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# =====================================
# 4. LOAD MODEL AT STARTUP
# =====================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading Food11 model from MLflow...")

    app.state.model = mlflow.pyfunc.load_model(
        MODEL_URI
    )

    print("Model loaded successfully!")

    yield


# =====================================
# 5. CREATE FASTAPI APPLICATION
# =====================================

app = FastAPI(
    title="Food11 Prediction API",
    description="Food classification using ResNet18 and MLflow",
    version="1.0.0",
    lifespan=lifespan,
)


# =====================================
# 6. HEALTH ENDPOINT
# =====================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =====================================
# 7. PREDICTION ENDPOINT
# =====================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Validate uploaded file type
    if file.content_type not in [
        "image/jpeg",
        "image/png",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are allowed",
        )

    # Read and convert uploaded image
    try:

        contents = await file.read()

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )

    # Apply same preprocessing as Lab 2
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Convert tensor to NumPy array
    model_input = image_tensor.numpy()

    # Predict using MLflow model
    outputs = app.state.model.predict(
        model_input
    )

    # Convert output to tensor
    logits = torch.as_tensor(
        np.asarray(outputs)
    )

    # Convert logits into probabilities
    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    # Get predicted class and confidence
    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    class_index = int(
        predicted_index.item()
    )

    predicted_category = CLASS_NAMES[
        class_index
    ]

    confidence_score = float(
        confidence.item()
    )

    # Return prediction result
    return {
        "predicted_category": predicted_category,
        "predicted_class_index": class_index,
        "confidence": confidence_score,
    }
