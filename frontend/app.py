
import os
import requests
import streamlit as st

# ==========================================
# Configuration
# ==========================================

INFERENCE_URL = os.environ.get(
    "INFERENCE_URL",
    "http://127.0.0.1:8000"
)

# Food11 category mapping
# Based on the standard Food-11 dataset labels
FOOD11_CLASSES = {
    "0": "Bread",
    "1": "Dairy product",
    "2": "Dessert",
    "3": "Egg",
    "4": "Fried food",
    "5": "Meat",
    "6": "Noodles/Pasta",
    "7": "Rice",
    "8": "Seafood",
    "9": "Soup",
    "10": "Vegetable/Fruit",
}

# ==========================================
# Streamlit page configuration
# ==========================================

st.set_page_config(
    page_title="Food11 Classifier",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Food11 Classifier")

st.write(
    "Upload a food image and let our AI model "
    "predict its category."
)

st.divider()

# ==========================================
# Image upload
# ==========================================

uploaded = st.file_uploader(
    "Choose a food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:

    st.image(
        uploaded,
        caption="Uploaded image",
        width=350
    )

    # ======================================
    # Prediction
    # ======================================

    if st.button("Predict", type="primary"):

        try:
            files = {
                "file": (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "image/jpeg"
                )
            }

            with st.spinner("Analyzing your image..."):

                response = requests.post(
                    f"{INFERENCE_URL}/predict",
                    files=files,
                    timeout=120
                )

            response.raise_for_status()

            result = response.json()

            # Validate API response
            if not isinstance(result, dict):
                st.error(
                    "Unexpected response format from FastAPI."
                )
                st.json(result)
                st.stop()

            # ==================================
            # Extract predicted category
            # ==================================

            prediction = (
                result.get("predicted_category")
                or result.get("category")
                or result.get("prediction")
                or result.get("predicted_class")
                or result.get("class_name")
                or result.get("label")
            )

            # Convert numeric category to food name
            if prediction is not None:
                prediction = FOOD11_CLASSES.get(
                    str(prediction),
                    str(prediction)
                )

            # ==================================
            # Extract confidence
            # ==================================

            confidence = result.get("confidence")

            if confidence is None:
                confidence = result.get("score")

            if confidence is None:
                confidence = result.get("probability")

            # ==================================
            # Display results
            # ==================================

            st.success("Prediction completed!")

            if prediction is not None:

                st.subheader("Prediction")

                st.write(
                    f"**Category:** {prediction}"
                )

            else:
                st.warning(
                    "No recognized prediction field "
                    "was found in the API response."
                )

            # ==================================
            # Display confidence
            # ==================================

            if confidence is not None:

                try:
                    confidence = float(confidence)

                    if 0 <= confidence <= 1:
                        confidence_percent = confidence * 100
                    else:
                        confidence_percent = confidence

                    st.write(
                        f"**Confidence:** "
                        f"{confidence_percent:.2f}%"
                    )

                    if 0 <= confidence_percent <= 100:

                        st.progress(
                            confidence_percent / 100
                        )

                except (ValueError, TypeError):

                    st.write(
                        f"**Confidence:** {confidence}"
                    )

            # ==================================
            # Display raw API response
            # ==================================

            with st.expander("View API response"):
                st.json(result)

        # ======================================
        # Error handling
        # ======================================

        except requests.exceptions.HTTPError as e:

            st.error(
                f"Inference API error: {e}"
            )

            st.text(response.text)

        except requests.exceptions.RequestException as e:

            st.error(
                f"Cannot connect to inference service: {e}"
            )

        except ValueError:

            st.error(
                "The inference API returned invalid JSON."
            )

st.divider()

st.caption(
    "Food11 Classification | "
    "MLflow + FastAPI + Docker Compose"
)
