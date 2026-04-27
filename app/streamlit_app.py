"""Streamlit demo for face mask detection.

Run locally from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

# Make src/ importable when Streamlit runs from app/ or from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_MODEL_PATH  # noqa: E402
from src.predict import load_model_artifact, predict_pil_image  # noqa: E402


@st.cache_resource(show_spinner=False)
def cached_load_model(model_path: str):
    return load_model_artifact(model_path)


def probability_table(probabilities: dict[str, float]) -> pd.DataFrame:
    rows = [
        {"Class": label, "Probability": probability}
        for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    ]
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(
        page_title="Face Mask Detection",
        page_icon="😷",
        layout="centered",
    )

    st.title("Face Mask Detection from Face Images")
    st.caption("DSCI 441 · HOG features + PCA + RBF SVM + Streamlit")

    with st.sidebar:
        st.header("Model settings")
        model_path = st.text_input("Saved model path", value=str(DEFAULT_MODEL_PATH.relative_to(PROJECT_ROOT)))
        resolved_model_path = Path(model_path)
        if not resolved_model_path.is_absolute():
            resolved_model_path = PROJECT_ROOT / resolved_model_path
        st.write("Current model file:")
        st.code(str(resolved_model_path), language="text")

        minimum_confidence = st.slider(
            "Low-confidence warning threshold",
            min_value=0.50,
            max_value=0.95,
            value=0.60,
            step=0.01,
            help="If the top class probability is below this value, the app will keep the prediction but mark it as low confidence.",
        )

        st.divider()
        st.write("Need to train first?")
        st.code("python main.py train --data-dir data/raw", language="bash")
        st.write("Then run this app:")
        st.code("streamlit run app/streamlit_app.py", language="bash")

    try:
        artifact = cached_load_model(str(resolved_model_path))
    except Exception as exc:  # noqa: BLE001 - display a useful user-facing message
        st.warning("No trained model is available yet.")
        st.write(
            "Place the Kaggle dataset in `data/raw`, train the model, and make sure "
            "`models/mask_model.joblib` exists before using the demo."
        )
        st.code("python main.py train --data-dir data/raw", language="bash")
        st.info(f"Model loading error: {exc}")
        st.stop()

    model_label = artifact.get("model_name", "saved_model")
    model_image_size = tuple(artifact.get("image_size", ()))
    st.success(f"Loaded model: `{model_label}`")
    if model_image_size:
        st.caption(f"Model preprocessing size: {model_image_size[0]}×{model_image_size[1]}")

    st.markdown(
        "Upload a face image. The app will preprocess it, extract HOG features, "
        "and classify it as **WithMask** or **WithoutMask**."
    )

    uploaded_file = st.file_uploader("Upload a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Upload an image to get a prediction.")
        st.stop()

    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Running prediction..."):
        result = predict_pil_image(image, artifact)

    prediction = result["prediction"]
    confidence = result["confidence"]
    low_confidence = bool(confidence == confidence and confidence < minimum_confidence)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prediction", prediction)
    with col2:
        if confidence == confidence:  # NaN-safe check
            st.metric("Confidence", f"{confidence:.1%}")
        else:
            st.metric("Confidence", "N/A")

    if low_confidence:
        st.warning(
            "Low-confidence prediction: the model made a decision, but the top probability is below "
            f"{minimum_confidence:.0%}. Try a clearer, centered face image if possible."
        )

    probabilities = result.get("probabilities", {})
    if probabilities:
        st.subheader("Class probabilities")
        df = probability_table(probabilities)
        st.dataframe(df, use_container_width=True, hide_index=True)
        for label, probability in probabilities.items():
            st.write(f"{label}: {probability:.1%}")
            st.progress(float(probability))

    st.divider()
    st.caption(
        "Scope note: this demo performs only binary mask/no-mask classification. "
        "It is not identity recognition, person tracking, or a surveillance system."
    )


if __name__ == "__main__":
    main()
