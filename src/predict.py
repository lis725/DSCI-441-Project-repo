"""Prediction helpers used by the Streamlit application."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from PIL import Image

from .config import DEFAULT_MODEL_PATH, IMAGE_SIZE, POSITIVE_LABEL
from .features import extract_hog_features_from_pil, open_image


def load_model_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> Dict:
    """Load a saved joblib model artifact."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {"model": artifact, "model_name": model_path.stem, "image_size": IMAGE_SIZE, "positive_label": POSITIVE_LABEL}


def _probabilities_from_model(model, X: np.ndarray) -> Dict[str, float]:
    """Return class probabilities or probability-like scores."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(model.classes_)
        return {str(cls): float(p) for cls, p in zip(classes, proba)}

    # Fallback for estimators without predict_proba.
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        if np.ndim(decision) == 1:
            p_pos = 1.0 / (1.0 + np.exp(-decision[0]))
            classes = list(getattr(model, "classes_", ["WithoutMask", "WithMask"]))
            if len(classes) == 2:
                return {str(classes[0]): float(1 - p_pos), str(classes[1]): float(p_pos)}
    return {}


def predict_pil_image(image: Image.Image, artifact: Dict) -> Dict:
    """Predict mask/no-mask for a PIL image using a saved artifact."""
    model = artifact["model"]
    image_size = tuple(artifact.get("image_size", IMAGE_SIZE))
    X = extract_hog_features_from_pil(image, image_size=image_size).reshape(1, -1)
    prediction = str(model.predict(X)[0])
    probabilities = _probabilities_from_model(model, X)
    confidence = probabilities.get(prediction, None)
    if confidence is None and probabilities:
        confidence = max(probabilities.values())
    elif confidence is None:
        confidence = float("nan")

    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "probabilities": probabilities,
        "model_name": artifact.get("model_name", "saved_model"),
    }


def predict_image_path(image_path: str | Path, artifact: Dict) -> Dict:
    """Predict from a file path."""
    with open_image(image_path) as image:
        return predict_pil_image(image, artifact)
