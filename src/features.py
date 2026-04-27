"""Image preprocessing and HOG feature extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog

from .config import IMAGE_SIZE

# Stronger HOG setting than the initial 64x64/9-orientation version.
# 12 orientations keeps more edge-direction detail around masks and faces.
# 8x8 cells are detailed enough while still more stable than very tiny cells.
HOG_CONFIG = {
    "orientations": 12,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "feature_vector": True,
}


def _resampling_filter():
    """Return a Pillow resampling filter compatible with old/new versions."""
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:  # pragma: no cover - old Pillow fallback
        return Image.LANCZOS


def open_image(path_or_file) -> Image.Image:
    """Open an image, fix EXIF orientation, and convert to RGB."""
    image = Image.open(path_or_file)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def preprocess_image(image: Image.Image, image_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize and convert a PIL image into a normalized grayscale array."""
    image = ImageOps.grayscale(image.convert("RGB"))
    image = image.resize(image_size, _resampling_filter())
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr


def extract_hog_features_from_pil(
    image: Image.Image,
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Extract a one-dimensional HOG feature vector from a PIL image."""
    arr = preprocess_image(image, image_size=image_size)
    features = hog(arr, **HOG_CONFIG)
    return features.astype(np.float32)


def extract_hog_features_from_path(
    image_path: str | Path,
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Extract HOG features from one image path."""
    with open_image(image_path) as image:
        return extract_hog_features_from_pil(image, image_size=image_size)


def extract_feature_matrix(
    image_paths: Iterable[str | Path],
    image_size: Tuple[int, int] = IMAGE_SIZE,
    verbose: bool = True,
) -> np.ndarray:
    """Extract HOG features for multiple image paths.

    This function avoids extra dependencies such as tqdm so that it works in
    simple course-project environments.
    """
    image_paths = list(image_paths)
    features: List[np.ndarray] = []
    failures: List[str] = []

    total = len(image_paths)
    for idx, path in enumerate(image_paths, start=1):
        try:
            features.append(extract_hog_features_from_path(path, image_size=image_size))
        except Exception as exc:  # noqa: BLE001 - helpful for messy image datasets
            failures.append(f"{path}: {exc}")
        if verbose and (idx == total or idx % 500 == 0):
            print(f"  extracted HOG features for {idx}/{total} images")

    if not features:
        detail = "\n".join(failures[:5])
        raise RuntimeError(f"Feature extraction failed for all images. Examples:\n{detail}")

    if failures:
        print(f"Warning: skipped {len(failures)} unreadable image(s).")
        for item in failures[:5]:
            print("  ", item)

    return np.vstack(features)
