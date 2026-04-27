"""Evaluation utilities: metrics, bootstrap intervals, and confusion matrices."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .config import CLASS_ORDER, DEFAULT_MODEL_PATH, IMAGE_SIZE, POSITIVE_LABEL, RANDOM_STATE, RAW_DATA_DIR, REPORTS_DIR
from .data import find_image_records, limit_per_class, prepare_splits
from .features import extract_feature_matrix


def _safe_class_order(y_true: Iterable[str], y_pred: Iterable[str]) -> list[str]:
    seen = list(dict.fromkeys([*CLASS_ORDER, *list(y_true), *list(y_pred)]))
    return [label for label in seen if label in set([*list(y_true), *list(y_pred), *CLASS_ORDER])]


def compute_metrics(y_true, y_pred, positive_label: str = POSITIVE_LABEL) -> Dict[str, float]:
    """Compute standard binary-classification metrics.

    Metrics are calculated manually so bootstrap resamples still work when a
    tiny resample happens to contain only one class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    accuracy = float(np.mean(y_true == y_pred))

    true_pos = y_true == positive_label
    pred_pos = y_pred == positive_label
    tp = int(np.sum(true_pos & pred_pos))
    fp = int(np.sum(~true_pos & pred_pos))
    fn = int(np.sum(true_pos & ~pred_pos))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def bootstrap_confidence_intervals(
    y_true,
    y_pred,
    positive_label: str = POSITIVE_LABEL,
    n_bootstraps: int = 500,
    random_state: int = RANDOM_STATE,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Nonparametric bootstrap confidence intervals for metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    boot = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        m = compute_metrics(y_true[idx], y_pred[idx], positive_label=positive_label)
        for key, value in m.items():
            boot[key].append(value)

    intervals = {}
    lower = 100 * alpha / 2
    upper = 100 * (1 - alpha / 2)
    for key, values in boot.items():
        arr = np.asarray(values)
        intervals[key] = {
            "low": float(np.percentile(arr, lower)),
            "high": float(np.percentile(arr, upper)),
        }
    return intervals


def plot_and_save_confusion_matrix(
    y_true,
    y_pred,
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Save a confusion matrix image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [label for label in CLASS_ORDER if label in set(y_true) or label in set(y_pred)]
    if not labels:
        labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def evaluate_predictions(
    y_true,
    y_pred,
    model_name: str,
    positive_label: str = POSITIVE_LABEL,
    n_bootstraps: int = 500,
    output_dir: str | Path = REPORTS_DIR,
) -> Dict:
    """Compute metrics, CIs, and confusion-matrix figure for predictions."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred, positive_label=positive_label)
    ci = bootstrap_confidence_intervals(
        y_true,
        y_pred,
        positive_label=positive_label,
        n_bootstraps=n_bootstraps,
    )
    cm_path = plot_and_save_confusion_matrix(
        y_true,
        y_pred,
        figures_dir / f"confusion_matrix_{model_name}.png",
        title=f"Confusion Matrix: {model_name}",
    )

    report = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_test": int(len(y_true)),
        "positive_label": positive_label,
        "metrics": metrics,
        "bootstrap_confidence_intervals": ci,
        "confusion_matrix_path": str(cm_path),
    }

    report_path = output_dir / f"metrics_{model_name}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["metrics_json_path"] = str(report_path)
    return report


def print_report(report: Dict) -> None:
    """Pretty-print an evaluation report."""
    print("\nEvaluation report")
    print("-" * 60)
    print(f"Model: {report['model_name']}")
    print(f"Test images: {report['n_test']}")
    for metric, value in report["metrics"].items():
        ci = report["bootstrap_confidence_intervals"].get(metric, {})
        print(f"{metric:>10}: {value:.4f}  95% CI [{ci.get('low', float('nan')):.4f}, {ci.get('high', float('nan')):.4f}]")
    print(f"Confusion matrix: {report['confusion_matrix_path']}")
    print(f"Metrics JSON: {report['metrics_json_path']}")


def _load_artifact(model_path: str | Path) -> Dict:
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    # Backward compatibility if someone saved only the sklearn estimator.
    return {"model": artifact, "model_name": Path(model_path).stem, "image_size": IMAGE_SIZE, "positive_label": POSITIVE_LABEL}


def evaluate_saved_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    data_dir: str | Path = RAW_DATA_DIR,
    max_eval_per_class: Optional[int] = None,
    n_bootstraps: int = 500,
    output_dir: str | Path = REPORTS_DIR,
) -> Dict:
    """Evaluate a saved model on the held-out test set."""
    artifact = _load_artifact(model_path)
    model = artifact["model"]
    model_name = artifact.get("model_name", Path(model_path).stem)
    image_size = tuple(artifact.get("image_size", IMAGE_SIZE))
    positive_label = artifact.get("positive_label", POSITIVE_LABEL)

    records = find_image_records(data_dir)
    _, _, test = prepare_splits(records)
    test = limit_per_class(test, max_eval_per_class)

    print(f"Extracting test features from {len(test)} images...")
    X_test = extract_feature_matrix(test["path"].tolist(), image_size=image_size)
    y_true = test["label"].to_numpy()
    y_pred = model.predict(X_test)

    report = evaluate_predictions(
        y_true,
        y_pred,
        model_name=model_name,
        positive_label=positive_label,
        n_bootstraps=n_bootstraps,
        output_dir=output_dir,
    )
    print_report(report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved mask-detection model.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data-dir", default=str(RAW_DATA_DIR))
    parser.add_argument("--max-eval-per-class", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    evaluate_saved_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        max_eval_per_class=args.max_eval_per_class,
        n_bootstraps=args.bootstrap_samples,
    )


if __name__ == "__main__":
    main()
