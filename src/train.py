"""Training script for the DSCI 441 mask detection project.

The current project version uses one final non-deep-learning model:
HOG features + StandardScaler + PCA + RBF-kernel SVM. Optional GridSearchCV tuning is available with --model milestone_grid.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np

from .config import (
    DEFAULT_MODEL_PATH,
    IMAGE_SIZE,
    POSITIVE_LABEL,
    RANDOM_STATE,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from .data import find_image_records, limit_per_class, prepare_splits, summarize_records
from .evaluate import compute_metrics, evaluate_predictions, print_report
from .features import HOG_CONFIG, extract_feature_matrix
from .models import MODEL_NAME, GRID_MODEL_NAME, FINAL_PCA_COMPONENTS, FINAL_SVM_C, FINAL_SVM_GAMMA, build_model, list_available_models


METRIC_NAMES = ["accuracy", "precision", "recall", "f1"]


def _jsonable(value):
    """Convert numpy/scikit-learn values into JSON-safe Python values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _grid_search_summary(model) -> Dict:
    """Return readable GridSearchCV information if the model was tuned."""
    if not hasattr(model, "best_params_"):
        return {}

    summary = {
        "best_params": _jsonable(model.best_params_),
        "best_cv_score": float(model.best_score_),
    }

    if hasattr(model, "cv_results_"):
        ranks = np.asarray(model.cv_results_["rank_test_score"])
        top_indices = np.argsort(ranks)[:5]
        top_rows = []
        for idx in top_indices:
            row = {
                "rank": int(model.cv_results_["rank_test_score"][idx]),
                "mean_cv_test_f1": float(model.cv_results_["mean_test_score"][idx]),
                "std_cv_test_f1": float(model.cv_results_["std_test_score"][idx]),
                "params": _jsonable(model.cv_results_["params"][idx]),
            }
            if "mean_train_score" in model.cv_results_:
                row["mean_cv_train_f1"] = float(model.cv_results_["mean_train_score"][idx])
            top_rows.append(row)
        summary["top_grid_search_results"] = top_rows

    return summary


def _save_split_metrics_plot(split_metrics: Dict[str, Dict[str, float]], output_dir: str | Path, model_name: str) -> str:
    """Save one grouped bar chart of train/validation/test metrics."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    available_splits = [split for split in ["train", "validation", "test"] if split in split_metrics]
    x = np.arange(len(METRIC_NAMES))
    width = 0.8 / max(len(available_splits), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, split in enumerate(available_splits):
        values = [split_metrics[split][metric] for metric in METRIC_NAMES]
        offset = (i - (len(available_splits) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=split)

    ax.set_ylabel("Score")
    ax.set_title(f"Train / Validation / Test Metrics: {model_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_NAMES)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()

    plot_path = figures_dir / f"split_metrics_{model_name}.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return str(plot_path)


def _diagnose_overfitting(split_metrics: Dict[str, Dict[str, float]]) -> Dict:
    """Create simple generalization-gap diagnostics.

    This is not a formal hypothesis test. It is a practical check for whether the
    model is much better on the training data than on validation/test data.
    """
    train = split_metrics.get("train")
    validation = split_metrics.get("validation")
    test = split_metrics.get("test")

    gaps: Dict[str, Optional[float]] = {}
    for metric in METRIC_NAMES:
        if train and validation:
            gaps[f"train_minus_validation_{metric}"] = float(train[metric] - validation[metric])
        else:
            gaps[f"train_minus_validation_{metric}"] = None
        if train and test:
            gaps[f"train_minus_test_{metric}"] = float(train[metric] - test[metric])
        else:
            gaps[f"train_minus_test_{metric}"] = None
        if validation and test:
            gaps[f"validation_minus_test_{metric}"] = float(validation[metric] - test[metric])
        else:
            gaps[f"validation_minus_test_{metric}"] = None

    main_gap = gaps.get("train_minus_validation_f1")
    if main_gap is None:
        main_gap = gaps.get("train_minus_test_f1")

    if main_gap is None:
        status = "not_enough_splits"
        explanation = "A validation or test split is missing, so overfitting cannot be checked from split gaps."
    elif main_gap >= 0.10:
        status = "likely_overfitting"
        explanation = "Training F1 is at least 0.10 higher than validation/test F1. The model may be fitting training-specific noise."
    elif main_gap >= 0.05:
        status = "possible_mild_overfitting"
        explanation = "Training F1 is 0.05 to 0.10 higher than validation/test F1. Watch the gap, but this may still be acceptable."
    elif main_gap <= -0.03:
        status = "unusual_split_behavior"
        explanation = "Validation/test F1 is higher than training F1. Check whether the split difficulty or sample size differs."
    else:
        status = "no_clear_overfitting"
        explanation = "Training and validation/test F1 scores are close. There is no obvious overfitting signal from these splits."

    return {
        "status": status,
        "explanation": explanation,
        "generalization_gaps": gaps,
        "rule_of_thumb": {
            "main_signal": "Compare train F1 with validation F1 first, and train F1 with test F1 second.",
            "rough_thresholds": {
                "gap_under_0.05": "usually fine",
                "gap_0.05_to_0.10": "possible mild overfitting",
                "gap_over_0.10": "likely overfitting",
            },
        },
    }


def _save_generalization_report(
    *,
    model,
    model_name: str,
    split_metrics: Dict[str, Dict[str, float]],
    output_dir: str | Path,
) -> Dict:
    """Save train/validation/test metrics and overfitting diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = _diagnose_overfitting(split_metrics)
    plot_path = _save_split_metrics_plot(split_metrics, output_dir=output_dir, model_name=model_name)

    report = {
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "split_metrics": split_metrics,
        "overfitting_diagnostics": diagnostics,
        "grid_search_summary": _grid_search_summary(model),
        "split_metrics_plot": plot_path,
    }

    report_path = output_dir / f"generalization_{model_name}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["generalization_json_path"] = str(report_path)
    return report


def _print_generalization_report(report: Dict) -> None:
    """Print a compact train/validation/test diagnostic table."""
    print("\nGeneralization / overfitting diagnostics")
    print("-" * 60)
    for split, metrics in report["split_metrics"].items():
        metric_text = ", ".join(f"{name}={metrics[name]:.4f}" for name in METRIC_NAMES)
        print(f"{split:>10}: {metric_text}")

    diagnostics = report["overfitting_diagnostics"]
    print(f"Status: {diagnostics['status']}")
    print(f"Meaning: {diagnostics['explanation']}")

    grid_summary = report.get("grid_search_summary", {})
    if grid_summary:
        print(f"Best CV F1: {grid_summary.get('best_cv_score', float('nan')):.4f}")
        print(f"Best params: {grid_summary.get('best_params')}")

    print(f"Generalization JSON: {report['generalization_json_path']}")
    print(f"Split metrics plot: {report['split_metrics_plot']}")


def train_model(
    data_dir: str | Path = RAW_DATA_DIR,
    model_name: str = MODEL_NAME,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    image_size: tuple[int, int] = IMAGE_SIZE,
    max_train_per_class: Optional[int] = None,
    max_eval_per_class: Optional[int] = None,
    n_bootstraps: int = 500,
    output_dir: str | Path = REPORTS_DIR,
    random_state: int = RANDOM_STATE,
    n_jobs: int = 1,
) -> Dict:
    """Train the final model, evaluate it, save diagnostics, and save an artifact."""
    data_dir = Path(data_dir)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading images from: {data_dir}")
    records = find_image_records(data_dir)
    print("\nDataset summary by discovered split and label")
    print("-" * 60)
    print(summarize_records(records))

    train, validation, test = prepare_splits(records, random_state=random_state)
    train = limit_per_class(train, max_train_per_class, random_state=random_state)
    validation = limit_per_class(validation, max_eval_per_class, random_state=random_state)
    test = limit_per_class(test, max_eval_per_class, random_state=random_state)

    print("\nFinal split sizes")
    print("-" * 60)
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")

    print(f"\nExtracting training HOG features from {len(train)} images...")
    X_train = extract_feature_matrix(train["path"].tolist(), image_size=image_size)
    y_train = train["label"].to_numpy()

    print(f"\nBuilding model: {model_name}")
    if model_name == GRID_MODEL_NAME:
        print(f"Optional GridSearchCV tuning n_jobs: {n_jobs}")
    else:
        print(
            "Using fixed final hyperparameters: "
            f"PCA n_components={FINAL_PCA_COMPONENTS}, "
            f"SVM C={FINAL_SVM_C}, gamma={FINAL_SVM_GAMMA!r}."
        )
    model = build_model(model_name=model_name, random_state=random_state, n_jobs=n_jobs)
    model.fit(X_train, y_train)

    print("\nPredicting training split for overfitting diagnostics...")
    y_train_pred = model.predict(X_train)
    split_metrics: Dict[str, Dict[str, float]] = {
        "train": compute_metrics(y_train, y_train_pred, positive_label=POSITIVE_LABEL),
    }

    if len(validation) > 0:
        print(f"\nExtracting validation HOG features from {len(validation)} images...")
        X_validation = extract_feature_matrix(validation["path"].tolist(), image_size=image_size)
        y_validation = validation["label"].to_numpy()
        y_validation_pred = model.predict(X_validation)
        split_metrics["validation"] = compute_metrics(y_validation, y_validation_pred, positive_label=POSITIVE_LABEL)

    print(f"\nExtracting test HOG features from {len(test)} images...")
    X_test = extract_feature_matrix(test["path"].tolist(), image_size=image_size)
    y_test = test["label"].to_numpy()
    y_test_pred = model.predict(X_test)
    split_metrics["test"] = compute_metrics(y_test, y_test_pred, positive_label=POSITIVE_LABEL)

    report = evaluate_predictions(
        y_test,
        y_test_pred,
        model_name=model_name,
        positive_label=POSITIVE_LABEL,
        n_bootstraps=n_bootstraps,
        output_dir=output_dir,
    )
    print_report(report)

    generalization_report = _save_generalization_report(
        model=model,
        model_name=model_name,
        split_metrics=split_metrics,
        output_dir=output_dir,
    )
    _print_generalization_report(generalization_report)

    artifact = {
        "model": model,
        "model_name": model_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_size": image_size,
        "final_fixed_hyperparameters": {
            "pca__n_components": FINAL_PCA_COMPONENTS,
            "clf__C": FINAL_SVM_C,
            "clf__gamma": FINAL_SVM_GAMMA,
        },
        "hog_config": HOG_CONFIG,
        "class_names": sorted(set(y_train)),
        "positive_label": POSITIVE_LABEL,
        "training_summary": {
            "data_dir": str(data_dir),
            "n_train": int(len(train)),
            "n_validation": int(len(validation)),
            "n_test": int(len(test)),
            "max_train_per_class": max_train_per_class,
            "max_eval_per_class": max_eval_per_class,
            "random_state": random_state,
        },
        "test_report": report,
        "generalization_report": generalization_report,
    }
    joblib.dump(artifact, model_path)
    print(f"\nSaved trained model to: {model_path}")

    metadata_path = output_dir / f"model_metadata_{model_name}.json"
    metadata = {k: v for k, v in artifact.items() if k != "model"}
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {metadata_path}")
    return artifact


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the final mask-detection model.")
    parser.add_argument("--data-dir", default=str(RAW_DATA_DIR), help="Path to extracted dataset folder.")
    parser.add_argument("--model", default=MODEL_NAME, choices=list_available_models(), help="Model to train.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH), help="Where to save the model artifact.")
    parser.add_argument("--image-size", type=int, nargs=2, default=IMAGE_SIZE, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--max-train-per-class", type=int, default=None, help="Optional cap per class for faster experiments.")
    parser.add_argument("--max-eval-per-class", type=int, default=None, help="Optional cap per class for quick validation/test diagnostics.")
    parser.add_argument("--bootstrap-samples", type=int, default=500, help="Number of bootstrap samples for confidence intervals.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for optional milestone_grid tuning. Ignored by the fixed final model.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    train_model(
        data_dir=args.data_dir,
        model_name=args.model,
        model_path=args.model_path,
        image_size=tuple(args.image_size),
        max_train_per_class=args.max_train_per_class,
        max_eval_per_class=args.max_eval_per_class,
        n_bootstraps=args.bootstrap_samples,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
