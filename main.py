"""Root command-line entry point for the final mask-detection model.

Examples
--------
Train the final non-deep-learning milestone model:
    python main.py train --data-dir data/raw

Evaluate the saved model:
    python main.py evaluate --data-dir data/raw --model-path models/mask_model.joblib
"""
from __future__ import annotations

import argparse

from src.config import DEFAULT_MODEL_PATH, IMAGE_SIZE, RAW_DATA_DIR
from src.evaluate import evaluate_saved_model
from src.models import MODEL_NAME, list_available_models
from src.train import train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DSCI 441 Mask Detection Project")
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser("train", help="Train and save the final model.")
    train.add_argument("--data-dir", default=str(RAW_DATA_DIR))
    train.add_argument("--model", default=MODEL_NAME, choices=list_available_models())
    train.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    train.add_argument("--image-size", type=int, nargs=2, default=IMAGE_SIZE, metavar=("WIDTH", "HEIGHT"))
    train.add_argument("--max-train-per-class", type=int, default=None)
    train.add_argument("--max-eval-per-class", type=int, default=None)
    train.add_argument("--bootstrap-samples", type=int, default=500)
    train.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs for optional milestone_grid tuning. Ignored by the fixed final model.")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved model on the held-out test set.")
    evaluate.add_argument("--data-dir", default=str(RAW_DATA_DIR))
    evaluate.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    evaluate.add_argument("--max-eval-per-class", type=int, default=None)
    evaluate.add_argument("--bootstrap-samples", type=int, default=500)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
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
    elif args.command == "evaluate":
        evaluate_saved_model(
            model_path=args.model_path,
            data_dir=args.data_dir,
            max_eval_per_class=args.max_eval_per_class,
            n_bootstraps=args.bootstrap_samples,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
