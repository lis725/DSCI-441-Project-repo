"""Auxiliary data-checking script for the course submission.

Run this after placing the Kaggle dataset in data/raw:
    python aux_1.py --data-dir data/raw
"""
from __future__ import annotations

import argparse

from src.config import RAW_DATA_DIR
from src.data import find_image_records, prepare_splits, summarize_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Check discovered data structure and split sizes.")
    parser.add_argument("--data-dir", default=str(RAW_DATA_DIR))
    args = parser.parse_args()

    records = find_image_records(args.data_dir)
    print("Discovered records")
    print("-" * 60)
    print(summarize_records(records))

    train, validation, test = prepare_splits(records)
    print("\nPrepared split sizes")
    print("-" * 60)
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(validation)}")
    print(f"Test:       {len(test)}")


if __name__ == "__main__":
    main()
