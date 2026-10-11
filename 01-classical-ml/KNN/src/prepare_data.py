"""Download the MAGIC Gamma Telescope data and create stratified CSV splits."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TARGET = "class"
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-state", type=int, default=42, help="Seed for reproducible splits (default: 42)."
    )
    random_state = parser.parse_args().random_state
    dataset = fetch_ucirepo(id=159)
    features = dataset.data.features
    target = dataset.data.targets[TARGET].map({"g": 1, "h": 0})
    if target.isna().any():
        raise ValueError("Unexpected class values; expected 'g' and 'h'.")

    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.2, random_state=random_state, stratify=target
    )
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x,
        train_y,
        test_size=0.25,
        random_state=random_state,
        stratify=train_y,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, x, y in (
        ("train", train_x, train_y),
        ("valid", valid_x, valid_y),
        ("test", test_x, test_y),
    ):
        split = x.reset_index(drop=True).copy()
        split[TARGET] = y.reset_index(drop=True)
        split.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"Wrote {name}: {len(split):,} rows")


if __name__ == "__main__":
    main()
