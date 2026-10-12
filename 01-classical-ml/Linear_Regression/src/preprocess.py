"""Load, combine, and clean the Mexico City real-estate CSV files.

The cleaning steps mirror the preprocessing cells in ``notebooks/real-estate.ipynb``.
The notebook's later 10th/90th percentile filtering is exploratory/modeling
selection and is deliberately not part of this reusable cleaning function.
"""

from pathlib import Path
from typing import Iterable
import argparse

import pandas as pd


SOURCE_FILES = tuple(
    f"mexico-city-real-estate-{number}.csv" for number in range(1, 6)
)
CLEAN_FILE = "mexico-real-estate-combined-clean.csv"
DROP_COLUMNS = (
    "operation",
    "currency",
    "price",
    "price_aprox_local_currency",
    "price_usd_per_m2",
    "floor",
    "rooms",
    "expenses",
    "properati_url",
    "price_per_m2",
    "surface_total_in_m2",
)


def _read_matching_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    """Read CSVs and concatenate them after validating their column schemas."""
    frames = []
    expected_columns = None

    for path in paths:
        frame = pd.read_csv(path)
        if expected_columns is None:
            expected_columns = list(frame.columns)
        elif set(frame.columns) != set(expected_columns):
            missing = sorted(set(expected_columns) - set(frame.columns))
            extra = sorted(set(frame.columns) - set(expected_columns))
            raise ValueError(
                f"CSV schema mismatch in {path.name}; missing columns: {missing}; "
                f"unexpected columns: {extra}"
            )
        # Align ordering too; equal sets in a different order should not create
        # misaligned values when concatenating.
        frames.append(frame.reindex(columns=expected_columns))

    if not frames:
        raise ValueError("No input CSV files were provided")
    return pd.concat(frames, ignore_index=True)


def load_and_clean_data(data_dir: str | Path | None = None,
                        output_path: str | Path | None = CLEAN_FILE) -> pd.DataFrame:
    """Combine the five source CSVs and apply the notebook's cleaning steps.

    Args:
        data_dir: Directory containing the five numbered source CSV files.
            Defaults to this project's ``data/`` directory.
        output_path: Where to save the cleaned combined CSV. Defaults to
            ``data/mexico-real-estate-combined-clean.csv``. Pass ``None`` to
            return the cleaned frame without writing a file. Relative paths
            are resolved inside ``data_dir``.

    Returns:
        The cleaned dataframe, with ``state``, ``lat``, and ``lon`` derived
        from the original location fields.
    """
    project_root = Path(__file__).resolve().parent.parent
    data_path = Path(data_dir) if data_dir is not None else project_root / "data"
    input_paths = [data_path / name for name in SOURCE_FILES]
    missing_paths = [str(path) for path in input_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Missing source CSV file(s): " + ", ".join(missing_paths))

    data = _read_matching_csvs(input_paths)

    missing_drop_columns = sorted(set(DROP_COLUMNS) - set(data.columns))
    required_location_columns = {"place_with_parent_names", "lat-lon"}
    missing_location_columns = sorted(required_location_columns - set(data.columns))
    if missing_drop_columns or missing_location_columns:
        missing = missing_drop_columns + missing_location_columns
        raise ValueError(f"Input data is missing required notebook columns: {missing}")

    data = data.drop(columns=list(DROP_COLUMNS)).copy()
    data["place_with_parent_names"] = data["place_with_parent_names"].astype("string")
    data["lat-lon"] = data["lat-lon"].astype("string")

    data = (
        data.assign(
            state=lambda frame: frame["place_with_parent_names"].str.split("|", expand=True)[2],
            lat=lambda frame: pd.to_numeric(
                frame["lat-lon"].str.split(",").str[0], errors="coerce"
            ),
            lon=lambda frame: pd.to_numeric(
                frame["lat-lon"].str.split(",").str[1], errors="coerce"
            ),
        )
        .drop(columns=["place_with_parent_names", "lat-lon"])
    )

    data["surface_covered_in_m2"] = (
        data["surface_covered_in_m2"].interpolate(method="linear").round(2)
    )
    # In particular, don't impute missing price targets; discard rows with any
    # unresolved feature or target values as the notebook does.
    data = data.dropna().reset_index(drop=True)

    destination = None
    if output_path is not None:
        destination = Path(output_path)
        if not destination.is_absolute():
            destination = data_path / destination

    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(destination, index=False)

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", help="Directory containing the five source CSVs")
    parser.add_argument("--output", help="Clean CSV path; defaults to the data directory")
    args = parser.parse_args()
    cleaned = load_and_clean_data(data_dir=args.data_dir, output_path=args.output or CLEAN_FILE)
    print(f"Saved cleaned data: {len(cleaned):,} rows, {len(cleaned.columns)} columns")
