import csv
from pathlib import Path

import pandas as pd

from src.config import CSV_PATH


def _expand_known_header_issues(columns: list[str]) -> list[str]:
    """Fix the Steam CSV header where Discount and DLC count are merged."""
    fixed_columns: list[str] = []
    for column in columns:
        if column == "DiscountDLC count":
            fixed_columns.extend(["Discount", "DLC count"])
        else:
            fixed_columns.append(column)
    return fixed_columns


def read_csv_header(path: Path = CSV_PATH) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        return next(reader)


def load_games_csv(path: Path = CSV_PATH) -> pd.DataFrame:
    raw_header = read_csv_header(path)
    columns = _expand_known_header_issues(raw_header)
    df = pd.read_csv(
        path,
        header=0,
        names=columns,
        encoding="utf-8-sig",
        low_memory=False,
        on_bad_lines="warn",
    )
    df.columns = [str(column).strip() for column in df.columns]
    return df
