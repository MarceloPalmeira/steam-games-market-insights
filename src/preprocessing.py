from __future__ import annotations

import ast
import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import ANALYSIS_DATE


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        found = normalized.get(normalize_column_name(candidate))
        if found is not None:
            return found
    return None


def parse_bool(value: object) -> int:
    text = str(value).strip().lower()
    return int(text in {"true", "1", "yes", "sim"})


def parse_owners_range(value: object) -> tuple[float, float, float]:
    if pd.isna(value):
        return np.nan, np.nan, np.nan
    text = str(value).replace(",", "").replace(" ", "")
    if "-" in text:
        left, right = text.split("-", 1)
        try:
            lower = float(left)
            upper = float(right)
            return lower, upper, (lower + upper) / 2
        except ValueError:
            return np.nan, np.nan, np.nan
    try:
        number = float(text)
        return number, number, number
    except ValueError:
        return np.nan, np.nan, np.nan


def parse_list_like(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text or text in {"[]", "0"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass
    return [part.strip() for part in text.split(",") if part.strip()]


def contains_any(values: list[str], terms: Iterable[str]) -> int:
    joined = " ".join(values).lower()
    return int(any(term.lower() in joined for term in terms))


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_and_engineer_base(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    numeric_columns = [
        "Peak CCU",
        "Required age",
        "Price",
        "Discount",
        "DLC count",
        "Metacritic score",
        "User score",
        "Positive",
        "Negative",
        "Score rank",
        "Achievements",
        "Recommendations",
        "Average playtime forever",
        "Average playtime two weeks",
        "Median playtime forever",
        "Median playtime two weeks",
    ]
    for column in numeric_columns:
        if column in data.columns:
            data[column] = safe_numeric(data[column])

    owner_values = data.get("Estimated owners", pd.Series(index=data.index, dtype=object)).apply(
        parse_owners_range
    )
    data["owners_lower"] = owner_values.apply(lambda item: item[0])
    data["owners_upper"] = owner_values.apply(lambda item: item[1])
    data["owners_midpoint"] = owner_values.apply(lambda item: item[2])

    release_date = pd.to_datetime(data.get("Release date"), errors="coerce")
    analysis_date = pd.Timestamp(ANALYSIS_DATE)
    data["release_date_parsed"] = release_date
    data["release_year"] = release_date.dt.year
    data["game_age_years"] = ((analysis_date - release_date).dt.days / 365.25).clip(lower=0)

    data["positive_reviews"] = data.get("Positive", 0).fillna(0)
    data["negative_reviews"] = data.get("Negative", 0).fillna(0)
    data["total_reviews"] = data["positive_reviews"] + data["negative_reviews"]
    data["positive_ratio"] = np.where(
        data["total_reviews"] > 0,
        data["positive_reviews"] / data["total_reviews"],
        np.nan,
    )
    data["review_score_weighted"] = data["positive_ratio"].fillna(0) * np.log1p(data["total_reviews"])

    data["price"] = data.get("Price", 0).fillna(0).clip(lower=0)
    data["log_price"] = np.log1p(data["price"])
    data["free_to_play"] = (data["price"] <= 0).astype(int)

    data["peak_ccu"] = data.get("Peak CCU", 0).fillna(0).clip(lower=0)
    data["recommendations"] = data.get("Recommendations", 0).fillna(0).clip(lower=0)
    data["average_playtime_forever"] = data.get("Average playtime forever", 0).fillna(0).clip(lower=0)
    data["metacritic_score"] = data.get("Metacritic score", 0).fillna(0)
    data["has_metacritic"] = (data["metacritic_score"] > 0).astype(int)
    data["achievements"] = data.get("Achievements", 0).fillna(0).clip(lower=0)
    data["log_achievements"] = np.log1p(data["achievements"])
    data["has_achievements"] = (data["achievements"] > 0).astype(int)
    data["required_age"] = data.get("Required age", 0).fillna(0).clip(lower=0)
    data["discount"] = data.get("Discount", 0).fillna(0).clip(lower=0)
    data["dlc_count"] = data.get("DLC count", 0).fillna(0).clip(lower=0)

    for platform in ["Windows", "Mac", "Linux"]:
        if platform in data.columns:
            data[platform.lower()] = data[platform].apply(parse_bool)
        else:
            data[platform.lower()] = 0

    list_columns = {
        "Categories": "categories_list",
        "Genres": "genres_list",
        "Tags": "tags_list",
        "Supported languages": "supported_languages_list",
        "Full audio languages": "full_audio_languages_list",
    }
    for source, target in list_columns.items():
        if source in data.columns:
            data[target] = data[source].apply(parse_list_like)
        else:
            data[target] = [[] for _ in range(len(data))]

    data["category_count"] = data["categories_list"].apply(len)
    data["genre_count"] = data["genres_list"].apply(len)
    data["tag_count"] = data["tags_list"].apply(len)
    data["supported_languages_count"] = data["supported_languages_list"].apply(len)
    data["full_audio_languages_count"] = data["full_audio_languages_list"].apply(len)

    combined_taxonomy = data.apply(
        lambda row: row["categories_list"] + row["genres_list"] + row["tags_list"], axis=1
    )
    data["has_multiplayer"] = combined_taxonomy.apply(
        lambda values: contains_any(values, ["multi-player", "multiplayer", "mmo", "online pvp"])
    )
    data["has_singleplayer"] = combined_taxonomy.apply(
        lambda values: contains_any(values, ["single-player", "singleplayer"])
    )
    data["has_coop"] = combined_taxonomy.apply(lambda values: contains_any(values, ["co-op", "coop"]))
    data["is_indie"] = combined_taxonomy.apply(lambda values: contains_any(values, ["indie"]))
    data["is_action"] = combined_taxonomy.apply(lambda values: contains_any(values, ["action"]))
    data["is_adventure"] = combined_taxonomy.apply(lambda values: contains_any(values, ["adventure"]))
    data["is_casual"] = combined_taxonomy.apply(lambda values: contains_any(values, ["casual"]))
    data["is_simulation"] = combined_taxonomy.apply(lambda values: contains_any(values, ["simulation"]))
    data["is_strategy"] = combined_taxonomy.apply(lambda values: contains_any(values, ["strategy"]))
    data["is_rpg"] = combined_taxonomy.apply(lambda values: contains_any(values, ["rpg", "role-playing"]))
    data["is_early_access"] = combined_taxonomy.apply(lambda values: contains_any(values, ["early access"]))
    data["vr_supported"] = combined_taxonomy.apply(lambda values: contains_any(values, ["vr"]))
    data["has_steam_achievements"] = combined_taxonomy.apply(
        lambda values: contains_any(values, ["steam achievements"])
    )
    data["has_family_sharing"] = combined_taxonomy.apply(lambda values: contains_any(values, ["family sharing"]))
    data["has_steam_cloud"] = combined_taxonomy.apply(lambda values: contains_any(values, ["steam cloud"]))
    data["has_trading_cards"] = combined_taxonomy.apply(
        lambda values: contains_any(values, ["steam trading cards"])
    )

    text = data.get("About the game", pd.Series("", index=data.index)).fillna("").astype(str)
    data["about_length"] = text.str.len()
    data["about_word_count"] = text.str.split().str.len().fillna(0)
    data["has_website"] = data.get("Website", pd.Series("", index=data.index)).fillna("").astype(str).str.len().gt(0).astype(int)
    data["has_support_url"] = data.get("Support url", pd.Series("", index=data.index)).fillna("").astype(str).str.len().gt(0).astype(int)

    return data
