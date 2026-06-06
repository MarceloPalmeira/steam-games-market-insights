from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import SCORE_COLUMN, SUCCESS_COMPONENTS, SUCCESS_TOP_SHARE, TARGET_COLUMN


LEAKAGE_COLUMNS = [
    "Estimated owners",
    "owners_lower",
    "owners_upper",
    "owners_midpoint",
    "Peak CCU",
    "peak_ccu",
    "Positive",
    "Negative",
    "positive_reviews",
    "negative_reviews",
    "total_reviews",
    "review_score_weighted",
    "positive_ratio",
    "Recommendations",
    "recommendations",
    "Average playtime forever",
    "average_playtime_forever",
    SCORE_COLUMN,
    TARGET_COLUMN,
]

FEATURE_COLUMNS = [
    "release_year",
    "game_age_years",
    "price",
    "log_price",
    "free_to_play",
    "required_age",
    "discount",
    "dlc_count",
    "metacritic_score",
    "has_metacritic",
    "achievements",
    "log_achievements",
    "has_achievements",
    "windows",
    "mac",
    "linux",
    "category_count",
    "genre_count",
    "tag_count",
    "supported_languages_count",
    "full_audio_languages_count",
    "has_multiplayer",
    "has_singleplayer",
    "has_coop",
    "is_indie",
    "is_action",
    "is_adventure",
    "is_casual",
    "is_simulation",
    "is_strategy",
    "is_rpg",
    "is_early_access",
    "vr_supported",
    "has_steam_achievements",
    "has_family_sharing",
    "has_steam_cloud",
    "has_trading_cards",
    "about_length",
    "about_word_count",
    "has_website",
    "has_support_url",
]


def add_success_target(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    data = df.copy()
    component_scores: list[pd.Series] = []
    component_details: dict[str, dict[str, float]] = {}

    for component in SUCCESS_COMPONENTS:
        values = data[component].fillna(0).clip(lower=0)
        transformed = np.log1p(values)
        ranked = transformed.rank(pct=True, method="average")
        component_scores.append(ranked)
        component_details[component] = {
            "min": float(values.min()),
            "median": float(values.median()),
            "p75": float(values.quantile(0.75)),
            "p90": float(values.quantile(0.90)),
            "max": float(values.max()),
        }

    data[SCORE_COLUMN] = pd.concat(component_scores, axis=1).mean(axis=1)
    cutoff = float(data[SCORE_COLUMN].quantile(1 - SUCCESS_TOP_SHARE))
    data[TARGET_COLUMN] = (data[SCORE_COLUMN] >= cutoff).astype(int)

    target_info = {
        "target_column": TARGET_COLUMN,
        "score_column": SCORE_COLUMN,
        "components": SUCCESS_COMPONENTS,
        "top_share": SUCCESS_TOP_SHARE,
        "cutoff": cutoff,
        "positive_rate": float(data[TARGET_COLUMN].mean()),
        "leakage_columns_removed": LEAKAGE_COLUMNS,
        "component_details": component_details,
    }
    return data, target_info


def get_model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    available_features = [column for column in FEATURE_COLUMNS if column in df.columns]
    feature_data = df[available_features].replace([np.inf, -np.inf], np.nan)
    target = df[TARGET_COLUMN]
    return feature_data, target, available_features
