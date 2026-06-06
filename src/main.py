from __future__ import annotations

import importlib.util
from src.config import CORE_DEPENDENCIES, ensure_output_dirs


def _missing_dependencies() -> list[str]:
    missing = []
    for import_name, package_name in CORE_DEPENDENCIES.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(package_name)
    return missing


def main() -> int:
    missing = _missing_dependencies()
    if missing:
        print("Dependencias obrigatorias ausentes:")
        for package in missing:
            print(f"- {package}")
        print("\nInstale os pacotes com:")
        print("python3 -m pip install -r requirements.txt")
        return 1

    from src.eda import run_eda
    from src.explainability import run_explainability
    from src.config import SCORE_COLUMN, SUCCESS_COMPONENTS, TARGET_COLUMN
    from src.features import FEATURE_COLUMNS, add_success_target, get_model_matrix
    from src.insights import (
        build_actionable_insights,
        write_insights,
        write_presentation_script,
        write_report_base,
        write_success_definition,
    )
    from src.load_data import load_games_csv
    from src.modeling import train_and_evaluate_models
    from src.preprocessing import clean_and_engineer_base

    ensure_output_dirs()

    print("Carregando games.csv...")
    raw_df = load_games_csv()
    print(f"Registros carregados: {len(raw_df):,}")

    print("Limpando dados e criando features base...")
    engineered_df = clean_and_engineer_base(raw_df)

    print("Criando variavel-alvo success_commercial...")
    modeled_df, target_info = add_success_target(engineered_df)
    write_success_definition(target_info)
    keep_columns = sorted(
        {
            *FEATURE_COLUMNS,
            *SUCCESS_COMPONENTS,
            SCORE_COLUMN,
            TARGET_COLUMN,
            "owners_midpoint",
            "peak_ccu",
            "price",
            "total_reviews",
            "positive_ratio",
            "recommendations",
            "average_playtime_forever",
            "metacritic_score",
            "achievements",
            "free_to_play",
            "has_multiplayer",
            "is_indie",
            "has_achievements",
            "has_metacritic",
            "has_singleplayer",
        }
    )
    modeled_df = modeled_df[[column for column in keep_columns if column in modeled_df.columns]].copy()
    del raw_df, engineered_df

    print("Gerando EDA, tabelas e testes estatisticos...")
    eda_tables = run_eda(modeled_df)

    print("Treinando modelos...")
    features, target, feature_names = get_model_matrix(modeled_df)
    best_model, metrics_df, cv_df, x_test, y_test = train_and_evaluate_models(features, target)
    print(f"Melhor modelo: {best_model.name}")

    print("Gerando explicabilidade...")
    feature_importance, _ = run_explainability(best_model, x_test, y_test, feature_names)

    print("Gerando insights e relatorio-base...")
    insights = build_actionable_insights(eda_tables["group_comparisons"], feature_importance)
    write_insights(insights)
    write_report_base(
        target_info,
        metrics_df,
        cv_df,
        eda_tables["statistical_tests"],
        insights,
        eda_tables["summary"],
        feature_importance,
    )
    write_presentation_script(metrics_df, insights, feature_importance)

    print("Pipeline finalizado com sucesso.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
