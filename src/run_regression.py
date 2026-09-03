"""Evaluate chronological baselines and regression models for departure delay."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, PREDICTOR_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "regression_features.parquet"
DEFAULT_TABLE = PROJECT_ROOT / "outputs" / "tables" / "regression_model_comparison.csv"
DEFAULT_FALLBACK_TABLE = (
    PROJECT_ROOT / "outputs" / "tables" / "regression_grouped_baseline_fallbacks.csv"
)
DEFAULT_IMPORTANCE_TABLE = (
    PROJECT_ROOT / "outputs" / "tables" / "regression_feature_importance.csv"
)
DEFAULT_VALIDATION_FIGURE = (
    PROJECT_ROOT / "outputs" / "figures" / "regression_validation_comparison.png"
)
DEFAULT_TEST_FIGURE = (
    PROJECT_ROOT / "outputs" / "figures" / "regression_test_performance.png"
)
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "outputs" / "predictions" / "regression_test_predictions.parquet"
)
DEFAULT_MODEL = PROJECT_ROOT / "outputs" / "models" / "regression_pipeline.joblib"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "regression_report.json"

TARGET = "departure_delay_seconds"
MODEL_NAMES = {
    "ridge": "Ridge regression",
    "random_forest": "Random Forest",
}
GROUP_LEVELS = [
    (
        "route + stop + weekday + hour",
        ["route_name", "stop_id", "service_day_of_week", "scheduled_hour"],
    ),
    ("route + stop + hour", ["route_name", "stop_id", "scheduled_hour"]),
    ("route + stop", ["route_name", "stop_id"]),
    ("route", ["route_name"]),
]


def load_split(path: Path, split: str) -> pd.DataFrame:
    """Load one chronological split and retain compact categorical dtypes."""
    columns = [
        "service_date",
        "scheduled_event_time",
        "data_split",
        *PREDICTOR_COLUMNS,
        TARGET,
    ]
    data = pd.read_parquet(
        path,
        columns=columns,
        filters=[("data_split", "==", split)],
        dtype_backend="pyarrow",
    )
    if data.empty:
        raise ValueError(f"No rows found for split: {split}")
    if data[TARGET].isna().any():
        raise ValueError(f"Missing target values found in split: {split}")
    return data


def prepare_predictors(data: pd.DataFrame) -> pd.DataFrame:
    """Convert predictors to representations accepted by scikit-learn."""
    result = data[PREDICTOR_COLUMNS].copy()
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].fillna("<MISSING>").astype("category")
    for column in NUMERICAL_FEATURES:
        result[column] = result[column].astype("float32")
    return result


def make_preprocessor() -> ColumnTransformer:
    """Encode categories and scale numeric fields using fitted training values."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    dtype=np.float32,
                ),
                CATEGORICAL_FEATURES,
            ),
            ("numerical", StandardScaler(), NUMERICAL_FEATURES),
        ],
        sparse_threshold=1.0,
    )


def make_model(name: str) -> Ridge | RandomForestRegressor:
    """Return one fixed, untuned candidate model."""
    if name == "ridge":
        return Ridge(alpha=10.0)
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=60,
            max_depth=18,
            min_samples_leaf=25,
            max_features="sqrt",
            max_samples=400_000,
            n_jobs=-1,
            random_state=42,
        )
    raise ValueError(f"Unknown model: {name}")


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics in seconds and the unitless R-squared score."""
    return {
        "mae_seconds": float(mean_absolute_error(actual, predicted)),
        "rmse_seconds": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def metric_row(
    phase: str,
    model: str,
    training_period: str,
    evaluation_period: str,
    actual: np.ndarray,
    predicted: np.ndarray,
    fit_seconds: float | None,
    selected_for_test: bool,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "model": model,
        "training_period": training_period,
        "evaluation_period": evaluation_period,
        "evaluation_rows": len(actual),
        **calculate_metrics(actual, predicted),
        "fit_seconds": fit_seconds,
        "selected_for_test": selected_for_test,
    }


def grouped_median_predictions(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    minimum_group_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Predict from increasingly broad training-only historical median groups."""
    required = [
        "route_name",
        "stop_id",
        "service_day_of_week",
        "scheduled_minute_of_service_day",
        TARGET,
    ]
    train = training[required].copy()
    evaluate = evaluation[required[:-1]].copy()
    train["scheduled_hour"] = (
        train["scheduled_minute_of_service_day"] // 60
    ).astype("int16")
    evaluate["scheduled_hour"] = (
        evaluate["scheduled_minute_of_service_day"] // 60
    ).astype("int16")

    predictions = np.full(len(evaluate), np.nan, dtype=np.float64)
    fallbacks: list[dict[str, Any]] = []

    for level_name, keys in GROUP_LEVELS:
        summary = train.groupby(keys, observed=True, dropna=False)[TARGET].agg(
            ["median", "size"]
        )
        summary = summary.loc[summary["size"].ge(minimum_group_size), "median"]
        if len(keys) == 1:
            candidates = summary.reindex(evaluate[keys[0]]).to_numpy(dtype=float)
        else:
            evaluation_index = pd.MultiIndex.from_frame(evaluate[keys])
            candidates = summary.reindex(evaluation_index).to_numpy(dtype=float)
        use = np.isnan(predictions) & ~np.isnan(candidates)
        predictions[use] = candidates[use]
        fallbacks.append(
            {
                "fallback_level": level_name,
                "records": int(use.sum()),
                "percent": float(use.mean() * 100),
            }
        )

    global_median = float(training[TARGET].median())
    use_global = np.isnan(predictions)
    predictions[use_global] = global_median
    fallbacks.append(
        {
            "fallback_level": "global median",
            "records": int(use_global.sum()),
            "percent": float(use_global.mean() * 100),
        }
    )
    if np.isnan(predictions).any():
        raise RuntimeError("Grouped baseline produced missing predictions")
    return predictions, fallbacks


def fit_candidates(
    training: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[list[dict[str, Any]], str]:
    """Fit fixed candidates on training data and select the lowest validation MAE."""
    preprocessor = make_preprocessor()
    train_predictors = prepare_predictors(training)
    validation_predictors = prepare_predictors(validation)
    transform_start = time.perf_counter()
    train_matrix = preprocessor.fit_transform(train_predictors)
    validation_matrix = preprocessor.transform(validation_predictors)
    transform_seconds = time.perf_counter() - transform_start
    train_target = training[TARGET].to_numpy(dtype=np.float32)
    validation_target = validation[TARGET].to_numpy(dtype=np.float32)

    results: list[dict[str, Any]] = []
    for name in MODEL_NAMES:
        model = make_model(name)
        print(f"Fitting validation candidate: {MODEL_NAMES[name]}", flush=True)
        fit_start = time.perf_counter()
        model.fit(train_matrix, train_target)
        fit_seconds = time.perf_counter() - fit_start
        prediction = model.predict(validation_matrix)
        row = metric_row(
            "validation",
            MODEL_NAMES[name],
            "2026-01-01 to 2026-04-30",
            "2026-05-01 to 2026-05-31",
            validation_target,
            prediction,
            fit_seconds,
            False,
        )
        row["preprocessing_fit_transform_seconds"] = transform_seconds
        results.append(row)
        del model, prediction
        gc.collect()

    selected_name = min(
        MODEL_NAMES,
        key=lambda candidate: next(
            row["mae_seconds"]
            for row in results
            if row["model"] == MODEL_NAMES[candidate]
        ),
    )
    for row in results:
        row["selected_for_test"] = row["model"] == MODEL_NAMES[selected_name]
    return results, selected_name


def fit_final_model(
    development: pd.DataFrame,
    test: pd.DataFrame,
    selected_name: str,
    model_path: Path,
) -> tuple[np.ndarray, float, int, pd.DataFrame]:
    """Refit the validation-selected pipeline on January-May and predict June."""
    preprocessor = make_preprocessor()
    development_predictors = prepare_predictors(development)
    test_predictors = prepare_predictors(test)
    development_matrix = preprocessor.fit_transform(development_predictors)
    test_matrix = preprocessor.transform(test_predictors)
    model = make_model(selected_name)
    print(
        f"Refitting selected model on January-May: {MODEL_NAMES[selected_name]}",
        flush=True,
    )
    fit_start = time.perf_counter()
    model.fit(development_matrix, development[TARGET].to_numpy(dtype=np.float32))
    fit_seconds = time.perf_counter() - fit_start
    predictions = model.predict(test_matrix)
    encoded_features = int(development_matrix.shape[1])
    encoded_names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        encoded_importance = np.asarray(model.feature_importances_)
        importance_method = "impurity-based importance"
    else:
        encoded_importance = np.abs(np.asarray(model.coef_))
        encoded_importance = encoded_importance / encoded_importance.sum()
        importance_method = "normalized absolute coefficient"

    importance_rows = []
    for column in PREDICTOR_COLUMNS:
        belongs_to_column = np.array(
            [
                name.startswith(f"categorical__{column}_")
                or name == f"numerical__{column}"
                for name in encoded_names
            ]
        )
        importance_rows.append(
            {
                "feature": column,
                "importance": float(encoded_importance[belongs_to_column].sum()),
                "method": importance_method,
            }
        )
    importance = pd.DataFrame(importance_rows).sort_values(
        "importance", ascending=False, ignore_index=True
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "predictor_columns": PREDICTOR_COLUMNS,
            "target": TARGET,
            "trained_through": "2026-05-31",
        },
        model_path,
        compress=3,
    )
    return predictions, fit_seconds, encoded_features, importance


def save_validation_figure(results: pd.DataFrame, path: Path) -> None:
    validation = results.loc[results["phase"].eq("validation")].copy()
    order = validation.sort_values("mae_seconds")["model"]
    validation = validation.set_index("model").loc[order]
    positions = np.arange(len(validation))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(
        positions - width / 2,
        validation["mae_seconds"],
        width,
        label="MAE",
        color="#33658A",
    )
    axis.bar(
        positions + width / 2,
        validation["rmse_seconds"],
        width,
        label="RMSE",
        color="#F6AE2D",
    )
    axis.set_xticks(positions, validation.index, rotation=18, ha="right")
    axis.set_ylabel("Error (seconds)")
    axis.set_title("Regression performance on May 2026 validation data")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_test_figure(actual: np.ndarray, predicted: np.ndarray, path: Path) -> None:
    rng = np.random.default_rng(42)
    sample_size = min(50_000, len(actual))
    sample = rng.choice(len(actual), size=sample_size, replace=False)
    actual_sample = actual[sample]
    predicted_sample = predicted[sample]
    lower = float(np.quantile(actual_sample, 0.01))
    upper = float(np.quantile(actual_sample, 0.99))
    absolute_error = np.abs(actual - predicted)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hexbin(
        actual_sample,
        predicted_sample,
        gridsize=55,
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    axes[0].plot([lower, upper], [lower, upper], color="#D1495B", linewidth=1.5)
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_xlabel("Actual departure delay (seconds)")
    axes[0].set_ylabel("Predicted departure delay (seconds)")
    axes[0].set_title("Actual versus predicted (1st–99th percentile view)")

    error_limit = float(np.quantile(absolute_error, 0.99))
    axes[1].hist(
        absolute_error[absolute_error <= error_limit],
        bins=60,
        color="#33658A",
        edgecolor="white",
    )
    axes[1].axvline(
        np.median(absolute_error),
        color="#D1495B",
        linestyle="--",
        label=f"Median = {np.median(absolute_error):.0f} s",
    )
    axes[1].set_xlabel("Absolute error (seconds)")
    axes[1].set_ylabel("Test records")
    axes[1].set_title("Absolute-error distribution (through 99th percentile)")
    axes[1].legend(frameon=False)
    fig.suptitle("Selected regression model on June 2026 test data", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_analysis(
    input_path: Path,
    comparison_path: Path,
    fallback_path: Path,
    importance_path: Path,
    validation_figure: Path,
    test_figure: Path,
    predictions_path: Path,
    model_path: Path,
    report_path: Path,
    minimum_group_size: int,
) -> dict[str, Any]:
    """Run validation selection and one final chronological test evaluation."""
    if not input_path.exists():
        raise FileNotFoundError(f"Regression feature table not found: {input_path}")
    if minimum_group_size < 1:
        raise ValueError("minimum_group_size must be positive")

    train = load_split(input_path, "train")
    validation = load_split(input_path, "validation")
    test = load_split(input_path, "test")
    train_target = train[TARGET].to_numpy(dtype=np.float32)
    validation_target = validation[TARGET].to_numpy(dtype=np.float32)
    test_target = test[TARGET].to_numpy(dtype=np.float32)

    results: list[dict[str, Any]] = []
    fallback_rows: list[dict[str, Any]] = []

    train_median = float(np.median(train_target))
    results.append(
        metric_row(
            "validation",
            "Global median baseline",
            "2026-01-01 to 2026-04-30",
            "2026-05-01 to 2026-05-31",
            validation_target,
            np.full(len(validation), train_median),
            None,
            False,
        )
    )
    grouped_validation, validation_fallbacks = grouped_median_predictions(
        train, validation, minimum_group_size
    )
    results.append(
        metric_row(
            "validation",
            "Historical grouped median",
            "2026-01-01 to 2026-04-30",
            "2026-05-01 to 2026-05-31",
            validation_target,
            grouped_validation,
            None,
            False,
        )
    )
    fallback_rows.extend(
        {"phase": "validation", **row} for row in validation_fallbacks
    )

    candidate_results, selected_name = fit_candidates(train, validation)
    results.extend(candidate_results)

    development = pd.concat([train, validation], ignore_index=True)
    development_target = development[TARGET].to_numpy(dtype=np.float32)
    development_median = float(np.median(development_target))
    results.append(
        metric_row(
            "test",
            "Global median baseline",
            "2026-01-01 to 2026-05-31",
            "2026-06-01 to 2026-06-30",
            test_target,
            np.full(len(test), development_median),
            None,
            False,
        )
    )
    grouped_test, test_fallbacks = grouped_median_predictions(
        development, test, minimum_group_size
    )
    results.append(
        metric_row(
            "test",
            "Historical grouped median",
            "2026-01-01 to 2026-05-31",
            "2026-06-01 to 2026-06-30",
            test_target,
            grouped_test,
            None,
            False,
        )
    )
    fallback_rows.extend({"phase": "test", **row} for row in test_fallbacks)

    (
        final_predictions,
        final_fit_seconds,
        encoded_features,
        feature_importance,
    ) = fit_final_model(development, test, selected_name, model_path)
    results.append(
        metric_row(
            "test",
            MODEL_NAMES[selected_name],
            "2026-01-01 to 2026-05-31",
            "2026-06-01 to 2026-06-30",
            test_target,
            final_predictions,
            final_fit_seconds,
            True,
        )
    )

    comparison = pd.DataFrame(results)
    numeric_columns = [
        "mae_seconds",
        "rmse_seconds",
        "r2",
        "fit_seconds",
        "preprocessing_fit_transform_seconds",
    ]
    for column in numeric_columns:
        if column in comparison:
            comparison[column] = comparison[column].round(4)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    fallbacks = pd.DataFrame(fallback_rows)
    fallbacks["percent"] = fallbacks["percent"].round(4)
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    fallbacks.to_csv(fallback_path, index=False)

    importance_path.parent.mkdir(parents=True, exist_ok=True)
    feature_importance.assign(
        importance=feature_importance["importance"].round(6)
    ).to_csv(importance_path, index=False)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_output = test[
        [
            "service_date",
            "scheduled_event_time",
            "route_name",
            "direction_code",
            "stop_id",
            TARGET,
        ]
    ].copy()
    prediction_output["predicted_departure_delay_seconds"] = final_predictions
    prediction_output["absolute_error_seconds"] = np.abs(
        test_target - final_predictions
    )
    prediction_output.to_parquet(predictions_path, index=False)

    save_validation_figure(comparison, validation_figure)
    save_test_figure(test_target, final_predictions, test_figure)

    selected_validation = next(
        row
        for row in results
        if row["phase"] == "validation"
        and row["model"] == MODEL_NAMES[selected_name]
    )
    selected_test = results[-1]
    report = {
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
        "random_seed": 42,
        "minimum_group_size": minimum_group_size,
        "candidate_models": list(MODEL_NAMES.values()),
        "selected_machine_learning_model": MODEL_NAMES[selected_name],
        "selection_metric": "validation MAE",
        "validation_rows": len(validation),
        "test_rows": len(test),
        "encoded_feature_columns": encoded_features,
        "selected_validation_metrics": {
            key: selected_validation[key] for key in ("mae_seconds", "rmse_seconds", "r2")
        },
        "selected_test_metrics": {
            key: selected_test[key] for key in ("mae_seconds", "rmse_seconds", "r2")
        },
        "outputs": {
            "comparison": str(comparison_path.resolve()),
            "fallbacks": str(fallback_path.resolve()),
            "feature_importance": str(importance_path.resolve()),
            "validation_figure": str(validation_figure.resolve()),
            "test_figure": str(test_figure.resolve()),
            "test_predictions": str(predictions_path.resolve()),
            "model": str(model_path.resolve()),
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate departure-delay regression with chronological splits."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--fallbacks", type=Path, default=DEFAULT_FALLBACK_TABLE)
    parser.add_argument("--importance", type=Path, default=DEFAULT_IMPORTANCE_TABLE)
    parser.add_argument("--validation-figure", type=Path, default=DEFAULT_VALIDATION_FIGURE)
    parser.add_argument("--test-figure", type=Path, default=DEFAULT_TEST_FIGURE)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--minimum-group-size", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_analysis(
        args.input,
        args.comparison,
        args.fallbacks,
        args.importance,
        args.validation_figure,
        args.test_figure,
        args.predictions,
        args.model,
        args.report,
        args.minimum_group_size,
    )
    print(
        "Selected machine-learning model: "
        f"{report['selected_machine_learning_model']}"
    )
    print(
        "June test MAE: "
        f"{report['selected_test_metrics']['mae_seconds']:.2f} seconds"
    )


if __name__ == "__main__":
    main()
