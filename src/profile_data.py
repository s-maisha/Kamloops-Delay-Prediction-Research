"""Create a reproducible structural and data-quality profile of the ingested data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "transit_events.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "data_profile.json"

SOURCE_COLUMNS = [
    "service_date",
    "transit_system_name",
    "route_name",
    "direction_code",
    "trip_id",
    "vehicle_id",
    "stop_id",
    "stop_name",
    "stop_sequence",
    "scheduled_arrival_time",
    "scheduled_departure_time",
    "actual_arrival_time",
    "actual_departure_time",
    "delay_on_departure_seconds",
    "on_time_performance_deviation_seconds",
    "on_time_performance_status",
]

TIMESTAMP_COLUMNS = [
    "scheduled_arrival_time",
    "scheduled_departure_time",
    "actual_arrival_time",
    "actual_departure_time",
]

DELAY_COLUMNS = [
    "delay_on_departure_seconds",
    "on_time_performance_deviation_seconds",
]

QUANTILES = [0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1]


def native(value: Any) -> Any:
    """Convert NumPy and pandas scalar values into JSON-compatible values."""
    if value is pd.NA or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return value


def value_counts_dict(series: pd.Series, limit: int | None = None) -> dict[str, int]:
    counts = series.value_counts(dropna=False)
    if limit is not None:
        counts = counts.head(limit)
    return {"<MISSING>" if pd.isna(key) else str(key): int(value) for key, value in counts.items()}


def column_profile(series: pd.Series, total_rows: int) -> dict[str, Any]:
    missing = int(series.isna().sum())
    unique = int(series.nunique(dropna=True))
    counts = series.value_counts(dropna=False)
    most_common_value = counts.index[0] if len(counts) else None
    most_common_count = int(counts.iloc[0]) if len(counts) else 0
    examples = [native(value) for value in series.dropna().drop_duplicates().head(3)]
    profile: dict[str, Any] = {
        "dtype": str(series.dtype),
        "non_missing": total_rows - missing,
        "missing": missing,
        "missing_percent": round(100 * missing / total_rows, 6),
        "unique_non_missing": unique,
        "most_common_value": native(most_common_value),
        "most_common_count": most_common_count,
        "most_common_percent": round(100 * most_common_count / total_rows, 6),
        "examples": examples,
    }
    if unique <= 25:
        profile["value_counts"] = value_counts_dict(series)
    return profile


def numeric_profile(series: pd.Series) -> dict[str, Any]:
    non_missing = series.dropna()
    quantiles = non_missing.quantile(QUANTILES)
    q1 = float(quantiles.loc[0.25])
    q3 = float(quantiles.loc[0.75])
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    return {
        "minimum": native(non_missing.min()),
        "maximum": native(non_missing.max()),
        "mean": float(non_missing.mean()),
        "quantiles": {str(level): native(value) for level, value in quantiles.items()},
        "iqr_lower_fence": lower_fence,
        "iqr_upper_fence": upper_fence,
        "below_iqr_fence": int((non_missing < lower_fence).sum()),
        "above_iqr_fence": int((non_missing > upper_fence).sum()),
    }


def timestamp_profile(data: pd.DataFrame, column: str) -> dict[str, Any]:
    values = data[column]
    non_missing = values.dropna()
    # Removing the UTC type is sufficient here because the source timezone is UTC;
    # no conversion to a different local timezone is intended.
    values_without_zone = values.astype(pd.ArrowDtype(pa.timestamp("us")))
    embedded_dates = values_without_zone.dt.normalize()
    offsets = (embedded_dates - data["service_date"]).dt.days
    hours = values_without_zone.dt.hour
    return {
        "minimum": native(non_missing.min()),
        "maximum": native(non_missing.max()),
        "calendar_date_offset_from_service_date": value_counts_dict(offsets),
        "midnight_to_04_59_count": int(hours.between(0, 4, inclusive="both").sum()),
    }


def relation_profile(data: pd.DataFrame) -> dict[str, Any]:
    departure_pair = data[
        ["scheduled_departure_time", "actual_departure_time"]
    ].notna().all(axis=1)
    calculated_departure_deviation = (
        data.loc[departure_pair, "actual_departure_time"]
        - data.loc[departure_pair, "scheduled_departure_time"]
    ).dt.total_seconds()
    reported_deviation = data.loc[
        departure_pair, "on_time_performance_deviation_seconds"
    ]
    comparison = (calculated_departure_deviation - reported_deviation).abs()

    arrival_pair = data[
        ["scheduled_arrival_time", "actual_arrival_time"]
    ].notna().all(axis=1)
    calculated_arrival_deviation = (
        data.loc[arrival_pair, "actual_arrival_time"]
        - data.loc[arrival_pair, "scheduled_arrival_time"]
    ).dt.total_seconds()
    reported_arrival_deviation = data.loc[
        arrival_pair, "on_time_performance_deviation_seconds"
    ]
    arrival_comparison = (
        calculated_arrival_deviation - reported_arrival_deviation
    ).abs()

    both_delays = data[DELAY_COLUMNS].notna().all(axis=1)
    delay_difference = (
        data.loc[both_delays, "delay_on_departure_seconds"]
        - data.loc[both_delays, "on_time_performance_deviation_seconds"]
    ).abs()

    return {
        "scheduled_time_presence": {
            "arrival_only": int(
                (data["scheduled_arrival_time"].notna() & data["scheduled_departure_time"].isna()).sum()
            ),
            "departure_only": int(
                (data["scheduled_arrival_time"].isna() & data["scheduled_departure_time"].notna()).sum()
            ),
            "both": int(
                (data["scheduled_arrival_time"].notna() & data["scheduled_departure_time"].notna()).sum()
            ),
            "neither": int(
                (data["scheduled_arrival_time"].isna() & data["scheduled_departure_time"].isna()).sum()
            ),
        },
        "actual_time_presence": {
            "arrival_only": int(
                (data["actual_arrival_time"].notna() & data["actual_departure_time"].isna()).sum()
            ),
            "departure_only": int(
                (data["actual_arrival_time"].isna() & data["actual_departure_time"].notna()).sum()
            ),
            "both": int(
                (data["actual_arrival_time"].notna() & data["actual_departure_time"].notna()).sum()
            ),
            "neither": int(
                (data["actual_arrival_time"].isna() & data["actual_departure_time"].isna()).sum()
            ),
        },
        "rows_with_both_departure_timestamps": int(departure_pair.sum()),
        "calculated_departure_difference_equals_reported_deviation": int(
            comparison.eq(0).sum()
        ),
        "calculated_departure_difference_within_one_second": int(
            comparison.le(1).sum()
        ),
        "departure_timestamp_difference_absolute_error_quantiles": {
            str(level): native(value)
            for level, value in comparison.quantile([0, 0.5, 0.95, 0.99, 1]).items()
        },
        "rows_with_both_arrival_timestamps": int(arrival_pair.sum()),
        "calculated_arrival_difference_equals_reported_deviation": int(
            arrival_comparison.eq(0).sum()
        ),
        "arrival_timestamp_difference_absolute_error_maximum": native(
            arrival_comparison.max()
        ),
        "departure_delay_missing_despite_actual_departure_present": int(
            (
                data["delay_on_departure_seconds"].isna()
                & data["actual_departure_time"].notna()
            ).sum()
        ),
        "rows_with_both_delay_fields": int(both_delays.sum()),
        "two_delay_fields_equal": int(delay_difference.eq(0).sum()),
        "two_delay_fields_absolute_difference_quantiles": {
            str(level): native(value)
            for level, value in delay_difference.quantile([0, 0.5, 0.95, 0.99, 1]).items()
        },
        "two_delay_fields_correlation": float(
            data.loc[both_delays, DELAY_COLUMNS].corr().iloc[0, 1]
        ),
    }


def status_profile(data: pd.DataFrame) -> dict[str, Any]:
    summary = (
        data.groupby("on_time_performance_status", dropna=False)
        .agg(
            rows=("on_time_performance_status", "size"),
            departure_delay_min=("delay_on_departure_seconds", "min"),
            departure_delay_median=("delay_on_departure_seconds", "median"),
            departure_delay_max=("delay_on_departure_seconds", "max"),
            performance_deviation_min=("on_time_performance_deviation_seconds", "min"),
            performance_deviation_median=("on_time_performance_deviation_seconds", "median"),
            performance_deviation_max=("on_time_performance_deviation_seconds", "max"),
        )
        .reset_index()
    )
    records = []
    for record in summary.to_dict(orient="records"):
        records.append({key: native(value) for key, value in record.items()})
    deviation = data["on_time_performance_deviation_seconds"]
    expected_status = pd.Series(pd.NA, index=data.index, dtype="string")
    expected_status.loc[deviation.le(-180).fillna(False)] = "Very Early"
    expected_status.loc[deviation.between(-179, -60).fillna(False)] = "Early"
    expected_status.loc[deviation.between(-59, 180).fillna(False)] = "On Time"
    expected_status.loc[deviation.between(181, 360).fillna(False)] = "Late"
    expected_status.loc[deviation.ge(361).fillna(False)] = "Very Late"
    reported_status = data["on_time_performance_status"].astype("string")

    return {
        "by_status": records,
        "empirically_observed_thresholds_seconds": {
            "Very Early": "<= -180",
            "Early": "-179 to -60",
            "On Time": "-59 to 180",
            "Late": "181 to 360",
            "Very Late": ">= 361",
        },
        "threshold_rule_mismatches": int(
            (
                expected_status.fillna("<MISSING>")
                != reported_status.fillna("<MISSING>")
            ).sum()
        ),
        "status_missing_mask_equals_deviation_missing_mask": bool(
            reported_status.isna().equals(deviation.isna())
        ),
        "departure_delay_sign_by_status": {
            ("<MISSING>" if pd.isna(status) else str(status)): {
                str(sign): int(count) for sign, count in group.items()
            }
            for status, group in pd.crosstab(
                data["on_time_performance_status"],
                np.sign(data["delay_on_departure_seconds"]),
                dropna=False,
            ).to_dict(orient="index").items()
        },
        "performance_deviation_sign_by_status": {
            ("<MISSING>" if pd.isna(status) else str(status)): {
                str(sign): int(count) for sign, count in group.items()
            }
            for status, group in pd.crosstab(
                data["on_time_performance_status"],
                np.sign(data["on_time_performance_deviation_seconds"]),
                dropna=False,
            ).to_dict(orient="index").items()
        },
    }


def stop_mapping_profile(data: pd.DataFrame) -> dict[str, Any]:
    pairs = data[["stop_id", "stop_name"]].dropna().drop_duplicates()
    names_per_id = pairs.groupby("stop_id")["stop_name"].nunique()
    ids_per_name = pairs.groupby("stop_name")["stop_id"].nunique()
    return {
        "distinct_stop_id_name_pairs": int(len(pairs)),
        "stop_ids_with_multiple_names": int(names_per_id.gt(1).sum()),
        "maximum_names_for_one_stop_id": int(names_per_id.max()),
        "stop_names_with_multiple_ids": int(ids_per_name.gt(1).sum()),
        "maximum_ids_for_one_stop_name": int(ids_per_name.max()),
    }


def extreme_records(data: pd.DataFrame) -> dict[str, Any]:
    columns = [
        "source_archive",
        "source_file",
        "source_row_number",
        "service_date",
        "route_name",
        "stop_id",
        "stop_name",
        "on_time_performance_status",
    ]
    results: dict[str, Any] = {}
    for delay_column in DELAY_COLUMNS:
        selected_columns = columns + [delay_column]
        smallest = data.nsmallest(5, delay_column)[selected_columns]
        largest = data.nlargest(5, delay_column)[selected_columns]
        results[delay_column] = {
            "smallest": [
                {key: native(value) for key, value in record.items()}
                for record in smallest.to_dict(orient="records")
            ],
            "largest": [
                {key: native(value) for key, value in record.items()}
                for record in largest.to_dict(orient="records")
            ],
        }
    return results


def build_profile(input_path: Path) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Ingested dataset not found: {input_path}")

    data = pd.read_parquet(input_path, dtype_backend="pyarrow")
    total_rows = len(data)
    missing_columns = [column for column in SOURCE_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(f"Required columns are missing: {missing_columns}")

    date_counts = data["service_date"].value_counts().sort_index()
    full_date_range = pd.date_range(
        data["service_date"].min(), data["service_date"].max(), freq="D"
    )
    missing_service_dates = full_date_range.difference(date_counts.index)

    duplicate_subset = data[SOURCE_COLUMNS]
    additional_duplicate_rows = int(duplicate_subset.duplicated(keep="first").sum())
    rows_in_duplicate_groups = int(duplicate_subset.duplicated(keep=False).sum())

    trip_ids = data["trip_id"].dropna().astype("string")
    scientific_trip_ids = trip_ids.str.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[Ee][+-]?\d+", na=False
    )

    profile: dict[str, Any] = {
        "input_file": str(input_path.resolve()),
        "input_bytes": input_path.stat().st_size,
        "rows": total_rows,
        "columns": len(data.columns),
        "memory_bytes_when_loaded": int(data.memory_usage(index=True, deep=True).sum()),
        "column_order": data.columns.tolist(),
        "columns_profile": {
            column: column_profile(data[column], total_rows) for column in data.columns
        },
        "service_date": {
            "minimum": data["service_date"].min().date().isoformat(),
            "maximum": data["service_date"].max().date().isoformat(),
            "distinct_dates": int(data["service_date"].nunique()),
            "missing_dates_within_range": [
                date.date().isoformat() for date in missing_service_dates
            ],
            "minimum_rows_on_one_date": int(date_counts.min()),
            "maximum_rows_on_one_date": int(date_counts.max()),
        },
        "duplicates_across_16_source_columns": {
            "additional_duplicate_rows": additional_duplicate_rows,
            "rows_in_duplicate_groups": rows_in_duplicate_groups,
        },
        "numeric_profiles": {
            column: numeric_profile(data[column]) for column in DELAY_COLUMNS
        },
        "timestamp_profiles": {
            column: timestamp_profile(data, column) for column in TIMESTAMP_COLUMNS
        },
        "delay_and_timestamp_relations": relation_profile(data),
        "status_profile": status_profile(data),
        "stop_mapping_profile": stop_mapping_profile(data),
        "extreme_records": extreme_records(data),
        "trip_id_format": {
            "non_missing": int(len(trip_ids)),
            "scientific_notation_count": int(scientific_trip_ids.sum()),
            "scientific_notation_percent": round(
                100 * scientific_trip_ids.sum() / len(trip_ids), 6
            ),
        },
    }
    return profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the ingested Kamloops transit dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = build_profile(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(f"Profiled {profile['rows']:,} rows and {profile['columns']} columns.")
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
