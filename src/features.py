"""Build leakage-safe datasets for departure delay prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "transit_events_clean.parquet"
DEFAULT_REGRESSION_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "regression_features.parquet"
)
DEFAULT_CLASSIFICATION_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "classification_features.parquet"
)
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "feature_report.json"

VALIDATION_START = date(2026, 5, 1)
TEST_START = date(2026, 6, 1)

SOURCE_COLUMNS = [
    "service_date",
    "route_name",
    "direction_code",
    "stop_id",
    "stop_sequence",
    "scheduled_event_time",
    "delay_on_departure_seconds",
    "on_time_performance_status",
    "event_type",
    "is_after_midnight_service",
]

CONTEXT_COLUMNS = ["service_date", "scheduled_event_time", "data_split"]
CATEGORICAL_FEATURES = [
    "route_name",
    "direction_code",
    "stop_id",
    "service_day_of_week",
]
NUMERICAL_FEATURES = [
    "stop_sequence",
    "scheduled_minute_of_service_day",
    "service_month_number",
]
PREDICTOR_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

STATUS_MAPPING = {
    "Very Early": "Early",
    "Early": "Early",
    "On Time": "On Time",
    "Late": "Late",
    "Very Late": "Late",
}


def sha256_file(path: Path) -> str:
    """Return a SHA-256 checksum without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assign_split(
    service_date: pd.Series,
    validation_start: date,
    test_start: date,
) -> pd.Series:
    """Assign whole service days to chronological train, validation, and test sets."""
    if validation_start >= test_start:
        raise ValueError("validation_start must be before test_start")

    split = pd.Series("train", index=service_date.index, dtype="string[pyarrow]")
    split.loc[service_date.ge(pd.Timestamp(validation_start))] = "validation"
    split.loc[service_date.ge(pd.Timestamp(test_start))] = "test"
    return split


def make_common_fields(
    data: pd.DataFrame,
    validation_start: date,
    test_start: date,
) -> pd.DataFrame:
    """Create predictors known before the scheduled event occurs."""
    scheduled = data["scheduled_event_time"]
    scheduled_clock = scheduled.astype(pd.ArrowDtype(pa.timestamp("us")))
    service_day_offset = data["is_after_midnight_service"].astype("int16")
    # NumPy-backed dates avoid a Windows Arrow timezone-database dependency for labels.
    service_date = pd.Series(data["service_date"].to_numpy(), index=data.index)

    result = pd.DataFrame(index=data.index)
    result["service_date"] = data["service_date"]
    result["scheduled_event_time"] = scheduled
    result["data_split"] = assign_split(
        data["service_date"], validation_start, test_start
    )
    result["route_name"] = data["route_name"]
    result["direction_code"] = data["direction_code"]
    result["stop_id"] = data["stop_id"]
    result["service_day_of_week"] = service_date.dt.day_name().astype(
        "string[pyarrow]"
    )
    result["stop_sequence"] = data["stop_sequence"]
    result["scheduled_minute_of_service_day"] = (
        scheduled_clock.dt.hour.astype("int16") * 60
        + scheduled_clock.dt.minute.astype("int16")
        + service_day_offset * 1440
    ).astype("int16")
    result["service_month_number"] = service_date.dt.month.astype("int8")
    return result


def update_counts(
    totals: Counter[str],
    frame: pd.DataFrame,
    target_column: str,
) -> None:
    totals["rows"] += len(frame)
    totals["missing_direction"] += int(frame["direction_code"].isna().sum())
    for split, count in frame["data_split"].value_counts().items():
        totals[f"split_{split}"] += int(count)
    totals["missing_target"] += int(frame[target_column].isna().sum())


def write_batch(
    writer: pq.ParquetWriter | None,
    frame: pd.DataFrame,
    path: Path,
) -> pq.ParquetWriter:
    """Write one batch while requiring a stable schema."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="snappy")
    elif writer.schema != table.schema:
        raise ValueError("Feature schema changed between processing batches")
    writer.write_table(table)
    return writer


def validate_output(
    path: Path,
    target_column: str,
    expected_rows: int,
) -> int:
    """Verify the generated feature table before it replaces an existing output."""
    output = pq.ParquetFile(path)
    expected_columns = CONTEXT_COLUMNS + PREDICTOR_COLUMNS + [target_column]
    if output.schema_arrow.names != expected_columns:
        raise RuntimeError(
            f"Unexpected columns in {path.name}: {output.schema_arrow.names}"
        )
    if output.metadata.num_rows != expected_rows:
        raise RuntimeError(
            f"Unexpected row count in {path.name}: {output.metadata.num_rows}"
        )
    row_count = output.metadata.num_rows
    del output
    return row_count


def build_features(
    input_path: Path,
    regression_output: Path,
    classification_output: Path,
    report_path: Path,
    validation_start: date,
    test_start: date,
    batch_size: int,
) -> dict[str, Any]:
    """Create separate regression and classification tables from cleaned events."""
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {input_path}")
    if validation_start >= test_start:
        raise ValueError("validation_start must be before test_start")

    source = pq.ParquetFile(input_path)
    missing_columns = sorted(set(SOURCE_COLUMNS) - set(source.schema_arrow.names))
    if missing_columns:
        raise ValueError(f"Cleaned dataset is missing columns: {missing_columns}")

    for path in (regression_output, classification_output, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    regression_temp = regression_output.with_suffix(".parquet.tmp")
    classification_temp = classification_output.with_suffix(".parquet.tmp")
    report_temp = report_path.with_suffix(".json.tmp")
    for path in (regression_temp, classification_temp, report_temp):
        path.unlink(missing_ok=True)

    regression_writer: pq.ParquetWriter | None = None
    classification_writer: pq.ParquetWriter | None = None
    regression_counts: Counter[str] = Counter()
    classification_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    source_date_min: date | None = None
    source_date_max: date | None = None

    try:
        for batch in source.iter_batches(columns=SOURCE_COLUMNS, batch_size=batch_size):
            data = batch.to_pandas(types_mapper=pd.ArrowDtype)
            departures = data.loc[data["event_type"].eq("departure")].copy()
            if departures.empty:
                continue

            batch_min = departures["service_date"].min().date()
            batch_max = departures["service_date"].max().date()
            source_date_min = batch_min if source_date_min is None else min(source_date_min, batch_min)
            source_date_max = batch_max if source_date_max is None else max(source_date_max, batch_max)

            common = make_common_fields(departures, validation_start, test_start)

            regression_mask = departures["delay_on_departure_seconds"].notna()
            regression = common.loc[regression_mask].copy()
            regression["departure_delay_seconds"] = departures.loc[
                regression_mask, "delay_on_departure_seconds"
            ]
            regression = regression[CONTEXT_COLUMNS + PREDICTOR_COLUMNS + ["departure_delay_seconds"]]
            update_counts(regression_counts, regression, "departure_delay_seconds")
            regression_writer = write_batch(
                regression_writer, regression, regression_temp
            )

            classification_mask = departures["on_time_performance_status"].notna()
            source_status = departures.loc[
                classification_mask, "on_time_performance_status"
            ]
            unexpected_statuses = sorted(set(source_status.unique()) - set(STATUS_MAPPING))
            if unexpected_statuses:
                raise ValueError(f"Unexpected service statuses: {unexpected_statuses}")

            classification = common.loc[classification_mask].copy()
            classification["service_status"] = source_status.map(STATUS_MAPPING).astype(
                "string[pyarrow]"
            )
            classification = classification[
                CONTEXT_COLUMNS + PREDICTOR_COLUMNS + ["service_status"]
            ]
            update_counts(classification_counts, classification, "service_status")
            class_counts.update(
                {
                    str(label): int(count)
                    for label, count in classification["service_status"]
                    .value_counts()
                    .items()
                }
            )
            classification_writer = write_batch(
                classification_writer, classification, classification_temp
            )

        if regression_writer is None or classification_writer is None:
            raise RuntimeError("No eligible departure records were written")
        regression_writer.close()
        classification_writer.close()
        regression_writer = None
        classification_writer = None

        regression_output_rows = validate_output(
            regression_temp, "departure_delay_seconds", regression_counts["rows"]
        )
        classification_output_rows = validate_output(
            classification_temp, "service_status", classification_counts["rows"]
        )
        if regression_counts["missing_target"] or classification_counts["missing_target"]:
            raise RuntimeError("Generated output contains a missing target")

        os.replace(regression_temp, regression_output)
        os.replace(classification_temp, classification_output)

        report: dict[str, Any] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
            },
            "source": {
                "file": str(input_path.resolve()),
                "sha256": sha256_file(input_path),
                "departure_date_min": source_date_min.isoformat() if source_date_min else None,
                "departure_date_max": source_date_max.isoformat() if source_date_max else None,
            },
            "prediction_point": "Immediately before the scheduled departure at a stop",
            "split_boundaries": {
                "train": f"service_date < {validation_start.isoformat()}",
                "validation": (
                    f"{validation_start.isoformat()} <= service_date < "
                    f"{test_start.isoformat()}"
                ),
                "test": f"service_date >= {test_start.isoformat()}",
            },
            "context_columns_not_used_as_predictors": CONTEXT_COLUMNS,
            "categorical_features": CATEGORICAL_FEATURES,
            "numerical_features": NUMERICAL_FEATURES,
            "regression": {
                "target": "departure_delay_seconds",
                "rows": regression_counts["rows"],
                "split_rows": {
                    name: regression_counts[f"split_{name}"]
                    for name in ("train", "validation", "test")
                },
                "missing_direction_rows": regression_counts["missing_direction"],
                "output_file": str(regression_output.resolve()),
                "output_rows": regression_output_rows,
                "output_bytes": regression_output.stat().st_size,
                "output_sha256": sha256_file(regression_output),
            },
            "classification": {
                "target": "service_status",
                "source": "on_time_performance_status",
                "mapping": STATUS_MAPPING,
                "rows": classification_counts["rows"],
                "split_rows": {
                    name: classification_counts[f"split_{name}"]
                    for name in ("train", "validation", "test")
                },
                "class_rows": dict(sorted(class_counts.items())),
                "missing_direction_rows": classification_counts["missing_direction"],
                "output_file": str(classification_output.resolve()),
                "output_rows": classification_output_rows,
                "output_bytes": classification_output.stat().st_size,
                "output_sha256": sha256_file(classification_output),
            },
            "excluded_predictor_groups": {
                "outcomes_and_target_derivatives": [
                    "actual_arrival_time",
                    "actual_departure_time",
                    "actual_event_time",
                    "delay_on_departure_seconds",
                    "on_time_performance_deviation_seconds",
                    "on_time_performance_status",
                    "has_performance_measurement",
                    "has_departure_delay",
                ],
                "constant_or_uncertain_identifiers": [
                    "transit_system_name",
                    "trip_id",
                    "vehicle_id",
                ],
                "duplicate_or_descriptive_fields": ["stop_name", "event_type"],
                "provenance": ["source_archive", "source_file", "source_row_number"],
            },
        }
        report_temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(report_temp, report_path)
        return report
    finally:
        if regression_writer is not None:
            regression_writer.close()
        if classification_writer is not None:
            classification_writer.close()
        regression_temp.unlink(missing_ok=True)
        classification_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chronological regression and classification feature tables."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--regression-output", type=Path, default=DEFAULT_REGRESSION_OUTPUT)
    parser.add_argument(
        "--classification-output", type=Path, default=DEFAULT_CLASSIFICATION_OUTPUT
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--validation-start",
        type=date.fromisoformat,
        default=VALIDATION_START,
        help="First service date assigned to validation (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--test-start",
        type=date.fromisoformat,
        default=TEST_START,
        help="First service date assigned to test (YYYY-MM-DD).",
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_features(
        args.input,
        args.regression_output,
        args.classification_output,
        args.report,
        args.validation_start,
        args.test_start,
        args.batch_size,
    )
    print(
        "Regression rows: "
        f"{report['regression']['rows']:,} -> {args.regression_output}"
    )
    print(
        "Classification rows: "
        f"{report['classification']['rows']:,} -> {args.classification_output}"
    )
    print(f"Feature report: {args.report}")


if __name__ == "__main__":
    main()
