"""Create a conservative analytical dataset without altering supplied values."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "transit_events.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "transit_events_clean.parquet"
DEFAULT_REPORT = PROJECT_ROOT / "data" / "processed" / "cleaning_report.json"

INPUT_COLUMNS = [
    "source_archive",
    "source_file",
    "source_row_number",
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

ADDED_COLUMNS = [
    "event_type",
    "scheduled_event_time",
    "actual_event_time",
    "is_after_midnight_service",
    "has_performance_measurement",
    "has_departure_delay",
]

TEXT_COLUMNS = [
    "transit_system_name",
    "route_name",
    "direction_code",
    "trip_id",
    "vehicle_id",
    "stop_id",
    "stop_name",
    "on_time_performance_status",
]


def sha256_file(path: Path) -> str:
    """Return a SHA-256 checksum without loading the complete file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_status(deviation: pd.Series) -> pd.Series:
    """Apply the status boundaries demonstrated by every labelled source row."""
    status = pd.Series(pd.NA, index=deviation.index, dtype="string[pyarrow]")
    status.loc[deviation.le(-180).fillna(False)] = "Very Early"
    status.loc[deviation.between(-179, -60).fillna(False)] = "Early"
    status.loc[deviation.between(-59, 180).fillna(False)] = "On Time"
    status.loc[deviation.between(181, 360).fillna(False)] = "Late"
    status.loc[deviation.ge(361).fillna(False)] = "Very Late"
    return status


def validate_text_fields(data: pd.DataFrame) -> None:
    """Reject unexpected text formatting instead of silently changing labels."""
    for column in TEXT_COLUMNS:
        non_missing = data[column].dropna()
        if non_missing.ne(non_missing.str.strip()).any():
            raise ValueError(f"Unexpected surrounding whitespace in {column}")
        if non_missing.str.contains(r"[\r\n\t]", regex=True, na=False).any():
            raise ValueError(f"Unexpected control whitespace in {column}")


def prepare_batch(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate one batch and append analysis-oriented event fields."""
    validate_text_fields(data)

    arrival_event = data["scheduled_arrival_time"].notna()
    departure_event = data["scheduled_departure_time"].notna()
    invalid_schedule_pair = arrival_event.eq(departure_event)
    if invalid_schedule_pair.any():
        raise ValueError(
            f"Found {int(invalid_schedule_pair.sum())} rows without exactly one "
            "scheduled event timestamp"
        )

    data["event_type"] = pd.Series(
        arrival_event.map({True: "arrival", False: "departure"}),
        index=data.index,
        dtype="string[pyarrow]",
    )
    data["scheduled_event_time"] = data["scheduled_departure_time"].where(
        departure_event, data["scheduled_arrival_time"]
    )
    data["actual_event_time"] = data["actual_departure_time"].where(
        departure_event, data["actual_arrival_time"]
    )

    deviation = data["on_time_performance_deviation_seconds"]
    reported_status = data["on_time_performance_status"].astype("string[pyarrow]")
    if not deviation.isna().equals(reported_status.isna()):
        raise ValueError("Performance deviation and status have inconsistent missingness")
    if not deviation.isna().equals(data["actual_event_time"].isna()):
        raise ValueError("Performance deviation and matching actual event have inconsistent missingness")

    derived_status = expected_status(deviation)
    status_mismatch = (
        derived_status.fillna("<MISSING>") != reported_status.fillna("<MISSING>")
    )
    if status_mismatch.any():
        raise ValueError(
            f"Found {int(status_mismatch.sum())} records inconsistent with status thresholds"
        )

    comparable = data["actual_event_time"].notna()
    calculated_deviation = (
        data.loc[comparable, "actual_event_time"]
        - data.loc[comparable, "scheduled_event_time"]
    ).dt.total_seconds()
    timestamp_mismatch = calculated_deviation.ne(deviation.loc[comparable])
    if timestamp_mismatch.any():
        raise ValueError(
            f"Found {int(timestamp_mismatch.sum())} timestamp/deviation mismatches"
        )

    # Strip the UTC type without shifting clock values to compare calendar dates.
    scheduled_without_zone = data["scheduled_event_time"].astype(
        pd.ArrowDtype(pa.timestamp("us"))
    )
    day_offset = (
        scheduled_without_zone.dt.normalize() - data["service_date"]
    ).dt.days
    invalid_offset = ~day_offset.isin([0, 1])
    if invalid_offset.any():
        raise ValueError(
            f"Found {int(invalid_offset.sum())} scheduled events outside service day or following day"
        )

    data["is_after_midnight_service"] = day_offset.eq(1)
    data["has_performance_measurement"] = deviation.notna()
    data["has_departure_delay"] = data["delay_on_departure_seconds"].notna()

    counts = {
        "arrival_events": int(arrival_event.sum()),
        "departure_events": int(departure_event.sum()),
        "after_midnight_events": int(data["is_after_midnight_service"].sum()),
        "performance_measurements_available": int(
            data["has_performance_measurement"].sum()
        ),
        "departure_delays_available": int(data["has_departure_delay"].sum()),
        "rows_with_missing_direction": int(data["direction_code"].isna().sum()),
        "rows_with_actual_departure_but_missing_departure_delay": int(
            (
                data["actual_departure_time"].notna()
                & data["delay_on_departure_seconds"].isna()
            ).sum()
        ),
    }
    return data, counts


def add_counts(total: dict[str, int], current: dict[str, int]) -> None:
    for key, value in current.items():
        total[key] = total.get(key, 0) + value


def clean(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    batch_size: int,
) -> dict[str, Any]:
    """Create and verify the cleaned analytical table using atomic outputs."""
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not input_path.exists():
        raise FileNotFoundError(f"Ingested dataset not found: {input_path}")

    source = pq.ParquetFile(input_path)
    if source.schema_arrow.names != INPUT_COLUMNS:
        raise ValueError(
            "Unexpected input columns. "
            f"Expected {INPUT_COLUMNS}, found {source.schema_arrow.names}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)
    temporary_report.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    total_rows = 0
    flag_counts: dict[str, int] = {}

    try:
        for batch in source.iter_batches(batch_size=batch_size):
            data = batch.to_pandas(types_mapper=pd.ArrowDtype)
            data, batch_counts = prepare_batch(data)
            add_counts(flag_counts, batch_counts)
            table = pa.Table.from_pandas(data, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_output, table.schema, compression="snappy"
                )
            elif writer.schema != table.schema:
                raise ValueError("Cleaned schema changed between processing batches")
            writer.write_table(table)
            total_rows += len(data)

        if writer is None:
            raise RuntimeError("No records were written")
        writer.close()
        writer = None
        os.replace(temporary_output, output_path)

        cleaned = pq.ParquetFile(output_path)
        if cleaned.metadata.num_rows != source.metadata.num_rows:
            raise RuntimeError("Cleaned output row count does not match ingested input")
        if cleaned.schema_arrow.names != INPUT_COLUMNS + ADDED_COLUMNS:
            raise RuntimeError("Cleaned output columns do not match the expected schema")

        report: dict[str, Any] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
            },
            "batch_size": batch_size,
            "input_file": str(input_path.resolve()),
            "input_sha256": sha256_file(input_path),
            "input_rows": source.metadata.num_rows,
            "input_columns": source.metadata.num_columns,
            "output_file": str(output_path.resolve()),
            "output_sha256": sha256_file(output_path),
            "output_bytes": output_path.stat().st_size,
            "output_rows": cleaned.metadata.num_rows,
            "output_columns": cleaned.metadata.num_columns,
            "rows_removed": source.metadata.num_rows - cleaned.metadata.num_rows,
            "added_columns": ADDED_COLUMNS,
            "flag_counts": flag_counts,
            "validated_status_threshold_mismatches": 0,
            "validated_timestamp_deviation_mismatches": 0,
            "text_values_modified": 0,
            "decisions": [
                "Retained every ingested record.",
                "Preserved all supplied columns and values.",
                "Did not impute missing direction or timestamp values.",
                "Did not remove or cap extreme delay values.",
                "Retained the unusable trip_id for source traceability only.",
                "Added canonical event fields and availability flags.",
            ],
        }
        temporary_report.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_report, report_path)
        return report
    except Exception:
        if writer is not None:
            writer.close()
        temporary_output.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and prepare the cleaned Kamloops transit dataset."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = clean(args.input, args.output, args.report, args.batch_size)
    print(
        f"Wrote {report['output_rows']:,} rows and {report['output_columns']} columns "
        f"to {report['output_file']}."
    )
    print(f"Rows removed: {report['rows_removed']:,}")


if __name__ == "__main__":
    main()
