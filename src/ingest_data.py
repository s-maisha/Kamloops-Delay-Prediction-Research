"""Build the analytical source table directly from the BC Transit ZIP files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile, ZipFile, ZipInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "transit_events.parquet"
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "ingestion_manifest.json"

EXPECTED_COLUMNS = [
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

INTEGER_COLUMNS = [
    "stop_sequence",
    "delay_on_departure_seconds",
    "on_time_performance_deviation_seconds",
]


def sha256_file(path: Path) -> str:
    """Return a SHA-256 checksum without loading the whole file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_members(archive: ZipFile) -> list[ZipInfo]:
    """Return readable CSV members after rejecting encrypted source data."""
    members = [
        member
        for member in archive.infolist()
        if not member.is_dir() and member.filename.lower().endswith(".csv")
    ]
    if not members:
        raise ValueError(f"No CSV files found in {archive.filename}")
    encrypted = [member.filename for member in members if member.flag_bits & 0x1]
    if encrypted:
        raise ValueError(f"Encrypted CSV members are not supported: {encrypted}")
    return sorted(members, key=lambda member: member.filename.casefold())


def read_header(archive: ZipFile, member: ZipInfo) -> list[str]:
    """Read and validate a CSV header before processing its records."""
    with archive.open(member) as binary:
        text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
        header = next(csv.reader(text), None)
    if header != EXPECTED_COLUMNS:
        raise ValueError(
            f"Unexpected schema in {member.filename}. "
            f"Expected {EXPECTED_COLUMNS}, found {header}."
        )
    return header


def parse_integer_column(
    values: pd.Series, column: str, source_name: str, first_source_row: int
) -> pd.Series:
    """Parse an integer field and report source rows that contain invalid values."""
    parsed = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & parsed.isna()
    fractional = parsed.notna() & parsed.mod(1).ne(0)
    invalid = invalid | fractional
    if invalid.any():
        sample_rows = (invalid[invalid].index - values.index[0] + first_source_row).tolist()[:5]
        raise ValueError(
            f"Invalid integer values in {source_name}, column {column}, "
            f"source rows {sample_rows}."
        )
    return parsed.astype("Int64")


def parse_datetime_column(
    values: pd.Series,
    column: str,
    source_name: str,
    first_source_row: int,
    *,
    format_string: str,
    utc: bool = False,
) -> pd.Series:
    """Parse a date/time field without treating missing values as malformed."""
    parsed = pd.to_datetime(values, format=format_string, errors="coerce", utc=utc)
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        sample_rows = (invalid[invalid].index - values.index[0] + first_source_row).tolist()[:5]
        raise ValueError(
            f"Invalid date/time values in {source_name}, column {column}, "
            f"source rows {sample_rows}."
        )
    return parsed


def prepare_chunk(
    chunk: pd.DataFrame,
    archive_name: str,
    member_name: str,
    first_source_row: int,
) -> pd.DataFrame:
    """Apply explicit types while retaining the source column meanings."""
    source_name = f"{archive_name}::{member_name}"
    chunk["service_date"] = parse_datetime_column(
        chunk["service_date"],
        "service_date",
        source_name,
        first_source_row,
        format_string="%m/%d/%Y",
    )
    for column in TIMESTAMP_COLUMNS:
        chunk[column] = parse_datetime_column(
            chunk[column],
            column,
            source_name,
            first_source_row,
            format_string="%Y-%m-%dT%H:%M:%S.%fZ",
            utc=True,
        )
    for column in INTEGER_COLUMNS:
        chunk[column] = parse_integer_column(
            chunk[column], column, source_name, first_source_row
        )

    # Identifiers remain strings so values such as scientific notation are not rounded.
    chunk.insert(0, "source_archive", archive_name)
    chunk.insert(1, "source_file", Path(member_name).name)
    chunk.insert(
        2,
        "source_row_number",
        pd.array(range(first_source_row, first_source_row + len(chunk)), dtype="Int64"),
    )
    return chunk


def ingest(
    raw_dir: Path,
    output_path: Path,
    manifest_path: Path,
    chunk_size: int,
) -> dict[str, object]:
    """Validate all sources and write one atomic Parquet dataset."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")

    zip_paths = sorted(raw_dir.glob("*.zip"), key=lambda path: path.name.casefold())
    if not zip_paths:
        raise FileNotFoundError(f"No ZIP archives found in {raw_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)
    temporary_manifest.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    file_records: list[dict[str, object]] = []
    archive_records: list[dict[str, object]] = []
    total_rows = 0
    overall_min_date: pd.Timestamp | None = None
    overall_max_date: pd.Timestamp | None = None

    try:
        for zip_path in zip_paths:
            archive_record: dict[str, object] = {
                "archive": zip_path.name,
                "compressed_bytes": zip_path.stat().st_size,
                "sha256": sha256_file(zip_path),
            }
            with ZipFile(zip_path) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise BadZipFile(f"CRC check failed for {zip_path.name}::{bad_member}")
                members = csv_members(archive)
                archive_record["csv_files"] = len(members)

                for member in members:
                    read_header(archive, member)
                    member_rows = 0
                    member_min_date: pd.Timestamp | None = None
                    member_max_date: pd.Timestamp | None = None

                    with archive.open(member) as binary:
                        chunks = pd.read_csv(
                            binary,
                            dtype="string",
                            encoding="utf-8-sig",
                            chunksize=chunk_size,
                            on_bad_lines="error",
                        )
                        for chunk in chunks:
                            first_source_row = member_rows + 2
                            chunk = prepare_chunk(
                                chunk,
                                zip_path.name,
                                member.filename,
                                first_source_row,
                            )
                            chunk_min = chunk["service_date"].min()
                            chunk_max = chunk["service_date"].max()
                            member_min_date = (
                                chunk_min
                                if member_min_date is None
                                else min(member_min_date, chunk_min)
                            )
                            member_max_date = (
                                chunk_max
                                if member_max_date is None
                                else max(member_max_date, chunk_max)
                            )

                            table = pa.Table.from_pandas(chunk, preserve_index=False)
                            if writer is None:
                                writer = pq.ParquetWriter(
                                    temporary_output,
                                    table.schema,
                                    compression="snappy",
                                )
                            elif table.schema != writer.schema:
                                raise ValueError(
                                    f"Converted schema changed while reading {member.filename}"
                                )
                            writer.write_table(table)
                            member_rows += len(chunk)
                            total_rows += len(chunk)

                    overall_min_date = (
                        member_min_date
                        if overall_min_date is None
                        else min(overall_min_date, member_min_date)
                    )
                    overall_max_date = (
                        member_max_date
                        if overall_max_date is None
                        else max(overall_max_date, member_max_date)
                    )
                    file_records.append(
                        {
                            "archive": zip_path.name,
                            "source_file": Path(member.filename).name,
                            "uncompressed_bytes": member.file_size,
                            "rows": member_rows,
                            "minimum_service_date": member_min_date.date().isoformat(),
                            "maximum_service_date": member_max_date.date().isoformat(),
                        }
                    )
                    print(f"Validated {member.filename}: {member_rows:,} rows")
            archive_records.append(archive_record)

        if writer is None:
            raise RuntimeError("No records were written")
        writer.close()
        writer = None
        os.replace(temporary_output, output_path)

        parquet_file = pq.ParquetFile(output_path)
        if parquet_file.metadata.num_rows != total_rows:
            raise RuntimeError(
                f"Parquet row count {parquet_file.metadata.num_rows:,} does not match "
                f"source row count {total_rows:,}"
            )

        manifest: dict[str, object] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "software": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
            },
            "chunk_size": chunk_size,
            "parquet_compression": "snappy",
            "raw_directory": str(raw_dir.resolve()),
            "output_file": str(output_path.resolve()),
            "output_sha256": sha256_file(output_path),
            "output_bytes": output_path.stat().st_size,
            "row_count": total_rows,
            "column_count": len(parquet_file.schema_arrow.names),
            "minimum_service_date": overall_min_date.date().isoformat(),
            "maximum_service_date": overall_max_date.date().isoformat(),
            "archives": archive_records,
            "files": file_records,
            "columns": parquet_file.schema_arrow.names,
        }
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_manifest, manifest_path)
        return manifest
    except Exception:
        if writer is not None:
            writer.close()
        temporary_output.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate BC Transit source ZIPs and build a Parquet dataset."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = ingest(
        raw_dir=args.raw_dir,
        output_path=args.output,
        manifest_path=args.manifest,
        chunk_size=args.chunk_size,
    )
    print(
        f"Wrote {manifest['row_count']:,} rows to {manifest['output_file']} "
        f"({manifest['output_bytes']:,} bytes)."
    )


if __name__ == "__main__":
    main()
