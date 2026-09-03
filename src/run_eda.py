"""Generate focused exploratory summaries and figures from the cleaned data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "transit_events_clean.parquet"
DEFAULT_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

STATUS_ORDER = ["Very Early", "Early", "On Time", "Late", "Very Late"]
STATUS_COLORS = ["#2F6690", "#6FA8DC", "#6A994E", "#F4A261", "#C44536"]
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_ORDER = ["January", "February", "March", "April", "May", "June"]

READ_COLUMNS = [
    "source_file",
    "source_row_number",
    "service_date",
    "route_name",
    "stop_id",
    "stop_name",
    "event_type",
    "scheduled_event_time",
    "is_after_midnight_service",
    "has_performance_measurement",
    "has_departure_delay",
    "delay_on_departure_seconds",
    "on_time_performance_deviation_seconds",
    "on_time_performance_status",
]


def route_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def add_time_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Derive descriptive calendar fields without shifting supplied clock values."""
    scheduled_clock = data["scheduled_event_time"].astype(
        pd.ArrowDtype(pa.timestamp("us"))
    )
    data["scheduled_hour"] = scheduled_clock.dt.hour.astype("int16")
    data["weekday_number"] = data["service_date"].dt.dayofweek.astype("int8")
    data["weekday"] = data["weekday_number"].map(dict(enumerate(WEEKDAY_ORDER)))
    data["month_number"] = data["service_date"].dt.month.astype("int8")
    data["month"] = data["month_number"].map(
        {number: month for number, month in enumerate(MONTH_ORDER, start=1)}
    )
    data["period_group"] = data["weekday_number"].ge(5).map(
        {True: "Weekend", False: "Weekday"}
    )
    return data


def add_status_groups(data: pd.DataFrame) -> pd.DataFrame:
    status = data["on_time_performance_status"]
    data["is_early"] = status.isin(["Very Early", "Early"])
    data["is_on_time"] = status.eq("On Time")
    data["is_late"] = status.isin(["Late", "Very Late"])
    return data


def summarize_groups(data: pd.DataFrame, groups: Sequence[str]) -> pd.DataFrame:
    summary = (
        data.groupby(list(groups), observed=True, dropna=False)
        .agg(
            records=("on_time_performance_deviation_seconds", "size"),
            mean_deviation_seconds=("on_time_performance_deviation_seconds", "mean"),
            median_deviation_seconds=("on_time_performance_deviation_seconds", "median"),
            std_deviation_seconds=("on_time_performance_deviation_seconds", "std"),
            p90_deviation_seconds=(
                "on_time_performance_deviation_seconds",
                lambda values: values.quantile(0.90),
            ),
            p95_deviation_seconds=(
                "on_time_performance_deviation_seconds",
                lambda values: values.quantile(0.95),
            ),
            early_percent=("is_early", lambda values: 100 * values.mean()),
            on_time_percent=("is_on_time", lambda values: 100 * values.mean()),
            late_percent=("is_late", lambda values: 100 * values.mean()),
        )
        .reset_index()
    )
    numeric_columns = summary.select_dtypes(include="number").columns.difference(
        ["records"]
    )
    summary[numeric_columns] = summary[numeric_columns].astype(float).round(3)
    return summary


def build_tables(data: pd.DataFrame, table_dir: Path) -> dict[str, pd.DataFrame]:
    performance = data.loc[data["has_performance_measurement"]].copy()
    performance = add_status_groups(performance)

    overall = pd.DataFrame(
        [
            {
                "total_clean_records": len(data),
                "performance_records": len(performance),
                "departure_delay_records": int(data["has_departure_delay"].sum()),
                "after_midnight_records": int(data["is_after_midnight_service"].sum()),
                "mean_deviation_seconds": round(
                    float(performance["on_time_performance_deviation_seconds"].mean()), 3
                ),
                "median_deviation_seconds": round(
                    float(performance["on_time_performance_deviation_seconds"].median()), 3
                ),
                "p90_deviation_seconds": round(
                    float(performance["on_time_performance_deviation_seconds"].quantile(0.90)), 3
                ),
                "p95_deviation_seconds": round(
                    float(performance["on_time_performance_deviation_seconds"].quantile(0.95)), 3
                ),
                "early_percent": round(100 * float(performance["is_early"].mean()), 3),
                "on_time_percent": round(100 * float(performance["is_on_time"].mean()), 3),
                "late_percent": round(100 * float(performance["is_late"].mean()), 3),
            }
        ]
    )

    status = (
        performance["on_time_performance_status"]
        .value_counts()
        .reindex(STATUS_ORDER)
        .rename_axis("status")
        .reset_index(name="records")
    )
    status["percent"] = (100 * status["records"] / len(performance)).round(3)

    route = summarize_groups(performance, ["route_name"])
    route["route_sort"] = route["route_name"].map(
        lambda value: route_sort_key(value)[1]
    )
    route = route.sort_values("route_sort").drop(columns="route_sort")

    hourly = summarize_groups(performance, ["scheduled_hour"]).sort_values(
        "scheduled_hour"
    )
    weekday = summarize_groups(
        performance, ["weekday_number", "weekday", "period_group"]
    ).sort_values("weekday_number")
    period = summarize_groups(performance, ["period_group"]).sort_values(
        "period_group"
    )
    monthly = summarize_groups(
        performance, ["month_number", "month"]
    ).sort_values("month_number")
    event_type = summarize_groups(performance, ["event_type"]).sort_values(
        "event_type"
    )
    stop = summarize_groups(performance, ["stop_id", "stop_name"]).sort_values(
        ["records", "stop_id"], ascending=[False, True]
    )

    extreme_columns = [
        "source_file",
        "source_row_number",
        "service_date",
        "route_name",
        "stop_id",
        "stop_name",
        "event_type",
        "on_time_performance_deviation_seconds",
        "on_time_performance_status",
    ]
    earliest = performance.nsmallest(
        10, "on_time_performance_deviation_seconds"
    )[extreme_columns].assign(extreme_type="most_early")
    latest = performance.nlargest(
        10, "on_time_performance_deviation_seconds"
    )[extreme_columns].assign(extreme_type="most_late")
    extremes = pd.concat([earliest, latest], ignore_index=True)

    tables = {
        "eda_overall_summary.csv": overall,
        "eda_status_summary.csv": status,
        "eda_route_summary.csv": route,
        "eda_hourly_summary.csv": hourly,
        "eda_weekday_summary.csv": weekday,
        "eda_period_summary.csv": period,
        "eda_monthly_summary.csv": monthly,
        "eda_event_type_summary.csv": event_type,
        "eda_stop_summary.csv": stop,
        "eda_extreme_records.csv": extremes,
    }
    table_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in tables.items():
        table.to_csv(table_dir / filename, index=False)
    return tables


def save_delay_distribution(data: pd.DataFrame, figure_dir: Path) -> None:
    deviation = data.loc[
        data["has_performance_measurement"],
        "on_time_performance_deviation_seconds",
    ].astype(float)
    lower, upper = deviation.quantile([0.005, 0.995])
    visible = deviation.between(lower, upper)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(deviation.loc[visible], bins=90, color="#3B6C8E", edgecolor="none")
    for boundary, label in [
        (-180, "Very early"),
        (-60, "Early"),
        (180, "Late"),
        (360, "Very late"),
    ]:
        ax.axvline(boundary, color="#333333", linestyle="--", linewidth=1)
        ax.text(boundary, ax.get_ylim()[1] * 0.96, label, rotation=90, va="top", ha="right", fontsize=8)
    ax.set(
        title="Distribution of on-time performance deviation",
        xlabel="Actual minus scheduled event time (seconds)",
        ylabel="Records",
        xlim=(lower, upper),
    )
    ax.text(
        0.99,
        0.88,
        "Display limited to the 0.5th–99.5th percentiles;\nall records remain in the analysis.",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_delay_distribution.png", dpi=160)
    plt.close(fig)


def save_status_distribution(status: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(status["status"], status["percent"], color=STATUS_COLORS)
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in status["percent"]], padding=3)
    ax.set(title="Official performance status distribution", ylabel="Percent of measured events", ylim=(0, 72))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_status_distribution.png", dpi=160)
    plt.close(fig)


def save_route_performance(route: pd.DataFrame, figure_dir: Path) -> None:
    plotted = route.sort_values("median_deviation_seconds")
    route_labels = [
        f"{route_name} (n={records:,})"
        for route_name, records in zip(plotted["route_name"], plotted["records"])
    ]
    y = np.arange(len(plotted))
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharey=True)
    axes[0].barh(y, plotted["median_deviation_seconds"], color="#3B6C8E")
    axes[0].set_yticks(y, labels=route_labels)
    axes[0].set(title="Median deviation", xlabel="Seconds", ylabel="Route")
    axes[1].barh(y, plotted["on_time_percent"], color="#6A994E")
    axes[1].set(title="Official On Time share", xlabel="Percent", xlim=(0, 100))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Route-level performance measures")
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_route_performance.png", dpi=160)
    plt.close(fig)


def save_hourly_pattern(hourly: pd.DataFrame, figure_dir: Path) -> None:
    plotted = hourly.set_index("scheduled_hour").reindex(range(24))
    x = np.arange(24)
    median_values = plotted["median_deviation_seconds"].to_numpy(
        dtype=float, na_value=np.nan
    )
    on_time_values = plotted["on_time_percent"].to_numpy(
        dtype=float, na_value=np.nan
    )
    record_values = plotted["records"].to_numpy(dtype=float, na_value=0)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(x, median_values, marker="o", color="#3B6C8E")
    axes[0].set(title="Median performance deviation by scheduled hour", ylabel="Seconds")
    axes[1].plot(x, on_time_values, marker="o", color="#6A994E")
    axes[1].set(title="Official On Time share by scheduled hour", ylabel="Percent")
    axes[2].bar(x, record_values, color="#888888")
    axes[2].set(title="Measured event count by scheduled hour", xlabel="Scheduled clock hour", ylabel="Records", xticks=range(24))
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_hourly_pattern.png", dpi=160)
    plt.close(fig)


def save_calendar_patterns(
    weekday: pd.DataFrame, monthly: pd.DataFrame, figure_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(weekday["weekday"], weekday["median_deviation_seconds"], color="#3B6C8E")
    axes[0, 0].set(title="Median deviation by service weekday", ylabel="Seconds")
    axes[0, 1].bar(weekday["weekday"], weekday["on_time_percent"], color="#6A994E")
    axes[0, 1].set(title="On Time share by service weekday", ylabel="Percent")
    axes[1, 0].plot(monthly["month"], monthly["median_deviation_seconds"], marker="o", color="#3B6C8E")
    axes[1, 0].set(title="Median deviation by month", ylabel="Seconds")
    axes[1, 1].plot(monthly["month"], monthly["on_time_percent"], marker="o", color="#6A994E")
    axes[1, 1].set(title="On Time share by month", ylabel="Percent")
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=35)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_calendar_patterns.png", dpi=160)
    plt.close(fig)


def save_stop_patterns(stop: pd.DataFrame, figure_dir: Path) -> None:
    eligible = stop.loc[stop["records"] >= 1_000].copy()
    plotted = eligible.nlargest(15, "median_deviation_seconds").sort_values(
        "median_deviation_seconds"
    )
    labels = plotted["stop_name"] + " [" + plotted["stop_id"] + "]"
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(labels, plotted["median_deviation_seconds"], color="#C44536")
    ax.set(
        title="Stops with highest median deviation (at least 1,000 measured events)",
        xlabel="Median deviation (seconds)",
        ylabel="Stop name and ID",
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_stop_patterns.png", dpi=160)
    plt.close(fig)


def run_eda(input_path: Path, table_dir: Path, figure_dir: Path) -> dict[str, pd.DataFrame]:
    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {input_path}")
    data = pd.read_parquet(input_path, columns=READ_COLUMNS, dtype_backend="pyarrow")
    data = add_time_fields(data)
    tables = build_tables(data, table_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_delay_distribution(data, figure_dir)
    save_status_distribution(tables["eda_status_summary.csv"], figure_dir)
    save_route_performance(tables["eda_route_summary.csv"], figure_dir)
    save_hourly_pattern(tables["eda_hourly_summary.csv"], figure_dir)
    save_calendar_patterns(
        tables["eda_weekday_summary.csv"],
        tables["eda_monthly_summary.csv"],
        figure_dir,
    )
    save_stop_patterns(tables["eda_stop_summary.csv"], figure_dir)
    return tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused transit-delay EDA.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = run_eda(args.input, args.table_dir, args.figure_dir)
    print(f"Wrote {len(tables)} EDA tables to {args.table_dir.resolve()}")
    print(f"Wrote 6 EDA figures to {args.figure_dir.resolve()}")


if __name__ == "__main__":
    main()
