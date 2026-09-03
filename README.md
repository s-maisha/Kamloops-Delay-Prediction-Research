# Kamloops Delay Prediction Research

This project examines delay patterns in six months of official BC Transit operational data for the Kamloops transit system. The planned workflow includes reproducible data preparation, exploratory analysis, numerical delay prediction, service-status classification, route-level clustering, and a focused time-series forecasting case study.

## Data

The source delivery contains 3,621,203 stop-level records covering January 1 through June 30, 2026. The twelve source CSV files are distributed across three ZIP archives and share a common 16-column schema.

The original BC Transit files are preserved locally in `data/raw/`. They are excluded from version control because the supplied archives contain no redistribution terms. To reproduce the project, place the three original ZIP files in `data/raw/` before running the ingestion workflow.

## Current structure

```text
data/
  raw/          Original source archives; not tracked by Git
  processed/    Reproducible analytical datasets; not tracked by Git
src/            Reusable data-processing code
```

Additional folders will be added as the corresponding analysis stages are developed.

## Environment setup

Python 3.11 or later is recommended. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The ingestion and analysis commands will be documented as they are implemented.

## Data ingestion

Place the three original ZIP archives in `data/raw/`, then run:

```bash
python src/ingest_data.py
```

The script reads CSV records directly from the archives, checks source integrity and schema compatibility, parses dates and timestamps, and writes `data/processed/transit_events.parquet`. It also writes `data/processed/ingestion_manifest.json` with source checksums, row counts, date coverage, and output metadata. Both generated files are excluded from version control and can be recreated from the source archives.

## Dataset profiling

After ingestion, generate the structural and data-quality profile with:

```bash
python src/profile_data.py
```

The machine-readable profile is written to `data/processed/data_profile.json`. The principal findings and field definitions are documented in `docs/DATA_PROFILE.md` and `docs/DATA_DICTIONARY.md`.

## Data cleaning

After profiling, create the cleaned analytical table with:

```bash
python src/clean_data.py
```

The script validates event structure and status consistency, preserves all supplied records and values, and adds canonical event timestamps and data-availability flags. It writes `data/processed/transit_events_clean.parquet` and `data/processed/cleaning_report.json`. Cleaning decisions are documented in `docs/CLEANING.md`.

## Exploratory analysis

After cleaning, regenerate the exploratory tables and figures with:

```bash
python src/run_eda.py
```

The analysis writes compact CSV summaries to `outputs/tables/` and figures to `outputs/figures/`. The complete narrative and limitations are documented in `docs/EDA.md`, while `notebooks/02_eda.ipynb` provides an executable question–method–result–interpretation walkthrough.

## Feature and target preparation

Create the leakage-safe regression and classification tables with:

```bash
python src/features.py
```

The workflow selects departure events, derives schedule-known predictors, and assigns fixed chronological training, validation, and test sets. Target definitions, feature decisions, exclusions, and the exact date boundaries are documented in `docs/METHODS.md`. Generated feature tables remain excluded from version control.

## Data availability

The repository does not redistribute the original BC Transit archives. The code and documentation will describe how to reproduce derived results after obtaining the source files separately.
