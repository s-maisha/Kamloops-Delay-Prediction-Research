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

## Data availability

The repository does not redistribute the original BC Transit archives. The code and documentation will describe how to reproduce derived results after obtaining the source files separately.
