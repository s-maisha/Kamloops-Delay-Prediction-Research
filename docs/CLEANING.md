# Data Cleaning and Analytical Preparation

The cleaning workflow is implemented in `src/clean_data.py`. It reads the ingested Parquet table in 100,000-row batches and writes `data/processed/transit_events_clean.parquet`. The process is deliberately conservative because profiling found no malformed rows or exact duplicates and provided no evidence that extreme delay observations were invalid.

## Record retention

All 3,621,203 ingested records are retained. The 19 ingested columns are reproduced without changes, and six analytical fields are appended. An independent batch comparison confirmed that every original value remains identical between the ingested and cleaned tables.

No records were removed for missing direction, missing target values, or extreme delays. Different analyses require different subsets: numerical departure-delay modelling requires a recorded departure delay, while performance-status classification requires a recorded status. Removing every record that is incomplete for either task would unnecessarily discard observations usable elsewhere.

## Validation rules

The cleaning script stops with an error if any of the following conditions occur:

- The input schema differs from the expected 19-column ingested schema.
- A record does not contain exactly one scheduled event type.
- Performance deviation, performance status, and the matching actual event have inconsistent missingness.
- A status disagrees with the empirically observed deviation thresholds.
- Performance deviation disagrees with actual event time minus scheduled event time.
- A scheduled event falls outside its service date or the immediately following calendar date.
- Major text fields contain surrounding or control whitespace.

The completed run found zero status-threshold mismatches and zero timestamp-deviation mismatches.

## Cleaning decisions

### Source values

No source values are overwritten. The raw delay measures, statuses, identifiers, stop names, directions, and timestamps remain available in their supplied form.

### Missing values

Missing directions and timestamps are retained as missing. They are not filled with the most common category or an invented value. Target-availability flags allow later analyses to select records explicitly.

### Extreme delays

Extreme delay values are neither removed nor capped. Statistical extremeness alone does not demonstrate a data error, and large values can represent real disruptions. Their influence will be evaluated during exploratory analysis and model assessment.

### Constant trip identifier

The constant `trip_id` field is retained for source traceability but is not suitable for grouping or modelling. Removing it from the master cleaned table would obscure a limitation of the supplied data without improving data quality.

### Text fields

No surrounding whitespace, empty strings, tabs, or embedded line breaks were found in the inspected text fields. Consequently, route codes, directions, identifiers, stop names, and statuses were not rewritten.

## Added analytical fields

`event_type` identifies whether the record uses its scheduled arrival or scheduled departure field. `scheduled_event_time` and `actual_event_time` provide one consistent timestamp pair for either event type. Their difference is the supplied performance deviation for every record where the measurement exists.

`is_after_midnight_service` distinguishes 72,830 observations whose scheduled calendar date is one day after their assigned service date. `has_performance_measurement` identifies the 3,618,099 records available for performance-status analysis, and `has_departure_delay` identifies the 3,470,591 records with a numerical departure-delay target.

These flags describe data availability. They are not model predictions and do not replace the original fields.

## Output validation

The cleaned output contains 3,621,203 rows and 25 columns. Its SHA-256 checksum is `ef603733d39c65e755de8d5fd3462d7fa94d43ad3459623d5ef129e0fc81409a`. Two complete cleaning runs produced the same checksum.
