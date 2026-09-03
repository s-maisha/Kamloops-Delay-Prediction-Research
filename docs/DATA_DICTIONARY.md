# Data Dictionary

The analytical source table is generated from the original BC Transit ZIP archives by `src/ingest_data.py`. It contains 16 supplied fields and three provenance fields added during ingestion. Descriptions that are not supported by an accompanying source dictionary are limited to interpretations demonstrated by the observed values and relationships.

## Provenance fields

| Column | Stored type | Description | Example | Analytical role and limitations |
|---|---|---|---|---|
| `source_archive` | String | Name of the ZIP archive containing the source record. | `File 1- BCT Kamloops_Transit Data for Request # PI-2215.zip` | Audit field; excluded from predictive features. |
| `source_file` | String | Name of the CSV member containing the source record. | `BCT Kamp_Jan 01-15, 2026.csv` | Audit field; excluded from predictive features. |
| `source_row_number` | Integer | Original CSV row number, including the header as row 1. | `2` | Identifies a record within its source file; excluded from predictive features. |

## Supplied BC Transit fields

| Column | Stored type | Description | Example | Analytical role and limitations |
|---|---|---|---|---|
| `service_date` | Datetime | Operating service date assigned to the record. | `2026-01-01` | Primary date field. After-midnight events can have a timestamp on the following calendar date. |
| `transit_system_name` | String | Transit system associated with the record. | `Kamloops` | Constant in this delivery and therefore not informative as a model feature. |
| `route_name` | String | Route code supplied for the stop event. | `13` | Candidate categorical feature. Stored as text because route codes are labels rather than quantities. |
| `direction_code` | String | Supplied direction category. | `Inbound` | Candidate categorical feature; 1.068% of records are missing. Observed values are `Inbound`, `Outbound`, `Counterclockwise`, `West`, and `East`. |
| `trip_id` | String | Supplied trip identifier. | `2.03E+15` | Unusable as an identifier in this delivery because every record contains the same scientific-notation value. Full original identifier precision cannot be recovered. |
| `vehicle_id` | String | Identifier for the recorded transit vehicle. | `5002` | Potential categorical or grouping field. Its operational interpretation is not documented in the source archives. |
| `stop_id` | String | Identifier for the recorded stop. | `104625` | Candidate categorical or grouping field. Each observed stop ID maps to one stop name in this delivery. |
| `stop_name` | String | Human-readable stop name. | `Kamloops Transit Centre` | Descriptive and potential grouping field. A shared name can represent more than one stop ID. |
| `stop_sequence` | Integer | Supplied stop position within the service sequence. | `1` | Potential feature representing progress through a service pattern. The trip identifier cannot be used to reconstruct individual trips. |
| `scheduled_arrival_time` | UTC datetime | Scheduled arrival timestamp for an arrival-type record. | `2026-04-01T08:40:00Z` | Mutually exclusive with `scheduled_departure_time` in this dataset. Missing for 95.928% of records by design. |
| `scheduled_departure_time` | UTC datetime | Scheduled departure timestamp for a departure-type record. | `2026-02-01T07:56:00Z` | Mutually exclusive with `scheduled_arrival_time`. Missing for 4.072% of records. |
| `actual_arrival_time` | UTC datetime | Recorded actual arrival timestamp when available. | `2026-02-01T08:35:46Z` | Missing for 59.680% of records. Actual outcome information must not be used as a pre-event predictive feature. |
| `actual_departure_time` | UTC datetime | Recorded actual departure timestamp when available. | `2026-02-01T07:57:22Z` | Missing for 4.158% of records. Actual outcome information must not be used as a pre-event predictive feature. |
| `delay_on_departure_seconds` | Integer | Supplied departure-delay measure in seconds. | `496` | Candidate numerical target. It is strongly related to, but not interchangeable with, the performance deviation field. Missing for 4.159% of records. |
| `on_time_performance_deviation_seconds` | Integer | Difference in seconds between the relevant actual and scheduled timestamp. | `82` | Negative values are early, zero is exactly scheduled, and positive values are late. It directly determines the supplied status and therefore must not be used to predict that status. |
| `on_time_performance_status` | String | Supplied performance category derived from the performance deviation. | `On Time` | Candidate classification target. Missing for 0.086% of records, exactly where performance deviation is missing. |

## Empirically observed status thresholds

The following thresholds reproduce every non-missing status in the supplied six-month dataset. They are empirically established from the records because the source archives did not include formal documentation.

| Status | Performance deviation |
|---|---:|
| Very Early | 180 seconds early or more (`<= -180`) |
| Early | 60–179 seconds early (`-179` to `-60`) |
| On Time | 59 seconds early through 180 seconds late (`-59` to `180`) |
| Late | 181–360 seconds late |
| Very Late | 361 seconds late or more |

The status field must not be used as an input when predicting performance deviation. Conversely, performance deviation must not be used as an input when predicting status, because either choice would reveal the target directly.

## Modelling fields

The feature workflow creates separate departure-level modelling tables. Target definitions, exclusions, and chronological splits are documented in `docs/METHODS.md`.

| Column | Stored type | Description | Modelling role |
|---|---|---|---|
| `data_split` | String | Chronological assignment of `train`, `validation`, or `test`. | Evaluation context; never a predictor. |
| `service_day_of_week` | String | Weekday name derived from `service_date`. | Categorical predictor. |
| `scheduled_minute_of_service_day` | Integer | Scheduled clock time expressed as minutes from the service-day start. Values greater than 1,439 represent after-midnight service. | Numerical predictor. |
| `service_month_number` | Integer | Calendar month number derived from `service_date`. | Numerical predictor representing progression through the study period. |
| `departure_delay_seconds` | Integer | Unmodified copy of the supplied departure-delay measurement for eligible departure events. | Regression target; never a predictor. |
| `service_status` | String | Three-class target formed by consolidating the five supplied status labels into Early, On Time, and Late. | Classification target; never a predictor. |

## Fields added during cleaning

| Column | Stored type | Description | Example | Analytical role and limitations |
|---|---|---|---|---|
| `event_type` | String | Indicates which scheduled event field is populated. | `departure` | Values are `arrival` or `departure`; derived only from scheduled-field presence. |
| `scheduled_event_time` | UTC datetime | Canonical scheduled timestamp selected from the arrival or departure field according to `event_type`. | `2026-02-01T07:56:00Z` | Simplifies event-level analysis while leaving both original scheduled fields unchanged. |
| `actual_event_time` | UTC datetime | Matching actual timestamp for the selected event type. | `2026-02-01T07:57:22Z` | Missing when the corresponding actual event was not recorded. It is outcome information and can create leakage. |
| `is_after_midnight_service` | Boolean | True when the scheduled event calendar date is one day after `service_date`. | `false` | Preserves the distinction between operating service date and calendar date. |
| `has_performance_measurement` | Boolean | True when performance deviation and status are available. | `true` | Defines eligibility for performance-status analyses without deleting other records. |
| `has_departure_delay` | Boolean | True when `delay_on_departure_seconds` is available. | `true` | Defines eligibility for departure-delay analyses without deleting other records. |
