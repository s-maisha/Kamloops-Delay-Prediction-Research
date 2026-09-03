# Dataset Profile

This profile describes the ingested BC Transit dataset before cleaning. It was generated reproducibly with `src/profile_data.py`. No records were removed or altered during profiling.

## Dimensions and coverage

The table contains 3,621,203 records and 19 columns: 16 supplied columns and three ingestion-provenance columns. All 181 service dates from January 1 through June 30, 2026 are represented. Daily record counts range from 9,501 to 24,357.

The Parquet file occupies 85,257,969 bytes (81.3 MiB). Loading it with Arrow-backed Pandas types used approximately 1.035 GiB during profiling. Pandas remains practical at this scale, although memory-aware types and selective column loading are important.

## Completeness and cardinality

| Field | Missing | Missing percent | Distinct non-missing values |
|---|---:|---:|---:|
| `service_date` | 0 | 0.000% | 181 |
| `transit_system_name` | 0 | 0.000% | 1 |
| `route_name` | 0 | 0.000% | 26 |
| `direction_code` | 38,663 | 1.068% | 5 |
| `trip_id` | 0 | 0.000% | 1 |
| `vehicle_id` | 0 | 0.000% | 58 |
| `stop_id` | 0 | 0.000% | 593 |
| `stop_name` | 0 | 0.000% | 471 |
| `stop_sequence` | 0 | 0.000% | 72 |
| `scheduled_arrival_time` | 3,473,738 | 95.928% | 99,833 |
| `scheduled_departure_time` | 147,465 | 4.072% | 210,710 |
| `actual_arrival_time` | 2,161,117 | 59.680% | 1,359,014 |
| `actual_departure_time` | 150,568 | 4.158% | 2,976,874 |
| `delay_on_departure_seconds` | 150,612 | 4.159% | 2,892 |
| `on_time_performance_deviation_seconds` | 3,104 | 0.086% | 3,519 |
| `on_time_performance_status` | 3,104 | 0.086% | 5 |

The scheduled timestamp fields form a deliberate pair: 147,465 records contain only a scheduled arrival, and 3,473,738 contain only a scheduled departure. No record contains both or neither. High missingness in `scheduled_arrival_time` therefore reflects the record structure rather than evidence that 95.9% of schedules are absent.

The actual timestamp fields are less complete. There are 1,310,248 records with both actual arrival and departure, 149,838 with arrival only, 2,160,387 with departure only, and 730 with neither.

## Identifiers and mappings

All 3,621,203 `trip_id` values are the literal string `2.03E+15`. The field cannot distinguish trips and should not be used for grouping or prediction. The source representation suggests precision was lost before delivery; the omitted digits cannot be reconstructed from this dataset.

There are 26 route codes: 1, 2, 3, 4, 5, 6, 7, 9, 10, 13, 14, 16, 17, 18, and 70 through 81. The data contain 58 vehicle IDs, 593 stop IDs, and 471 stop names.

Each stop ID maps to exactly one stop name. However, 121 stop names map to more than one ID, with a maximum of three IDs for one shared name. Stop ID is therefore the more precise grouping field; the repeated names are not necessarily errors because separate physical stops can share a name.

## Delay fields and service status

For all 3,470,635 records containing scheduled and actual departure timestamps, `on_time_performance_deviation_seconds` exactly equals actual departure minus scheduled departure. It also exactly matches actual arrival minus scheduled arrival for all 147,464 records containing both arrival timestamps. These two groups account for every non-missing performance deviation.

The five status categories are completely determined by the performance deviation thresholds documented in `DATA_DICTIONARY.md`; zero mismatches were found. Missing status values occur on exactly the same 3,104 records as missing performance deviations.

`delay_on_departure_seconds` is a separate supplied measure. Among the 3,470,591 records containing both delay fields, the correlation is 0.9979, but only 1,031,276 records (29.715%) are exactly equal. Their median absolute difference is 3 seconds, the 95th percentile is 8 seconds, the 99th percentile is 20 seconds, and the maximum difference is 2,155 seconds. The fields must therefore remain distinct.

## Status counts

| Status | Records | Percent of all records |
|---|---:|---:|
| On Time | 2,368,167 | 65.397% |
| Late | 657,784 | 18.165% |
| Early | 278,416 | 7.688% |
| Very Late | 242,074 | 6.685% |
| Very Early | 71,658 | 1.979% |
| Missing | 3,104 | 0.086% |

These counts describe dataset composition rather than predictive performance.

## Delay ranges and extreme observations

| Measure | Minimum | Median | 99th percentile | Maximum |
|---|---:|---:|---:|---:|
| Departure delay | -1,327 s | 83 s | 688 s | 2,783 s |
| Performance deviation | -1,997 s | 82 s | 691 s | 4,307 s |

The largest departure delay is 2,783 seconds (46.4 minutes late), while the largest performance deviation is 4,307 seconds (71.8 minutes late). The most negative values represent 22.1 minutes early for departure delay and 33.3 minutes early for performance deviation.

Using the conventional 1.5-times-IQR statistical rule, 159,631 non-missing departure delays (4.600%) and 174,095 performance deviations (4.812%) lie outside their respective fences. These records are flagged for investigation rather than automatically treated as errors. Large operational delays can be genuine observations.

## Timestamp boundaries

Some timestamp values occur after midnight on the calendar day following `service_date`. The following-day counts are 3,617 scheduled arrivals, 69,213 scheduled departures, 13,055 actual arrivals, and 70,105 actual departures. This explains why the maximum timestamps reach July 1 even though the final service date is June 30.

The timestamps were supplied with a trailing `Z` and are stored with a UTC type. No source documentation explains whether the displayed clock values were exported as true UTC or whether local operating times were labelled with `Z`. Time-of-day interpretation should therefore retain this uncertainty until it can be checked against operational context.

## Duplicate records

No exact duplicates were found across the 16 supplied columns. Provenance fields were excluded from this duplicate test so that records copied across different files would still have been detected.

## Cleaning implications

The profile identifies issues for the cleaning stage without resolving them:

- `trip_id` cannot function as a trip identifier.
- The two delay fields must not be combined or substituted without justification.
- Status is a direct transformation of performance deviation and creates target leakage if used incorrectly.
- Missing scheduled arrival timestamps reflect mutually exclusive record structure.
- Missing actual timestamps and the 44 records with an actual departure but no departure-delay value require explicit handling.
- Extreme delays should be investigated and retained unless evidence supports exclusion.
- After-midnight records must remain associated with their service date.
