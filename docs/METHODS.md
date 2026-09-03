# Methods

## Prediction setting

The supervised-learning tasks represent predictions made immediately before a scheduled departure at a stop. Predictors are therefore restricted to information available from the schedule and route structure before the departure occurs. Arrival records are not included in these tasks because the principal numerical outcome is the supplied departure-delay measure.

Separate modelling tables are generated for regression and classification by `src/features.py`. The generated Parquet files remain outside version control and can be recreated from the cleaned analytical dataset.

## Targets

### Regression

The regression target is `departure_delay_seconds`, copied without modification from the supplied `delay_on_departure_seconds` field. Negative values indicate an early departure, zero indicates an exactly scheduled departure, and positive values indicate a late departure. All 3,470,591 departure records with this measurement are retained, including extreme values.

### Classification

The classification target is a three-class `service_status` field based on the supplied `on_time_performance_status`. The five observed source categories are consolidated without recalculating them from a delay value:

| Supplied status | Classification target |
|---|---|
| Very Early | Early |
| Early | Early |
| On Time | On Time |
| Late | Late |
| Very Late | Late |

This produces 3,470,635 eligible departure records: 299,316 Early, 2,295,086 On Time, and 876,233 Late. Retaining three broad classes supports an operationally interpretable comparison while preserving the source distinction between acceptable and non-acceptable timing.

## Predictors

Seven predictors are retained. Each is available before the scheduled departure and has a plausible relationship with recurring delay patterns.

| Predictor | Type | Reason for inclusion |
|---|---|---|
| `route_name` | Categorical | Represents differences in route geography, scheduling, and operating conditions. |
| `direction_code` | Categorical | Distinguishes directional service patterns on a route. Missing values are preserved for training-only preprocessing. |
| `stop_id` | Categorical | Represents location-specific delay patterns without duplicating the stop name. |
| `service_day_of_week` | Categorical | Captures recurring differences among service days. |
| `stop_sequence` | Numerical | Represents progress through the scheduled service pattern. |
| `scheduled_minute_of_service_day` | Numerical | Represents scheduled time as minutes since the service day began; values above 1,439 preserve after-midnight service. |
| `service_month_number` | Numerical | Represents broad progression across the six-month study period without treating future months as unseen categorical labels. |

The service date, scheduled timestamp, and assigned data split remain in each modelling table for traceability and evaluation. They are context columns rather than predictors.

## Leakage protection and exclusions

Actual timestamps, measured delay fields, performance status, and measurement-availability flags are outcome information. Including any of them as predictors would reveal the result that the model is intended to estimate.

Other exclusions are based on data quality or redundancy:

- `trip_id` is constant and cannot identify trips.
- `transit_system_name` is constant because the delivery contains only Kamloops records.
- `vehicle_id` is excluded because its operational meaning and availability before departure are not documented.
- `stop_name` duplicates information represented by `stop_id`.
- `event_type` is constant after selecting departure events.
- Source archive, filename, and row number are provenance fields rather than operational predictors.
- The after-midnight flag is already represented by `scheduled_minute_of_service_day` values greater than 1,439.

Later preprocessing steps, including missing-value handling and categorical encoding, must be fitted using the training set only. Historical grouped baselines must also be calculated only from observations earlier than the set being predicted.

Direction is missing in 17,632 regression records and remains missing at this stage. Four stop IDs in each later split do not occur in training, affecting 1,200 validation rows and 1,889 test rows in the regression table. Model pipelines must therefore handle missing and previously unseen categories without learning from validation or test data.

## Chronological evaluation

Whole calendar months define the fixed evaluation sets:

| Set | Service dates | Regression rows | Classification rows |
|---|---|---:|---:|
| Training | January 1–April 30, 2026 | 2,323,404 | 2,323,448 |
| Validation | May 1–31, 2026 | 573,555 | 573,555 |
| Test | June 1–30, 2026 | 573,632 | 573,632 |

The regression proportions are 66.945% training, 16.526% validation, and 16.528% test. Calendar-month boundaries provide substantial samples while keeping every training observation earlier than validation and every validation observation earlier than testing. The validation set will guide model comparison and limited tuning. The test set is reserved for final evaluation and will not be used to fit transformations, select features, or choose model settings.

## Generated datasets

Run the feature workflow after cleaning:

```bash
python src/features.py
```

The command creates `data/processed/regression_features.parquet`, `data/processed/classification_features.parquet`, and `data/processed/feature_report.json`. The report records source and output checksums, feature lists, exclusions, split boundaries, and row counts.

## Regression methods

Four approaches are compared on the May validation set. The global median predicts the January–April training median for every validation record. The historical grouped baseline calculates training-only medians using route, stop, service weekday, and scheduled service hour. Groups require at least 30 historical records and fall back through route-stop-hour, route-stop, route, and global levels when necessary.

Ridge regression provides a regularized linear comparison with `alpha=10`. Random Forest provides a nonlinear tree ensemble with 60 trees, maximum depth 18, minimum leaf size 25, square-root feature sampling, and a maximum bootstrap sample of 400,000 records per tree. A fixed random seed of 42 makes the forest reproducible. These initial settings are fixed before validation rather than selected through a broad tuning search.

Categorical predictors are one-hot encoded, categories with fewer than 20 training occurrences are grouped, and previously unseen categories are ignored safely. Numerical predictors are standardized. The encoder, category frequencies, and scaler are always fitted on the applicable training period only.

MAE is the primary selection metric because it expresses typical prediction error directly in seconds. RMSE and R² provide complementary information about large errors and explained variation. The machine-learning candidate with the lowest May MAE is refitted using January–May, then evaluated once on June. The global and grouped baselines are also recalculated from January–May for the June comparison.
