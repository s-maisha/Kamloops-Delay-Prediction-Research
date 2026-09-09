"""Search historical delay patterns with expanding chronological backtests."""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'departure_delay_seconds'
TABLES = ROOT / 'outputs/tables'
LOCAL = ROOT / 'data/processed/pattern_backtest'
FOLDS = ['2026-03-01', '2026-04-01', '2026-05-01']
# Configurations and acceptance criteria are fixed before inspecting their scores.
CONFIGS = [
    dict(name='original_grouped', minutes=60, minimum=30, days=None, day='weekday'),
    dict(name='half_hour', minutes=30, minimum=15, days=None, day='weekday'),
    dict(name='quarter_hour', minutes=15, minimum=10, days=None, day='weekday'),
    dict(name='schedule_slot', minutes=1, minimum=8, days=None, day='weekday'),
    dict(name='recent_56d', minutes=30, minimum=10, days=56, day='weekday'),
    dict(name='recent_28d', minutes=30, minimum=6, days=28, day='weekday'),
    dict(name='recent_slot_56d', minutes=1, minimum=6, days=56, day='weekday'),
    dict(name='day_type', minutes=30, minimum=15, days=None, day='day_type'),
]


def features(data, config):
    result = data.copy(deep=False)
    result = result.assign(
        slot=(data.scheduled_minute_of_service_day // config['minutes']).astype('int16'),
        day_key=data.weekday if config['day'] == 'weekday' else data.day_type,
    )
    return result


def fit(data, config):
    """Fit fallback tables exclusively from earlier, eligible observations."""
    data = features(data, config)
    levels = [
        ['route_name', 'stop_id', 'day_key', 'slot'],
        ['route_name', 'stop_id', 'slot'],
        ['route_name', 'stop_id'], ['route_name'],
    ]
    tables = []
    for keys in levels:
        groups = data.groupby(keys, observed=True)[TARGET].agg(['median', 'size'])
        tables.append((keys, groups.loc[groups['size'] >= config['minimum']]))
    return dict(config=config, tables=tables, median=float(data[TARGET].median()))


def predict(model, data):
    """Return delay estimates plus evidence counts and fallback depth."""
    data = features(data, model['config'])
    predictions = np.full(len(data), model['median'])
    support = np.zeros(len(data), dtype=np.int32)
    depth = np.full(len(data), len(model['tables']), dtype=np.int8)
    for level in reversed(range(len(model['tables']))):
        keys, table = model['tables'][level]
        index = pd.MultiIndex.from_frame(data[keys]) if len(keys) > 1 else data[keys[0]]
        values = table.reindex(index)
        available = values['median'].notna().to_numpy()
        predictions[available] = values['median'].to_numpy()[available]
        support[available] = values['size'].to_numpy()[available]
        depth[available] = level
    return predictions, support, depth


def earlier(data, cutoff, config):
    # Two service days prevent after-midnight outcomes crossing the fit boundary.
    end = cutoff - pd.Timedelta(days=2)
    mask = data.service_date < end
    if config['days'] is not None:
        mask &= data.service_date >= end - pd.Timedelta(days=config['days'])
    train = data.loc[mask]
    assert len(train) and train.service_date.max() < end
    return train


def score(actual, predictions):
    errors = np.abs(actual - predictions)
    return dict(mae=float(errors.mean()), rmse=float(np.sqrt(np.mean(errors**2))),
                within_60=float(np.mean(errors <= 60)), within_120=float(np.mean(errors <= 120)),
                p90_error=float(np.quantile(errors, .9)))


def evaluate(data, config, month, weekly=False):
    cutoff = pd.Timestamp(month)
    end = cutoff + pd.offsets.MonthBegin(1)
    validation = data.loc[(data.service_date >= cutoff) & (data.service_date < end)]
    preds = np.zeros(len(validation))
    support = np.zeros(len(validation), dtype=np.int32)
    depth = np.zeros(len(validation), dtype=np.int8)
    starts = pd.date_range(cutoff, end, freq='7D', inclusive='left') if weekly else [cutoff]
    for start in starts:
        mask = ((validation.service_date >= start) &
                (validation.service_date < min(start + pd.Timedelta(days=7), end))) if weekly else np.ones(len(validation), dtype=bool)
        model = fit(earlier(data, start, config), config)
        preds[mask], support[mask], depth[mask] = predict(model, validation.loc[mask])
    assert np.isfinite(preds).all()
    return validation, preds, support, depth


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(ROOT / 'data/processed/regression_features.parquet')
    raw['service_date'] = pd.to_datetime(raw.service_date)
    raw['weekday'] = raw.service_date.dt.dayofweek.astype('int8')
    raw['day_type'] = np.where(raw.weekday < 5, 0, raw.weekday - 4).astype('int8')
    for col in ['route_name', 'stop_id']:
        raw[col] = raw[col].astype('category')
    protocol = dict(configs=CONFIGS, folds=FOLDS, embargo_days=2,
                    primary_metric='equal-month mean MAE',
                    selection_rule='lowest equal-month MAE; weekly refresh tested once afterward',
                    provisional_gate='at least 5% MAE improvement on average and improvement in every development month',
                    june_status='previously examined retrospective audit, not independent confirmation',
                    data_source='BC Transit operational records from the supplied ZIP archives')
    (LOCAL / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows, cache = [], {}
    for config in CONFIGS:
        for month in FOLDS:
            started = time.perf_counter()
            frame, preds, _, _ = evaluate(raw, config, month)
            metrics = score(frame[TARGET].to_numpy(), preds)
            rows.append(dict(cycle=1, candidate=config['name'], month=month, rows=len(frame), **metrics))
            cache[(config['name'], month)] = preds
            print(f"{config['name']} {month}: MAE {metrics['mae']:.3f}s ({time.perf_counter()-started:.1f}s)", flush=True)
        pd.DataFrame(rows).to_csv(TABLES / 'pattern_backtest_scores.csv', index=False)
    result = pd.DataFrame(rows)
    selected = result.groupby('candidate').mae.mean().idxmin()
    config = next(c for c in CONFIGS if c['name'] == selected)
    for month in FOLDS:
        frame, preds, _, _ = evaluate(raw, config, month, weekly=True)
        name = selected + '_weekly'
        metrics = score(frame[TARGET].to_numpy(), preds)
        rows.append(dict(cycle=2, candidate=name, month=month, rows=len(frame), **metrics))
        cache[(name, month)] = preds
        print(f"{name} {month}: MAE {metrics['mae']:.3f}s", flush=True)
    result = pd.DataFrame(rows)
    winner = result.groupby('candidate').mae.mean().idxmin()
    weekly = winner.endswith('_weekly')
    selected_config = next(c for c in CONFIGS if c['name'] == winner.removesuffix('_weekly'))
    result.to_csv(TABLES / 'pattern_backtest_scores.csv', index=False)
    summary = result.groupby('candidate').agg(mean_month_mae=('mae', 'mean'),
        worst_month_mae=('mae', 'max'), within_120=('within_120', 'mean')).sort_values('mean_month_mae')
    summary.to_csv(TABLES / 'pattern_candidate_summary.csv')
    daily, route_rows = [], []
    for month in FOLDS + ['2026-06-01']:
        if month in FOLDS:
            frame = raw.loc[raw.service_date.dt.strftime('%Y-%m').eq(month[:7])]
            preds, base = cache[(winner, month)], cache[('original_grouped', month)]
        else:
            frame, preds, _, _ = evaluate(raw, selected_config, month, weekly=weekly)
            _, base, _, _ = evaluate(raw, CONFIGS[0], month)
        detail = frame[['service_date', 'route_name', TARGET]].copy()
        detail['prediction'] = preds
        detail['baseline_prediction'] = base
        detail['error'] = np.abs(frame[TARGET].to_numpy() - preds)
        detail['baseline_error'] = np.abs(frame[TARGET].to_numpy() - base)
        detail.to_parquet(LOCAL / f'predictions_{month}.parquet', index=False)
        day = detail.groupby('service_date', observed=True).agg(
            rows=('error','size'), error_sum=('error','sum'), baseline_error_sum=('baseline_error','sum'))
        daily.append(day.reset_index().assign(month=month))
        routes = detail.groupby('route_name', observed=True).agg(
            rows=('error','size'), mae=('error','mean'), baseline_mae=('baseline_error','mean'))
        route_rows.append(routes.reset_index().assign(month=month))
        print(f"SELECTED {winner} {month}: {score(frame[TARGET].to_numpy(),preds)}", flush=True)
    days = pd.concat(daily, ignore_index=True)
    days.to_csv(TABLES / 'pattern_daily_errors.csv', index=False)
    pd.concat(route_rows).to_csv(TABLES / 'pattern_route_errors.csv', index=False)
    rng = np.random.default_rng(42)
    confidence = []
    for month, day in days.groupby('month'):
        indices = rng.integers(0, len(day), size=(2000, len(day)))
        differences = day.baseline_error_sum.to_numpy() - day.error_sum.to_numpy()
        improvements = differences[indices].sum(axis=1) / day.rows.to_numpy()[indices].sum(axis=1)
        confidence.append(dict(month=month, mae_gain_seconds=float(differences.sum()/day.rows.sum()),
            day_bootstrap_lower=float(np.quantile(improvements,.025)),
            day_bootstrap_upper=float(np.quantile(improvements,.975))))
    pd.DataFrame(confidence).to_csv(TABLES / 'pattern_uncertainty.csv', index=False)
    gains = result.pivot(index='month',columns='candidate',values='mae')
    gain_percent = (1 - gains[winner] / gains.original_grouped) * 100
    final_model = fit(earlier(raw, pd.Timestamp('2026-07-03'), selected_config), selected_config)
    joblib.dump(final_model, LOCAL / 'pattern_model.joblib', compress=3)
    outcome = dict(selected=winner, config=selected_config, weekly=weekly,
        improvement_by_development_month_percent=gain_percent.to_dict(),
        provisional_gate_passed=bool(gain_percent.mean() >= 5 and (gain_percent > 0).all()),
        caveat='Development selection and day bootstrap do not remove model-selection bias or prove future accuracy.',
        next_requirement='New observed outcomes for independent validation')
    (LOCAL / 'outcome.json').write_text(json.dumps(outcome,indent=2))
    print(json.dumps(outcome,indent=2),flush=True)


if __name__ == '__main__':
    main()
