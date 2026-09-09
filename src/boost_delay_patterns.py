"""Correct historical medians using only temporally cross-fitted training features."""
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from features import sha256_file
from improve_delay import ROOT, TARGET, LOCAL, TABLES, CONFIGS, fit, predict, earlier, score


def make_history(raw):
    """Build each row's historical predictors from observations before its week."""
    outputs = []
    for month in pd.date_range('2026-02-01','2026-06-01',freq='MS'):
        end = month + pd.offsets.MonthBegin(1)
        for start in pd.date_range(month,end,freq='7D',inclusive='left'):
            chunk = raw.loc[(raw.service_date >= start) &
                            (raw.service_date < min(start + pd.Timedelta(days=7),end))].copy()
            if chunk.empty:
                continue
            for name in ['original_grouped','schedule_slot','day_type']:
                config = next(c for c in CONFIGS if c['name'] == name)
                model = fit(earlier(raw,start,config),config)
                value, support, depth = predict(model,chunk)
                chunk[name] = value
                if name == 'schedule_slot':
                    chunk['support_log'] = np.log1p(support)
                    chunk['depth'] = depth
            outputs.append(chunk)
        print(f"Cross-fitted history through {end.date()}",flush=True)
    return pd.concat(outputs,ignore_index=True)


NUMERIC = ['stop_sequence','scheduled_minute_of_service_day','original_grouped',
           'schedule_slot','day_type','support_log','depth']
CATEGORY = ['route_name','direction_code','weekday']


def encode(data, mappings):
    result = data[NUMERIC].astype(float).copy()
    for col in CATEGORY:
        result[col] = data[col].astype('string').fillna('<MISSING>').map(mappings[col]).astype(float)
    return result


def train_and_predict(train, evaluate, leaves):
    mappings = {c:{key:i for i,key in enumerate(sorted(train[c].astype('string').fillna('<MISSING>').unique()))} for c in CATEGORY}
    model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=140,
        max_leaf_nodes=leaves, learning_rate=.08, min_samples_leaf=100,
        l2_regularization=10, early_stopping=False,
        categorical_features=[False]*len(NUMERIC)+[True]*len(CATEGORY), random_state=42)
    # Bound training cost while retaining the complete evaluation period.
    sample = train.sample(n=min(len(train),500000),random_state=42)
    with threadpool_limits(limits=4):
        model.fit(encode(sample,mappings),sample[TARGET]-sample.schedule_slot)
        prediction = evaluate.schedule_slot.to_numpy() + model.predict(encode(evaluate,mappings))
    return prediction, dict(model=model,mappings=mappings,numeric=NUMERIC,categories=CATEGORY)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse-history', action='store_true', help='Reuse the local cross-fitted feature cache from this dataset.')
    args = parser.parse_args()
    raw = pd.read_parquet(ROOT / 'data/processed/regression_features.parquet')
    raw['service_date'] = pd.to_datetime(raw.service_date)
    raw['weekday'] = raw.service_date.dt.dayofweek.astype('int8')
    raw['day_type'] = np.where(raw.weekday < 5,0,raw.weekday-4).astype('int8')
    for c in ['route_name','stop_id']:
        raw[c] = raw[c].astype('category')
    if args.reuse_history:
        history = pd.read_parquet(LOCAL/'cross_fitted_features.parquet')
        if history.attrs.get('source_sha256') != sha256_file(ROOT/'data/processed/regression_features.parquet'):
            raise ValueError('The history cache has no matching source checksum. Rerun without --reuse-history.')
    else:
        history = make_history(raw)
        history.attrs['source_sha256'] = sha256_file(ROOT/'data/processed/regression_features.parquet')
        history.to_parquet(LOCAL/'cross_fitted_features.parquet',index=False)
    rows = []
    for leaves in [15,31]:
        for month in ['2026-03-01','2026-04-01','2026-05-01']:
            start = pd.Timestamp(month)
            train = history.loc[history.service_date < start-pd.Timedelta(days=2)]
            evaluate = history.loc[(history.service_date>=start)&(history.service_date<start+pd.offsets.MonthBegin(1))]
            prediction,_ = train_and_predict(train,evaluate,leaves)
            metrics = score(evaluate[TARGET].to_numpy(),prediction)
            rows.append(dict(candidate=f'residual_boost_{leaves}',month=month,**metrics))
            output = evaluate[['service_date','route_name',TARGET,'schedule_slot']].copy()
            output['prediction'] = prediction
            output.to_parquet(LOCAL/f'boost_{leaves}_{month}.parquet',index=False)
            print(f"Residual boost {leaves} {month}: {metrics}",flush=True)
            pd.DataFrame(rows).to_csv(TABLES/'pattern_boost_scores.csv',index=False)
    result = pd.DataFrame(rows)
    selected = result.groupby('candidate').mae.mean().idxmin()
    leaves = int(selected.rsplit('_',1)[1])
    comparison = pd.read_csv(TABLES/'pattern_candidate_summary.csv')
    prior_mae = float(comparison.mean_month_mae.min())
    selected_mae = float(result.groupby('candidate').mae.mean()[selected])
    outcome = dict(selected_candidate=selected,development_mae=selected_mae,
                   prior_best_development_mae=prior_mae,beats_prior=selected_mae<prior_mae,
                   training_sampling='up to 500000 rows; seed 42',
                   feature_history='weekly historical tables with two-service-day gap',
                   test_status='June previously used; retrospective audit only')
    if selected_mae < prior_mae:
        june = history.loc[history.service_date >= '2026-06-01']
        train = history.loc[history.service_date < '2026-05-30']
        prediction,artifact = train_and_predict(train,june,leaves)
        outcome['june_metrics'] = score(june[TARGET].to_numpy(),prediction)
        output = june[['service_date','route_name',TARGET,'schedule_slot']].copy()
        output['prediction'] = prediction
        output.to_parquet(LOCAL/'boost_selected_june.parquet',index=False)
        joblib.dump(artifact,LOCAL/'boost_correction_model.joblib',compress=3)
    (LOCAL/'boost_outcome.json').write_text(json.dumps(outcome,indent=2))
    print(json.dumps(outcome,indent=2),flush=True)


if __name__ == '__main__':
    main()
