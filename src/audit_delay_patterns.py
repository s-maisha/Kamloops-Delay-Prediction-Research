"""Audit error consistency and identify predictable route groups out of sample."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from improve_delay import ROOT, LOCAL, TABLES, TARGET, CONFIGS, earlier, fit, predict, score


def main():
    raw = pd.read_parquet(ROOT/'data/processed/regression_features.parquet')
    raw['service_date'] = pd.to_datetime(raw.service_date)
    raw['weekday'] = raw.service_date.dt.dayofweek.astype('int8')
    raw['day_type'] = np.where(raw.weekday < 5,0,raw.weekday-4).astype('int8')
    history = pd.read_parquet(LOCAL/'cross_fitted_features.parquet')
    selected = json.loads((LOCAL/'boost_outcome.json').read_text())['selected_candidate']
    leaves = int(selected.rsplit('_',1)[1])
    all_details, summaries, route_rows, daily_rows = [], [], [], []
    for month in ['2026-03-01','2026-04-01','2026-05-01','2026-06-01']:
        start = pd.Timestamp(month)
        frame = history.loc[(history.service_date>=start)&(history.service_date<start+pd.offsets.MonthBegin(1))].reset_index(drop=True)
        file = LOCAL/('boost_selected_june.parquet' if month=='2026-06-01' else f'boost_{leaves}_{month}.parquet')
        predictions = pd.read_parquet(file).reset_index(drop=True)
        assert frame.service_date.equals(predictions.service_date)
        np.testing.assert_array_equal(frame[TARGET],predictions[TARGET])
        np.testing.assert_array_equal(frame.route_name.astype(str),predictions.route_name.astype(str))
        baseline,_,_ = predict(fit(earlier(raw,start,CONFIGS[0]),CONFIGS[0]),frame)
        actual = frame[TARGET].to_numpy()
        detail = frame[['service_date','route_name',TARGET,'scheduled_minute_of_service_day','weekday']].copy()
        detail['month'] = month
        detail['prediction'] = predictions.prediction.to_numpy()
        detail['baseline_prediction'] = baseline
        detail['error'] = np.abs(actual-detail.prediction)
        detail['baseline_error'] = np.abs(actual-baseline)
        detail['within120'] = detail.error <= 120
        detail['hour'] = detail.scheduled_minute_of_service_day // 60
        all_details.append(detail)
        for method, estimate in [('original_grouped',baseline),('selected_boost',detail.prediction.to_numpy())]:
            summaries.append(dict(month=month,method=method,rows=len(frame),**score(actual,estimate)))
        summaries.append(dict(month=month,method='weekly_original_grouped',rows=len(frame),**score(actual,frame.original_grouped.to_numpy())))
        route = detail.groupby('route_name',observed=True).agg(rows=('error','size'),mae=('error','mean'),baseline_mae=('baseline_error','mean'),within120=('within120','mean'))
        route_rows.append(route.reset_index().assign(month=month))
        days = detail.groupby('service_date').agg(rows=('error','size'),error_sum=('error','sum'),baseline_sum=('baseline_error','sum'))
        daily_rows.append(days.reset_index().assign(month=month))
    data = pd.concat(all_details,ignore_index=True)
    summaries = pd.DataFrame(summaries)
    summaries.to_csv(TABLES/'pattern_selected_scores.csv',index=False)
    routes = pd.concat(route_rows,ignore_index=True)
    routes.to_csv(TABLES/'pattern_selected_routes.csv',index=False)
    daily = pd.concat(daily_rows,ignore_index=True)
    daily.to_csv(TABLES/'pattern_selected_days.csv',index=False)
    confidence = []
    rng = np.random.default_rng(42)
    for month,day in daily.groupby('month'):
        indices = rng.integers(0,len(day),size=(2000,len(day)))
        differences = day.baseline_sum.to_numpy()-day.error_sum.to_numpy()
        gain = differences[indices].sum(axis=1)/day.rows.to_numpy()[indices].sum(axis=1)
        confidence.append(dict(month=month,mae_gain=float(differences.sum()/day.rows.sum()),
            lower95=float(np.quantile(gain,.025)),upper95=float(np.quantile(gain,.975)),days_improved=int((differences>0).sum()),days=len(day)))
    pd.DataFrame(confidence).to_csv(TABLES/'pattern_selected_uncertainty.csv',index=False)
    # Eligibility uses development only; June is then audited for the frozen group.
    development_routes = routes.loc[routes.month<'2026-06-01']
    eligible = development_routes.groupby('route_name',observed=True).filter(
        lambda g: len(g)==3 and (g.rows>=1000).all() and (g.mae<=90).all() and (g.within120>=.75).all())
    selected_routes = sorted(eligible.route_name.astype(str).unique())
    subset = data.loc[data.month.eq('2026-06-01') & data.route_name.astype(str).isin(selected_routes)]
    subset_result = dict(routes=selected_routes,selection='At least 1000 records, MAE <=90 seconds, and >=75% within 120 seconds in EACH development month',
        june_rows=len(subset),june_coverage=float(len(subset)/573632),
        june_metrics=score(subset[TARGET].to_numpy(),subset.prediction.to_numpy()) if len(subset) else None)
    (LOCAL/'reliable_routes.json').write_text(json.dumps(subset_result,indent=2))
    dev = data.loc[data.month<'2026-06-01']
    for col in ['hour','weekday']:
        table = dev.groupby(col).agg(rows=('error','size'),mae=('error','mean'),baseline_mae=('baseline_error','mean'),within120=('within120','mean'))
        table.to_csv(TABLES/f'pattern_selected_{col}.csv')
    # Fourth cycle: a small, declared blend search assessed on development months.
    blend_rows = []
    for weight in [0,.25,.5,.75,1]:
        for month,frame in dev.groupby('month'):
            estimate=weight*frame.prediction.to_numpy()+(1-weight)*frame.baseline_prediction.to_numpy()
            blend_rows.append(dict(boost_weight=weight,month=month,**score(frame[TARGET].to_numpy(),estimate)))
    blends=pd.DataFrame(blend_rows)
    blends.to_csv(TABLES/'pattern_blend_scores.csv',index=False)
    weight=float(blends.groupby('boost_weight').mae.mean().idxmin())
    data['prediction']=weight*data.prediction+(1-weight)*data.baseline_prediction
    data['error']=np.abs(data[TARGET]-data.prediction)
    data['within120']=data.error<=120
    final_scores,final_routes,final_days=[],[],[]
    for month,frame in data.groupby('month'):
        final_scores.append(dict(month=month,boost_weight=weight,rows=len(frame),
            baseline_mae=float(frame.baseline_error.mean()),**score(frame[TARGET].to_numpy(),frame.prediction.to_numpy())))
        route=frame.groupby('route_name',observed=True).agg(rows=('error','size'),mae=('error','mean'),baseline_mae=('baseline_error','mean'),within120=('within120','mean'))
        final_routes.append(route.reset_index().assign(month=month))
        daily=frame.groupby('service_date').agg(rows=('error','size'),error_sum=('error','sum'),baseline_sum=('baseline_error','sum'))
        final_days.append(daily.reset_index().assign(month=month))
    final_scores=pd.DataFrame(final_scores)
    weekly=summaries.loc[summaries.method.eq('weekly_original_grouped')].set_index('month').mae
    final_scores['weekly_baseline_mae']=final_scores.month.map(weekly)
    final_scores['gain_percent']=(1-final_scores.mae/final_scores.baseline_mae)*100
    final_scores.to_csv(TABLES/'pattern_final_scores.csv',index=False)
    final_routes=pd.concat(final_routes,ignore_index=True)
    final_routes.to_csv(TABLES/'pattern_final_routes.csv',index=False)
    final_days=pd.concat(final_days,ignore_index=True)
    final_days.to_csv(TABLES/'pattern_final_days.csv',index=False)
    final_confidence=[]
    for month,day in final_days.groupby('month'):
        indices=rng.integers(0,len(day),size=(2000,len(day)))
        differences=day.baseline_sum.to_numpy()-day.error_sum.to_numpy()
        gain=differences[indices].sum(axis=1)/day.rows.to_numpy()[indices].sum(axis=1)
        final_confidence.append(dict(month=month,mae_gain=float(differences.sum()/day.rows.sum()),lower95=float(np.quantile(gain,.025)),upper95=float(np.quantile(gain,.975)),days_improved=int((differences>0).sum()),days=len(day)))
    pd.DataFrame(final_confidence).to_csv(TABLES/'pattern_final_uncertainty.csv',index=False)
    eligible=final_routes.loc[final_routes.month<'2026-06-01'].groupby('route_name',observed=True).filter(
        lambda g: len(g)==3 and (g.rows>=1000).all() and (g.mae<=90).all() and (g.within120>=.75).all())
    route_list=sorted(eligible.route_name.astype(str).unique())
    subset=data.loc[data.month.eq('2026-06-01') & data.route_name.astype(str).isin(route_list)]
    dev_scores=final_scores.loc[final_scores.month<'2026-06-01']
    outcome=dict(boost_weight=weight,correction=selected,
        development_mean_mae=float(dev_scores.mae.mean()),
        development_baseline_mean_mae=float(dev_scores.baseline_mae.mean()),
        provisional_gate_passed=bool((1-dev_scores.mae.mean()/dev_scores.baseline_mae.mean())>=.05 and (dev_scores.gain_percent>0).all()),
        predictable_routes=route_list,june_subset_rows=len(subset),
        june_subset_metrics=score(subset[TARGET].to_numpy(),subset.prediction.to_numpy()) if len(subset) else None,
        limitations='Repeated development selection; June is a previously examined retrospective audit. No production accuracy guarantee.')
    (LOCAL/'final_outcome.json').write_text(json.dumps(outcome,indent=2))
    data.to_parquet(LOCAL/'final_backtest_predictions.parquet',index=False)
    for col in ['hour','weekday']:
        table=data.loc[data.month<'2026-06-01'].groupby(col).agg(rows=('error','size'),mae=('error','mean'),baseline_mae=('baseline_error','mean'),within120=('within120','mean'))
        table.to_csv(TABLES/f'pattern_final_{col}.csv')
    print('FINAL:',json.dumps(outcome,indent=2),flush=True)
    print('Blend mean development MAE:',blends.groupby('boost_weight').mae.mean().to_dict(),flush=True)
    print('Predictable route subset:',json.dumps(subset_result,indent=2),flush=True)
    print('Day bootstrap:',confidence,flush=True)
    fig,ax=plt.subplots(figsize=(9,4.8))
    for column,label in [('baseline_mae','Original grouped median'),('mae','Selected blend of historical estimates')]:
        ax.plot(['March','April','May','June*'],final_scores[column],marker='o',label=label)
    ax.set(ylabel='Mean absolute error (seconds)',title='Chronological delay backtests')
    ax.grid(alpha=.2)
    ax.legend()
    fig.text(.1,.01,'*June was previously examined; this is a retrospective audit.',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,1])
    path=ROOT/'outputs/figures/pattern_backtest_comparison.png'
    fig.savefig(path,dpi=160)
    plt.close(fig)


if __name__=='__main__':
    main()
