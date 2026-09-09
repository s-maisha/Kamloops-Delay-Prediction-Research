"""Estimate one departure delay from the retained historical-pattern model."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from improve_delay import ROOT,LOCAL,CONFIGS,earlier,fit,predict
from boost_delay_patterns import encode


def predict_one(service_date, route, stop, sequence, minute, direction):
    date=pd.Timestamp(service_date)
    if date < pd.Timestamp('2026-06-01'):
        raise ValueError('The saved correction was fitted through May; requests must be June 1 or later.')
    if not 0 <= minute < 2880 or sequence < 1:
        raise ValueError('Use service-day minutes 0..2879 and a positive stop sequence.')
    raw=pd.read_parquet(ROOT/'data/processed/regression_features.parquet')
    raw['service_date']=pd.to_datetime(raw.service_date)
    raw['weekday']=raw.service_date.dt.dayofweek.astype('int8')
    raw['day_type']=np.where(raw.weekday<5,0,raw.weekday-4).astype('int8')
    if date > raw.service_date.max()+pd.Timedelta(days=14):
        raise ValueError('Available history is more than 14 days old at this date. Refresh the operational data before requesting a current estimate.')
    query=pd.DataFrame([dict(route_name=str(route),stop_id=str(stop),stop_sequence=sequence,
        scheduled_minute_of_service_day=minute,direction_code=direction,
        weekday=date.dayofweek,day_type=0 if date.dayofweek<5 else date.dayofweek-4)])
    month=date.replace(day=1)
    week=month+pd.Timedelta(days=((date.day-1)//7)*7)
    original=CONFIGS[0]
    baseline=float(predict(fit(earlier(raw,month,original),original),query)[0][0])
    evidence={}
    for name in ['original_grouped','schedule_slot','day_type']:
        config=next(c for c in CONFIGS if c['name']==name)
        estimate,support,depth=predict(fit(earlier(raw,week,config),config),query)
        query[name]=estimate
        if name=='schedule_slot':
            query['support_log']=np.log1p(support)
            query['depth']=depth
            evidence=dict(historical_group_records=int(support[0]),fallback_level=int(depth[0]))
    artifact=joblib.load(LOCAL/'boost_correction_model.joblib')
    with threadpool_limits(limits=4):
        corrected=float(query.schedule_slot.iloc[0]+artifact['model'].predict(encode(query,artifact['mappings']))[0])
    outcome=json.loads((LOCAL/'final_outcome.json').read_text())
    weight=outcome['boost_weight']
    estimate=weight*corrected+(1-weight)*baseline
    return dict(departure_delay_seconds=round(estimate,3),departure_delay_minutes=round(estimate/60,3),
        route=str(route),stop=str(stop),service_date=str(date.date()),
        history_service_dates_before=str((week-pd.Timedelta(days=2)).date()),
        correction_training_service_dates_before='2026-05-30',
        route_met_development_reliability_screen=str(route) in outcome['predictable_routes'],
        **evidence,interpretation='Expected delay from historical observations; not a live vehicle forecast.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date',required=True)
    parser.add_argument('--route',required=True)
    parser.add_argument('--stop',required=True)
    parser.add_argument('--sequence',required=True,type=int)
    parser.add_argument('--minute',required=True,type=int,help='Minutes since service-day start; 1500 means 01:00 next day.')
    parser.add_argument('--direction',default=None)
    args=parser.parse_args()
    print(json.dumps(predict_one(args.date,args.route,args.stop,args.sequence,args.minute,args.direction),indent=2))


if __name__=='__main__':
    main()
