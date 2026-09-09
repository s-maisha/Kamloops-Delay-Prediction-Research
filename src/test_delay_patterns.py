"""Checks for chronological isolation and prediction fallback behaviour."""
import unittest
import numpy as np
import pandas as pd
from improve_delay import fit, predict, earlier
from boost_delay_patterns import encode, NUMERIC, CATEGORY


class PatternTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(name='test', minutes=30, minimum=2, days=None, day='weekday')
        self.data = pd.DataFrame(dict(
            service_date=pd.to_datetime(['2026-01-01'] * 3 + ['2026-01-30','2026-02-01']),
            route_name=['1'] * 5, stop_id=['A'] * 5, weekday=[0]*5, day_type=[0]*5,
            scheduled_minute_of_service_day=[1500]*5,
            departure_delay_seconds=[10,20,30,10000,20000]))

    def test_future_labels_do_not_change_fit(self):
        cutoff = pd.Timestamp('2026-02-01')
        original = fit(earlier(self.data, cutoff, self.config), self.config)
        changed = self.data.copy()
        changed.loc[changed.service_date >= '2026-01-30','departure_delay_seconds'] = -99999
        alternate = fit(earlier(changed, cutoff, self.config), self.config)
        np.testing.assert_array_equal(predict(original,self.data)[0], predict(alternate,self.data)[0])
        self.assertEqual(original['median'],20)

    def test_unknown_categories_use_fallback(self):
        model = fit(self.data.iloc[:3],self.config)
        query = self.data.iloc[:1].copy()
        query['route_name'], query['stop_id'] = 'unknown', 'unknown'
        prediction, count, depth = predict(model, query)
        self.assertEqual(prediction[0],20)
        self.assertEqual(count[0],0)
        self.assertEqual(depth[0],4)

    def test_after_midnight_is_not_wrapped(self):
        model = fit(self.data.iloc[:3], self.config)
        query = self.data.iloc[:1].copy()
        self.assertEqual(predict(model,query)[2][0],0)
        query['scheduled_minute_of_service_day'] = 60
        self.assertEqual(predict(model,query)[2][0],2)

    def test_correction_features_exclude_outcomes_and_handle_unknowns(self):
        data=pd.DataFrame({c:[1.] for c in NUMERIC})
        for col in CATEGORY:
            data[col]='new_category'
        data['departure_delay_seconds']=123
        mappings={c:{'known':0} for c in CATEGORY}
        original=encode(data,mappings)
        data['departure_delay_seconds']=-99999
        changed=encode(data,mappings)
        pd.testing.assert_frame_equal(original,changed)
        self.assertTrue(original[CATEGORY].isna().all().all())


if __name__ == '__main__':
    unittest.main()
