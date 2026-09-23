import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pipeline import rates, aggregate, add_windows, METRICS


class KPITests(unittest.TestCase):
    def sample(self):
        d=pd.DataFrame({'season':['2023-24']*10,'player_id':[1]*10,'team':['A']*10,
            'date':pd.date_range('2023-01-01',periods=10,tz='UTC'),'fixture':range(10),'team_match':range(10),
            'minutes':[90,0,30,60,90]*2})
        for k in METRICS: d[k]=[1,0,1,1,1]*2
        return d

    def test_zero_and_weighted_rates(self):
        d=self.sample()
        self.assertTrue(np.isnan(rates(d).iloc[1].goals_per90))
        self.assertAlmostEqual(aggregate(d,['season']).iloc[0].goals_per90,8*90/540)

    def test_gap_breaks_rolling_window(self):
        d=self.sample().drop(index=3)
        r=add_windows(d)
        self.assertTrue(np.isnan(r.loc[r.fixture.eq(5),'rolling_goals'].iloc[0]))

    def test_nonoverlapping_comparison(self):
        r=add_windows(self.sample())
        self.assertAlmostEqual(r.iloc[9].previous_goals,r.iloc[4].rolling_goals)


if __name__=='__main__': unittest.main()
