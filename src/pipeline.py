"""Bronze -> Silver -> Gold. Run from any directory with this project's Python."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['goals', 'assists', 'contributions', 'threat', 'creativity', 'influence', 'points', 'ict']


def rates(frame):
    result = frame.copy()
    for metric in METRICS:
        result[metric + '_per90'] = result[metric] * 90 / result.minutes.replace(0, np.nan)
    return result


def aggregate(df, keys):
    out = df.groupby(keys, dropna=False).agg(**{k: (k, 'sum') for k in ['minutes', *METRICS]},
        records=('fixture', 'size'), appearances=('minutes', lambda x: x.gt(0).sum())).reset_index()
    out['minutes_share'] = out.minutes / (90 * out.records)
    return rates(out)


def add_windows(df):
    parts = []
    for _, group in df.groupby(['season', 'player_id', 'team'], dropna=False):
        g = group.sort_values(['date', 'fixture']).copy()
        if g.team_match.isna().all():
            g['segment'] = np.arange(len(g))
            g['rolling_minutes'] = np.nan
            for metric in METRICS:
                g['rolling_' + metric] = np.nan
                g['previous_' + metric] = np.nan
            parts.append(g)
            continue
        # Missing team fixtures break windows; do not manufacture zero appearances.
        continuous = g.team_match.diff().eq(1)
        g['segment'] = (~continuous).cumsum()
        chunks = []
        for _, h in g.groupby('segment'):
            h = h.copy()
            sums = h[['minutes', *METRICS]].rolling(5, min_periods=5).sum()
            eligible = sums.minutes.ge(180) & h.team_match.notna()
            h['rolling_minutes'] = sums.minutes.where(eligible)
            for metric in METRICS:
                h['rolling_' + metric] = (90 * sums[metric] / sums.minutes.replace(0, np.nan)).where(eligible)
                h['previous_' + metric] = h['rolling_' + metric].shift(5)
            chunks.append(h)
        parts.append(pd.concat(chunks))
    return pd.concat(parts).sort_values(['season', 'player_id', 'date'])


def train_model(df):
    samples = []
    columns = ['rolling_minutes'] + ['rolling_' + k for k in ['goals','assists','threat','creativity','influence','ict']]
    for _, g in df.loc[df.position.eq('FW') & df.team_match.notna()].groupby(['season','player_id','team','segment']):
        g = g.sort_values('date')
        x = g[columns].shift(1)
        x['history_end'] = g.date.shift(1)
        x['date'] = g.date
        x['season'] = g.season
        x['player_id'] = g.player_id
        x['fixture'] = g.fixture
        x['scored'] = g.goals.gt(0).astype(int)
        valid = x[columns].notna().all(axis=1) & x.history_end.add(pd.Timedelta(hours=3)).lt(x.date)
        samples.append(x.loc[valid])
    samples = pd.concat(samples, ignore_index=True)
    results, predictions = [], []
    for season in ['2021-22','2022-23','2023-24']:
        test = samples.loc[samples.season.eq(season)]
        train = samples.loc[samples.date < test.date.min()]
        assert train.date.max() + pd.Timedelta(hours=3) < test.date.min()
        model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(train[columns], train.scored)
        for name, probability in [('Training scoring rate', np.full(len(test),train.scored.mean())),
                                  ('KPI logistic model', model.predict_proba(test[columns])[:,1])]:
            results.append({'test_season':season,'model':name,'train_rows':len(train),'test_rows':len(test),
                'scoring_rate':test.scored.mean(),'brier':brier_score_loss(test.scored,probability),
                'log_loss':log_loss(test.scored,probability),'roc_auc':roc_auc_score(test.scored,probability),
                'average_precision':average_precision_score(test.scored,probability)})
            if name == 'KPI logistic model':
                out = test[['season','player_id','fixture','date','scored']].copy()
                out['probability'] = probability
                predictions.append(out)
    return pd.DataFrame(results), pd.concat(predictions), samples


def run():
    for directory in ['data/silver','data/gold','outputs']:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/bronze/cleaned_merged_seasons.csv'
    raw = pd.read_csv(source, low_memory=False)
    audit = {'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'input_rows':len(raw)}
    df = raw.rename(columns={'season_x':'season','name':'player','team_x':'team','element':'player_id',
        'kickoff_time':'date','goals_scored':'goals','total_points':'points','ict_index':'ict','opp_team_name':'opponent','GW':'gameweek'}).copy()
    audit['duplicates_removed'] = int(df.duplicated().sum())
    df = df.drop_duplicates()
    if df.duplicated(['season','player_id','fixture']).any():
        raise ValueError('Conflicting fixture identities require review.')
    df['date'] = pd.to_datetime(df.date, utc=True, errors='coerce')
    df['position'] = df.position.replace({'DEF':'DF','MID':'MF','FWD':'FW','GKP':'GK'})
    numeric = ['minutes','goals','assists','threat','creativity','influence','points','ict']
    for col in numeric:
        df[col] = pd.to_numeric(df[col],errors='coerce')
    valid = df.date.notna() & np.isfinite(df[numeric]).all(axis=1) & df.minutes.between(0,90)
    valid &= df[['goals','assists','threat','creativity','influence','ict']].ge(0).all(axis=1)
    valid &= df[['player','season','player_id']].notna().all(axis=1) & df.position.isin(['GK','DF','MF','FW'])
    audit['invalid_rows_removed'] = int((~valid).sum())
    df = df.loc[valid].copy()
    schedule = df.dropna(subset=['team'])[['season','team','fixture','date']].drop_duplicates().sort_values('date')
    schedule['team_match'] = schedule.groupby(['season','team']).cumcount()
    if schedule.duplicated(['season','team','fixture']).any():
        raise ValueError('Conflicting fixture times.')
    ambiguity = df.groupby(['season','player_id']).player.transform('nunique').gt(1)
    df.loc[ambiguity].to_csv(ROOT / 'data/silver/identity_quarantine.csv', index=False)
    audit['ambiguous_identity_rows_removed'] = int(ambiguity.sum())
    df = df.loc[~ambiguity].merge(schedule,on=['season','team','fixture','date'],how='left',validate='many_to_one')
    df['team'] = df.team.fillna('Unknown team')
    df['contributions'] = df.goals + df.assists
    df['minutes_share'] = df.minutes / 90
    df = rates(df)
    audit.update({'silver_rows':len(df),'zero_minute_rows_retained':int(df.minutes.eq(0).sum()),
                  'unknown_team_rows':int(df.team.eq('Unknown team').sum())})
    df.to_csv(ROOT / 'data/silver/player_matches.csv',index=False)
    monitor = add_windows(df)
    monitor.to_parquet(ROOT / 'data/gold/monitor.parquet',index=False)
    tables = {'player':['season','team','position','player_id','player'], 'team':['season','team'],
              'gameweek':['season','gameweek'], 'season':['season'], 'position':['season','position']}
    for name, keys in tables.items():
        aggregate(df,keys).to_csv(ROOT / f'data/gold/{name}_kpis.csv',index=False)
    scores,predictions,samples = train_model(monitor)
    scores.to_csv(ROOT / 'outputs/model_scores.csv',index=False)
    predictions.to_csv(ROOT / 'outputs/goal_predictions.csv',index=False)
    samples.to_csv(ROOT / 'outputs/model_features.csv',index=False)
    audit['model_examples'] = len(samples)
    assert len(raw) == len(df) + audit['duplicates_removed'] + audit['invalid_rows_removed'] + audit['ambiguous_identity_rows_removed']
    (ROOT/'outputs/audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    run()
