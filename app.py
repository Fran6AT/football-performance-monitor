from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Player Performance Monitor', page_icon='⚽', layout='wide')


@st.cache_data
def load():
    return pd.read_parquet(ROOT/'data/gold/monitor.parquet')


def trend(current, previous, threshold):
    if pd.isna(current) or pd.isna(previous):
        return 'Insufficient history'
    if previous == 0:
        return 'Stable at zero' if current == 0 else 'Up from zero'
    change = (current - previous) / abs(previous)
    return 'Rising' if change > threshold else 'Falling' if change < -threshold else 'Within threshold'


st.title('Premier League Player Performance Monitor')
st.caption('Historical KPI monitoring · supplied seasons through 2023–24 · no live feed')
if not (ROOT/'data/gold/monitor.parquet').exists():
    st.error('Run the data pipeline first: .venv/Scripts/python.exe src/pipeline.py')
    st.stop()
df = load()
with st.sidebar:
    st.header('Explore performance')
    season = st.selectbox('Season',sorted(df.season.unique(),reverse=True))
    scoped = df.loc[df.season.eq(season)]
    team = st.selectbox('Team',sorted(scoped.team.unique()))
    scoped = scoped.loc[scoped.team.eq(team)]
    positions = sorted(scoped.position.unique())
    position = st.selectbox('Position',positions,index=positions.index('FW') if 'FW' in positions else 0)
    scoped = scoped.loc[scoped.position.eq(position)]
    names = scoped[['player_id','player']].drop_duplicates().sort_values('player')
    labels = dict(zip(names.player_id,names.player))
    player_id = st.selectbox('Player',names.player_id.tolist(),format_func=lambda x: labels[x])
    records = scoped.loc[scoped.player_id.eq(player_id)].sort_values('date')
    fixture = st.selectbox('Match / gameweek',records.fixture.tolist(),index=len(records)-1,
        format_func=lambda x: records.loc[records.fixture.eq(x)].iloc[0].date.strftime('%d %b %Y') +
        ' · GW ' + str(records.loc[records.fixture.eq(x)].iloc[0].gameweek) + ' · ' + str(records.loc[records.fixture.eq(x)].iloc[0].opponent))
    threshold = st.slider('Trend change threshold (%)',5,50,20,5)/100
current = records.loc[records.fixture.eq(fixture)].iloc[0]
history = records.loc[records.date.le(current.date)]
tabs = st.tabs(['Player monitor','Peer comparison','Future scoring experiment','Data & definitions'])
with tabs[0]:
    st.subheader(f'{labels[player_id]} · {team}')
    st.caption(f"As of {current.date.strftime('%d %B %Y')} · gameweek {current.gameweek}. Season-to-date values cover this team spell only.")
    a,b,c,d = st.columns(4)
    a.metric('Recorded minutes',f'{history.minutes.sum():,.0f}')
    b.metric('Goals + assists',f'{history.contributions.sum():.0f}')
    c.metric('Minutes share',f'{history.minutes.sum()/(90*len(history)):.0%}')
    d.metric('Supplied match records',len(history))
    st.caption('Minutes share is utilisation across supplied records, not injury availability or share of all possible season minutes.')
    kpis = {'goals':'Goals /90','assists':'Assists /90','contributions':'Contributions /90',
            'threat':'Threat /90','creativity':'Creativity /90','influence':'Influence /90','points':'FPL points /90','ict':'ICT /90'}
    rows = []
    for key,label in kpis.items():
        rows.append({'KPI':label,'Selected match':current[key+'_per90'],'Last five team matches':current['rolling_'+key],
            'Team spell to date':90*history[key].sum()/history.minutes.sum() if history.minutes.sum()>0 else np.nan,
            'Previous five':current['previous_'+key],
            'Trend':trend(current['rolling_'+key],current['previous_'+key],threshold)})
    st.dataframe(pd.DataFrame(rows).set_index('KPI').round(2),width='stretch')
    st.caption('Trend compares two non-overlapping five-match windows; each needs ≥180 total minutes and complete records. The threshold is an editable descriptive rule, not a statistical significance test. Zero denominators stay blank.')
    chosen = st.selectbox('Trend KPI',list(kpis),format_func=lambda x:kpis[x],index=3)
    st.line_chart(history.set_index('date')[[chosen+'_per90','rolling_'+chosen]].rename(columns={chosen+'_per90':'Single match','rolling_'+chosen:'Five-match weighted rate'}))
    if team == 'Unknown team':
        st.info('Team names are missing for this season. Match-level KPIs are available, but consecutive-team-match trends are disabled.')
    with st.expander('Underlying match records'):
        st.dataframe(history[['date','gameweek','opponent','minutes','goals','assists','threat','creativity','influence','points']],hide_index=True)
    st.download_button('Download selected player history',history.to_csv(index=False),file_name='player_history.csv',mime='text/csv')
with tabs[1]:
    st.subheader('Compare within the same season and position')
    minimum = st.slider('Minimum recorded minutes',0,1800,450,90)
    peers = df.loc[df.season.eq(season) & df.position.eq(position) & df.date.le(current.date)]
    from src.pipeline import aggregate
    table = aggregate(peers,['player_id','player'])
    table = table.loc[table.minutes.ge(minimum) & table.minutes.gt(0)]
    st.caption('As-of comparison uses only matches completed by the selected date; totals cover all supplied team spells for each player.')
    st.dataframe(table[['player','minutes','goals_per90','assists_per90','threat_per90','creativity_per90','influence_per90']].sort_values('threat_per90',ascending=False).round(2),hide_index=True,width='stretch')
    if not table.empty:
        st.scatter_chart(table,x='creativity_per90',y='threat_per90')
with tabs[2]:
    st.subheader('Can previous KPIs predict scoring in the next match?')
    st.write('A logistic model uses only five earlier complete team matches for FPL forwards, with at least 180 minutes in that history. It is compared with the training scoring rate. Each test season follows its training seasons; current-match KPIs never enter its predictors.')
    scores = pd.read_csv(ROOT/'outputs/model_scores.csv')
    st.dataframe(scores.round(4),hide_index=True,width='stretch')
    st.caption('Lower Brier score and log loss are better. ROC AUC measures ranking. Average precision should be compared with scoring prevalence. Ordinary accuracy can be misleading because most matches have no goal. These models do not validate the trend flags as early-warning signals.')
    preds = pd.read_csv(ROOT/'outputs/goal_predictions.csv')
    selected = preds.loc[preds.season.eq(season) & preds.player_id.eq(player_id) & preds.fixture.eq(fixture)]
    if selected.empty:
        st.info('No held-out prediction for this record: only forward records in 2021–24 with eligible prior history are scored.')
    else:
        st.metric('Historical pre-match scoring probability',f"{selected.iloc[0].probability:.1%}")
        st.caption('This prediction was generated before the selected target match using prior-match features; the monitoring tab describes performance after that match.')
with tabs[3]:
    st.subheader('Bronze → Silver → Gold')
    st.write('Bronze preserves the original CSV. Silver validates types, normalises position labels, quarantines ambiguous identities and retains zero-minute games. Gold contains player, team, gameweek, season, position and monitoring tables.')
    st.json(json.loads((ROOT/'outputs/audit.json').read_text()))
    st.write('Per-90 = sum of a statistic ÷ sum of minutes × 90. Never average individual per-90 rates. FPL threat, creativity, influence and ICT are source-provided indices, not independently measured physical quantities. Assists follow the supplied FPL definition.')
    st.write('No composite PPE score is imposed: weights and overlapping KPI inputs would require validation. No pitch coordinates, injury availability, lineup feed or causal inference is provided. Missing team names remain unknown; player IDs are not assumed stable across seasons. Repeated players and overlapping windows make observations dependent.')
