from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
POSITIONS = {'GK': 'Goalkeeper', 'DF': 'Defender', 'MF': 'Midfielder', 'FW': 'Forward'}
METRICS = {
    'contributions': ('Goals + assists', 'Direct attacking contributions: goals scored plus assists credited by FPL.'),
    'goals': ('Goals', 'Goals scored by the player.'),
    'assists': ('Assists', 'Assists credited under the dataset’s Fantasy Premier League rules.'),
    'threat': ('Attacking threat', 'An FPL index of goal threat. It is not a goal count or a probability.'),
    'creativity': ('Chance creation', 'FPL’s creativity index describes creating attacking opportunities. It is not a count of chances.'),
    'influence': ('Overall influence', 'FPL’s influence index summarises a player’s impact on a match.'),
    'points': ('Fantasy points', 'Points awarded in Fantasy Premier League (FPL). These are different from goals.'),
    'ict': ('Combined FPL index', 'The source’s combined Influence, Creativity and Threat index.')}

st.set_page_config(page_title='Football Performance Explorer', page_icon='⚽', layout='wide')


@st.cache_data
def load():
    return pd.read_parquet(ROOT/'data/gold/monitor.parquet')


def trend(current, previous, threshold):
    if pd.isna(current) or pd.isna(previous):
        return 'Not enough data'
    if previous == 0:
        return 'Unchanged at zero' if current == 0 else 'Higher than zero'
    change = (current - previous) / abs(previous)
    return 'Increasing' if change > threshold else 'Decreasing' if change < -threshold else 'Little change'


def show_number(value):
    return 'Not available' if pd.isna(value) else f'{value:.2f}'


st.title('Football Performance Explorer')
st.write('See how a player is performing, follow changes over time, and compare them with similar players.')
st.caption('Historical Premier League data through May 2024. This dashboard does not show live matches.')
if not (ROOT/'data/gold/monitor.parquet').exists():
    st.warning('The player data has not been prepared yet.')
    with st.expander('Set up this dashboard'):
        st.write('Place the original CSV in data/bronze, then run this command in the project terminal:')
        st.code('.venv/Scripts/python.exe src/pipeline.py', language='powershell')
    st.stop()

df = load()
with st.sidebar:
    st.header('Quick guide')
    st.write('1. Choose a season, team and player.\n2. Pick a match to look back from.\n3. Explore the player overview, comparisons or scoring estimate.')
    st.info('New here? Start with Player overview. All figures there describe matches up to your selected date.')
    with st.expander('Advanced: trend sensitivity'):
        threshold = st.slider('Change needed to label a trend (%)',5,50,20,5,
            help='Compare the last five team matches with the five before them. A smaller value labels more changes as increasing or decreasing.')/100
        st.caption('This is a descriptive rule, not a statistical test.')

st.subheader('1. Choose a player')
a,b,c = st.columns([1,1,2])
with a:
    season = st.selectbox('Season', sorted(df.season.unique(),reverse=True), help='Choose a historical season.')
season_rows = df.loc[df.season.eq(season)]
with b:
    team = st.selectbox('Team', sorted(season_rows.team.unique()), help='Teams are listed as recorded in the source. Some older seasons have no team names.')
team_rows = season_rows.loc[season_rows.team.eq(team)]
# Prefer a player with enough recent history for an informative first view.
players = team_rows.groupby(['player_id','player']).minutes.sum().reset_index().sort_values('player')
labels = dict(zip(players.player_id,players.player))
forward_minutes = team_rows.loc[team_rows.position.eq('FW')].groupby('player_id').minutes.sum()
default_id = forward_minutes.idxmax() if not forward_minutes.empty else players.loc[players.minutes.idxmax(),'player_id']
latest_records = team_rows.sort_values('date').groupby('player_id').tail(1)
eligible_ids = latest_records.loc[latest_records.rolling_contributions.notna() & latest_records.previous_contributions.notna(), 'player_id']
eligible_players = players.loc[players.player_id.isin(eligible_ids)]
attacking_ids = latest_records.loc[latest_records.position.isin(['FW','MF']), 'player_id']
eligible_attackers = eligible_players.loc[eligible_players.player_id.isin(attacking_ids)]
if not eligible_attackers.empty:
    eligible_players = eligible_attackers
if not eligible_players.empty:
    default_id = eligible_players.loc[eligible_players.minutes.idxmax(), 'player_id']
ids = players.player_id.tolist()
with c:
    player_id = st.selectbox('Player — type a name to search', ids, index=ids.index(default_id), format_func=lambda x: labels[x])
records = team_rows.loc[team_rows.player_id.eq(player_id)].sort_values('date')
record_lookup = records.set_index('fixture')
st.subheader('2. Choose the match to look back from')
fixture = st.selectbox('View performance up to this match',records.fixture.tolist(),index=len(records)-1,
    format_func=lambda x: record_lookup.loc[x].date.strftime('%d %b %Y') + ' · vs ' + str(record_lookup.loc[x].opponent) + ' · Gameweek ' + str(record_lookup.loc[x].gameweek),
    help='The latest available match is selected first. Earlier matches let you explore what was known at that point.')
current = records.loc[records.fixture.eq(fixture)].iloc[0]
position = current.position
history = records.loc[records.date.le(current.date)]
st.caption(f"{labels[player_id]} · {POSITIONS.get(position,position)} · {team} · {season} · Up to {current.date.strftime('%d %B %Y')}")
if team == 'Unknown team':
    st.info('Team names are missing in this season. Player totals still work, but five-team-match trends are unavailable. Choose 2020–21 or later to explore those trends.')

page = st.radio('3. What would you like to explore?', ['Player overview','Compare players','Chance of scoring','Help & data'],horizontal=True)
if page == 'Player overview':
    st.subheader(f'How is {labels[player_id]} doing?')
    st.caption('Totals below cover the supplied matches for this team up to your selected date.')
    a,b,c,d = st.columns(4)
    a.metric('Minutes played',f'{history.minutes.sum():,.0f}')
    b.metric('Goals scored',f'{history.goals.sum():.0f}')
    c.metric('Assists',f'{history.assists.sum():.0f}')
    d.metric('Matches played',f'{history.minutes.gt(0).sum()} of {len(history)}',help='Matches with at least one minute played, out of the supplied team-match records.')
    if history.minutes.sum() == 0:
        st.info('This player has not played any recorded minutes in the selected period. Playing-time-adjusted statistics will appear once they play.')
    with st.expander('What does “per 90 minutes” mean?'):
        st.write('It expresses a statistic for the equivalent of one full match, helping compare players with different playing time. For example, 2 goals in 180 minutes equals 1 goal per 90 minutes.')
        st.write('Small amounts of playing time can produce extreme rates. We calculate recent form from total statistics divided by total minutes, rather than averaging individual match rates.')
    st.subheader('Recent form')
    metric = st.selectbox('What would you like to track?',list(METRICS),format_func=lambda x:METRICS[x][0])
    title,description = METRICS[metric]
    st.write(description)
    now, before = current['rolling_'+metric], current['previous_'+metric]
    state = trend(now,before,threshold)
    if pd.isna(now):
        st.info('Not enough recent playing time for a five-match rate. We need five consecutive team-match records and at least 180 minutes played across them. Missing records also prevent a calculation.')
    else:
        st.write(f'**{title}: {state.lower()}.** The latest five-match rate is **{now:.2f} per 90 minutes**.' if not pd.isna(before)
                 else f'**{title}: {now:.2f} per 90 minutes** over the latest five team matches. There is not enough earlier history to describe a change yet.')
    a,b,c = st.columns(3)
    a.metric('Last 5 team matches /90',show_number(now))
    b.metric('Previous 5 team matches /90',show_number(before))
    to_date = 90*history[metric].sum()/history.minutes.sum() if history.minutes.sum()>0 else np.nan
    c.metric('Season so far for this team /90',show_number(to_date))
    st.caption(f'A change greater than {threshold:.0%} is labelled increasing or decreasing. Each five-match period needs at least 180 minutes. These labels describe the data; they do not predict future performance.')
    chart = history.set_index('date')[[metric+'_per90','rolling_'+metric]].rename(columns={metric+'_per90':'Each match', 'rolling_'+metric:'Last 5 team matches'})
    if chart.notna().any().any():
        st.line_chart(chart,y_label=title+' per 90 minutes',x_label='Match date')
        st.caption('The five-match line smooths out match-to-match changes. Gaps mean there was not enough valid data.')
    with st.expander('See all performance measures'):
        rows=[]
        for key,(label,_) in METRICS.items():
            rows.append({'Measure (per 90 minutes)':label,'Last 5 matches':current['rolling_'+key],
                'Previous 5 matches':current['previous_'+key],'Season so far':90*history[key].sum()/history.minutes.sum() if history.minutes.sum()>0 else np.nan,
                'Direction':trend(current['rolling_'+key],current['previous_'+key],threshold)})
        st.dataframe(pd.DataFrame(rows).round(2),hide_index=True,width='stretch')
        st.caption('A blank is unavailable, not zero. Attacking measures are generally more useful for forwards and midfielders than goalkeepers.')
    with st.expander('See the matches behind these numbers'):
        detail=history[['date','opponent','minutes','goals','assists','points']].rename(columns={'date':'Date','opponent':'Opponent','minutes':'Minutes','goals':'Goals','assists':'Assists','points':'FPL points'})
        st.dataframe(detail,hide_index=True,width='stretch')
        st.download_button('Download this player’s match history',history.to_csv(index=False),file_name='player_history.csv',mime='text/csv')

elif page == 'Compare players':
    st.subheader(f'Compare {POSITIONS.get(position,position).lower()}s in {season}')
    st.write('Compare players in the same position across all teams. Choose a measure to sort the table.')
    minimum = st.slider('Only include players with at least this many minutes',0,1800,450,90,
        help='450 minutes is the equivalent of five full matches. Increasing this reduces comparisons based on very short appearances.')
    ranking = st.selectbox('Compare by',list(METRICS),format_func=lambda x:METRICS[x][0],index=0)
    st.caption(METRICS[ranking][1])
    peers = df.loc[df.season.eq(season)&df.position.eq(position)&df.date.le(current.date)]
    from src.pipeline import aggregate
    table=aggregate(peers,['player_id','player'])
    table=table.loc[table.minutes.ge(minimum)&table.minutes.gt(0)].sort_values(ranking+'_per90',ascending=False)
    st.caption('Only matches up to the selected date are included. A player’s totals here include all their teams in this season, so they may differ from the team-specific overview.')
    if table.empty:
        st.info('No players meet this playing-time requirement yet. Lower the minimum minutes or choose a later match.')
    else:
        comparison=table[['player','minutes',ranking+'_per90']].rename(columns={'player':'Player','minutes':'Minutes played',ranking+'_per90':METRICS[ranking][0]+' per 90'})
        selected_rows = set(table.index[table.player_id.eq(player_id)])
        highlighted = comparison.round(2).style.apply(
            lambda row: ['background-color: #D6F3E5; color: #123C2D; font-weight: bold'
                         if row.name in selected_rows else '' for _ in row], axis=1)
        st.dataframe(highlighted,hide_index=True,width='stretch')
        if selected_rows:
            st.caption(f'The green highlighted row is your selected player: {labels[player_id]}.')
        if player_id not in table.player_id.values:
            st.caption('Your selected player is not in this table because they have not reached the minimum minutes in this position.')
        with st.expander('Explore attacking styles: creating vs threatening'):
            st.write('Further right means a higher FPL creativity rate. Higher up means a higher FPL threat rate. These are indices, not counts of chances or expected goals.')
            st.scatter_chart(table.rename(columns={'creativity_per90':'Chance creation per 90','threat_per90':'Attacking threat per 90'}),x='Chance creation per 90',y='Attacking threat per 90')

elif page == 'Chance of scoring':
    st.subheader('What was the estimated chance of scoring?')
    st.write('This is a historical experiment: the model estimates whether a forward scores at least one goal in the selected match, using only earlier matches.')
    st.caption('Unlike Player overview, this estimate looks forward from before the selected match. It is not a prediction for a current upcoming fixture.')
    if not (ROOT/'outputs/goal_predictions.csv').exists():
        st.info('Scoring estimates have not been prepared yet. You can still explore the player overview and comparisons.')
    else:
        preds=pd.read_csv(ROOT/'outputs/goal_predictions.csv')
        selected=preds.loc[preds.season.eq(season)&preds.player_id.eq(player_id)&preds.fixture.eq(fixture)]
        if selected.empty:
            if position!='FW':
                reason='This experiment covers forwards only. Choose a forward to see an estimate.'
            elif season not in ['2021-22','2022-23','2023-24']:
                reason='Estimates are available for 2021–22 through 2023–24. Earlier data was used for training or lacks team names.'
            else:
                reason='This match does not have enough eligible earlier history. Try a later match: the model needs five complete prior team-match records and at least 180 minutes in that history.'
            st.info(reason)
        else:
            probability=selected.iloc[0].probability
            st.metric('Estimated chance of at least one goal',f'{probability:.0%}')
            st.write(f'About {round(probability*100)} in 100, according to the model. This is an estimate, not a guarantee or the model’s accuracy score.')
            if st.checkbox('Show what actually happened'):
                st.write('The player scored at least one goal.' if bool(selected.iloc[0].scored) else 'The player did not score.')
        with st.expander('How reliable is this experiment?'):
            st.write('The model learned from earlier seasons and was tested on later matches. It uses recent minutes and attacking statistics. It does not know injuries, lineups or tactical plans. The results do not prove that a rising trend causes future goals.')
            if (ROOT/'outputs/model_scores.csv').exists():
                scores=pd.read_csv(ROOT/'outputs/model_scores.csv')
                display=scores.pivot(index='test_season',columns='model',values='brier').rename(columns={'KPI logistic model':'Player-statistics model','Training scoring rate':'Simple scoring-rate baseline'})
                st.write('Probability error on later seasons — lower is better:')
                st.dataframe(display.round(3),width='stretch')
                st.caption('This is the Brier score: average squared error in predicted probabilities, not a percentage accuracy.')
                with st.expander('Full model evaluation for advanced users'):
                    st.dataframe(scores.round(4),hide_index=True)

else:
    st.subheader('A quick guide to the dashboard')
    st.markdown('**Player overview:** follow one player’s goals, assists and other performance measures.\n\n**Compare players:** compare players in the same position after allowing for their playing time.\n\n**Chance of scoring:** explore an experimental estimate made before a historical match.')
    with st.expander('Football and statistics terms',expanded=True):
        st.markdown('**Per 90 minutes:** a rate for the equivalent of one full match.\n\n**Assist:** a contribution credited toward a teammate’s goal under FPL rules.\n\n**Gameweek:** a fantasy-football round. A team can have more than one fixture in a gameweek, so this dashboard follows individual matches.\n\n**FPL:** Fantasy Premier League. Its points and performance indices are not the same as real goals.\n\n**Recent form:** the last five consecutive team-match records. A recorded non-appearance stays in the window; an absent record is not assumed to mean zero minutes.')
    with st.expander('Why are some figures missing?'):
        st.write('A player with zero minutes has no meaningful per-90 rate for that match. Recent rates require five complete records and 180 minutes; a trend needs two such periods. Older seasons lack team names, and ambiguous player identities are excluded. Missing values are never presented as zero.')
    with st.expander('What can this dashboard tell me?'):
        st.write('It describes patterns in the supplied historical records. More minutes do not prove a player was medically available, and a falling statistic does not establish declining ability. Position, opponents and limited playing time affect comparisons. No overall player rating is invented.')
    with st.expander('Where does the data come from?'):
        st.write('A user-supplied historical Premier League/FPL match file, covering selected seasons from 2016–17 to 2023–24. Its completeness and upstream source have not been independently verified. No live feed is connected.')
    with st.expander('Technical data-quality report'):
        st.write('The project preserves the original data, validates match records, and calculates summary tables. These layers are called Bronze, Silver and Gold.')
        if (ROOT/'outputs/audit.json').exists():
            st.json(json.loads((ROOT/'outputs/audit.json').read_text()))
