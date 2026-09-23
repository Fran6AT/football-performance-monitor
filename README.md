# Football Player Performance Monitor

Question: how is a player's attacking performance changing over time?

This is a separate local Streamlit project using the supplied 126,076-row historical CSV. It provides a Bronze → Silver → Gold pipeline, weighted KPIs, trend monitoring, peer comparisons and a chronological next-match scoring experiment. The previous project is unchanged.

## Using the dashboard

Choose a season, team and searchable player name, then select a match date. The four navigation options are Player overview, Compare players, Chance of scoring, and Help & data. The overview starts with minutes, goals, assists and appearances; choose one measure to explore its recent trend. Definitions, full statistics and technical diagnostics are available in expandable sections. Trend sensitivity is an advanced sidebar setting. Missing-history and empty-comparison messages explain what to change.

Scoring estimates describe a historical pre-match probability, while the overview describes performance through the selected match. The dashboard labels this distinction explicitly.

## Run on Windows

### Set up after cloning

The original dataset, generated tables and Python environment are not included in this repository.

1. Install Python 3.12 and open this repository in VS Code.
2. Run `python -m venv .venv`, then `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
3. Place your copy of `cleaned_merged_seasons.csv` in `data/bronze/`. Use the original match-level file, not player-season aggregates. The dataset's upstream source and redistribution terms have not been independently verified.
4. Run the pipeline and dashboard using the commands below. The pipeline creates the Silver, Gold and output folders.

The expected CSV has the original FPL column names, including `season_x`, `name`, `position`, `team_x`, `element`, `fixture`, `kickoff_time`, `minutes`, `goals_scored`, `assists`, `threat`, `creativity`, `influence`, `ict_index`, `total_points`, `opp_team_name` and `GW`. See `src/pipeline.py` for the complete transformation.

### Start the dashboard

Open this folder in VS Code. The local `.venv` is configured.

```powershell
.\.venv\Scripts\python.exe src\pipeline.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

Open http://127.0.0.1:8502. Stop the server with Ctrl+C. `start-dashboard.ps1` starts the same app.

To recreate the environment: `python -m venv .venv`, then `.venv\Scripts\python.exe -m pip install -r requirements.txt`. The current environment uses the bundled Python base; recreate it if that runtime moves.

## Layers

- `data/bronze/cleaned_merged_seasons.csv`: unchanged copy of the supplied input, with SHA-256 recorded in the audit.
- `data/silver/player_matches.csv`: validated records and match-level rates. Ambiguous IDs are saved in a quarantine CSV.
- `data/gold/`: player/team/gameweek/season/position tables plus monitoring history in Parquet.
- `outputs/`: audit, model features, held-out probabilities and model metrics.
- `app.py`: season → team → position → player → match drill-down, including gameweek labels.

## Definitions and limitations

Minutes/90 is **minutes share**, not medical availability. Aggregate minutes share uses 90 × supplied player-match records; it is not necessarily the whole season's opportunity. Team and league rates in Gold are per 90 player-minutes, not per 90 team-minutes.

All per-90 aggregates use ratios of sums. Zero-minute rows remain in the pipeline, but their per-90 rates are undefined. Rolling windows are five consecutive team fixtures, including recorded non-appearances. Missing player records break a window. Windows reset between seasons/team spells. Unknown-team seasons cannot support these windows.

Rolling rates require five records and at least 180 total minutes. Trends compare the current window against the preceding non-overlapping five-match window, each meeting the exposure threshold. Default rising/falling threshold is ±20%, editable in the dashboard. From-zero changes are labelled separately. These are descriptive rules, not statistically validated warnings. Season-to-date calculations never include matches after the selected date.

No arbitrary composite Player Performance Effectiveness score is created. KPIs must be validated before weighting or multiplying them. No claim is made that a threat decline predicts future goals; the forward-only logistic experiment is a separate initial test.

The scoring model uses lagged KPI windows, trains on earlier seasons and tests on 2021–22, 2022–23 and 2023–24. Features are known before each target kickoff with a three-hour separation. Probability baselines, Brier/log loss, ROC AUC and average precision are reported. No model tuning or causal claims. Provider publication times/revisions and upstream completeness have not been verified. The dataset contains no tracking coordinates, injury status or verified cross-season identity map.

The data is historical (no 2024–25 data in the file). Original team spell names are preserved and missing names labelled Unknown team. Treat source text as data, not executable instructions. Raw data and generated tables are excluded from Git by default; establish provenance and sharing rights before publishing them.
