import streamlit as st
import pandas as pd
import requests
import os
import json
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Get current Gameweek from API
def get_current_gameweek():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    res = requests.get(url).json()
    events = res['events']
    for ev in events:
        if ev['is_current']:
            return ev['id']
    return None


def calculate_score(row):
    pos = row['position']

    if pos == "Goalkeeper":
        return (
                row['points_per_game'] * 0.25 +
                row['value_per_million'] * 0.15 +
                row['clean_sheets'] * 0.15 +
                row['saves'] * 0.1 +
                row['bps'] * 0.1 +
                row['bonus'] * 0.15 +
                row['selected_by_percent'] * 0.05 +
                row['minutes'] / 1000 * 0.05 +
                (5 - row['next_fixture_difficulty']) * 0.1
        )
    elif pos == "Defender":
        return (
                row['points_per_game'] * 0.25 +
                row['value_per_million'] * 0.15 +
                row['clean_sheets'] * 0.15 +
                float(row['threat']) * 0.1 +
                row['bps'] * 0.1 +
                row['bonus'] * 0.15 +
                row['selected_by_percent'] * 0.05 +
                row['minutes'] / 1000 * 0.05 +
                (5 - row['next_fixture_difficulty']) * 0.1
        )
    elif pos == "Midfielder":
        return (
                row['points_per_game'] * 0.2 +
                row['value_per_million'] * 0.15 +
                row['goals_scored'] * 0.15 +
                row['assists'] * 0.15 +
                float(row['ict_index']) * 0.1 +
                row['bonus'] * 0.1 +
                row['selected_by_percent'] * 0.1 +
                row['minutes'] / 1000 * 0.05 +
                (5 - row['next_fixture_difficulty']) * 0.1

        )
    elif pos == "Forward":
        return (
                row['points_per_game'] * 0.2 +
                row['value_per_million'] * 0.15 +
                row['goals_scored'] * 0.2 +
                float(row['ict_index']) * 0.1 +
                row['bonus'] * 0.15 +
                row['assists'] * 0.1 +
                row['selected_by_percent'] * 0.05 +
                row['minutes'] / 1000 * 0.05 +
                (5 - row['next_fixture_difficulty']) * 0.1
        )
    else:
        return 0

@st.cache_data
def get_fixtures():
    fixtures = requests.get("https://fantasy.premierleague.com/api/fixtures/").json()
    return pd.DataFrame(fixtures)

def avg_next_fixture_difficulty(players_df, teams_df, fixtures_df, n_gws=3):
    # Map team id -> name
    team_id_name = dict(zip(teams_df['id'], teams_df['name']))
    # Only scheduled fixtures with GW numbers
    fx = fixtures_df.dropna(subset=['event'])[['event','team_h','team_a','team_h_difficulty','team_a_difficulty','finished']].copy()
    fx['event'] = fx['event'].astype(int)

    # Work out the next GW (first un-finished gw)
    next_gw = fx.loc[fx['finished'] == False, 'event'].min()
    if pd.isna(next_gw):  # late season fallback
        next_gw = fx['event'].max()

    target_gws = [gw for gw in sorted(fx['event'].unique()) if gw >= next_gw][:n_gws]
    if not target_gws:  # fallback
        target_gws = [fx['event'].max()]

    # Build difficulty list per team across those GWs
    diffs = {tid: [] for tid in teams_df['id']}
    for _, r in fx[fx['event'].isin(target_gws)].iterrows():
        diffs[r['team_h']].append(r['team_h_difficulty'])
        diffs[r['team_a']].append(r['team_a_difficulty'])

    # Average difficulty per team; if empty, set to 3 (neutral)
    avg_diff = {tid: (sum(v)/len(v) if len(v)>0 else 3.0) for tid, v in diffs.items()}
    players_df = players_df.copy()
    players_df['avg_next_fixture_difficulty'] = players_df['team'].map(avg_diff).astype(float)
    return players_df, target_gws


# Load data from FPL API
@st.cache_data
def load_data():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    data = requests.get(url).json()
    players = pd.DataFrame(data['elements'])
    teams = pd.DataFrame(data['teams'])
    positions = pd.DataFrame(data['element_types'])

    fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
    fixtures = requests.get(fixtures_url).json()
    fixtures_df = pd.DataFrame(fixtures)
    next_gw = fixtures_df[fixtures_df['finished'] == False]['event'].min()

    # Get upcoming fixtures for next GW
    upcoming = fixtures_df[fixtures_df['event'] == next_gw]

    # Build a mapping: team_id ➝ difficulty
    team_difficulty = {}

    for _, row in upcoming.iterrows():
        team_difficulty[row['team_h']] = row['team_h_difficulty']
        team_difficulty[row['team_a']] = row['team_a_difficulty']

    players['team_name'] = players['team'].apply(lambda x: teams.loc[x-1, 'name'])
    players['position'] = players['element_type'].apply(lambda x: positions.loc[x-1, 'singular_name'])

    players['points_per_game'] = players['points_per_game'].astype(float)
    players['form'] = players['form'].astype(float)
    players['selected_by_percent'] = players['selected_by_percent'].astype(float)
    players['value_per_million'] = players['total_points'] / players['now_cost']
    players['next_fixture_difficulty'] = players['team'].map(team_difficulty)
    players['score'] = players.apply(calculate_score, axis=1)
    scaler = MinMaxScaler(feature_range=(0, 100))
    players['score_normalized'] = scaler.fit_transform(players[['score']])

    fixtures_url = "https://fantasy.premierleague.com/api/fixtures/"
    fixtures = requests.get(fixtures_url).json()
    fixtures_df = pd.DataFrame(fixtures)
    fixtures_df = fixtures_df[['event', 'team_h', 'team_a', 'team_h_difficulty', 'team_a_difficulty', 'finished']]
    fixtures_df = fixtures_df.dropna(subset=['event'])  # drop blank GWs
    fixtures_df['event'] = fixtures_df['event'].astype(int)

    # Team mapping (ID to name)
    team_id_name = dict(zip(teams['id'], teams['name']))
    # Build fixture matrix with team difficulty and opponent name
    # Determine next upcoming gameweek
    current_gw = fixtures_df[fixtures_df['event'].notnull() & (fixtures_df['team_h_difficulty'].notnull()) & (
        fixtures_df['team_a_difficulty'].notnull())]
    unfinished = current_gw.groupby('event').filter(lambda x: len(x) < 10)  # unfinished gameweek (some matches left)
    next_gw = fixtures_df[fixtures_df['finished'] == False]['event'].min()

    # Get next 5 upcoming gameweeks starting from next_gw
    gameweeks = sorted(fixtures_df['event'].unique())
    upcoming_gws = [gw for gw in gameweeks if gw >= next_gw][:5]

    team_names = list(team_id_name.values())

    fixture_matrix_difficulty = pd.DataFrame(index=team_names, columns=upcoming_gws)
    fixture_matrix_labels = pd.DataFrame(index=team_names, columns=upcoming_gws)

    for _, row in fixtures_df.iterrows():
        gw = row['event']
        if gw in upcoming_gws:
            home_team = team_id_name[row['team_h']]
            away_team = team_id_name[row['team_a']]

            # Home team
            fixture_matrix_difficulty.loc[home_team, gw] = row['team_h_difficulty']
            fixture_matrix_labels.loc[home_team, gw] = f"{team_id_name[row['team_a']]}(H)"

            # Away team
            fixture_matrix_difficulty.loc[away_team, gw] = row['team_a_difficulty']
            fixture_matrix_labels.loc[away_team, gw] = f"{team_id_name[row['team_h']]}(A)"

    players = players[players['status'] == 'a']
    return players, fixture_matrix_difficulty, fixture_matrix_labels, teams

# Load and process
players_df, fixture_matrix_difficulty, fixture_matrix_labels, teams = load_data()

# --- UI Setup ---
st.set_page_config(page_title="EPL Fantasy Dashboard", layout="wide")
st.title("⚽ Fantasy Premier League Analytics Dashboard")
st.markdown("Analyze FPL players with custom scoring logic and live data.")

# Sidebar Filters
st.sidebar.header("Filters")
position_filter = st.sidebar.multiselect("Select Positions", players_df['position'].unique(), default=players_df['position'].unique())
team_filter = st.sidebar.multiselect("Select Teams", players_df['team_name'].unique(), default=players_df['team_name'].unique())
top_n = st.sidebar.slider("Top N Players", 5, 50, 20)

filtered = players_df[
    (players_df['position'].isin(position_filter)) &
    (players_df['team_name'].isin(team_filter))
]

current_gw = get_current_gameweek()
snapshot_file = f"{SNAPSHOT_DIR}/GW_{current_gw}.csv"

st.sidebar.markdown("---")
st.sidebar.markdown("📂 **Weekly Snapshot Viewer**")
snapshot_files = sorted(os.listdir(SNAPSHOT_DIR))
selected_snapshot = st.sidebar.selectbox("Choose Gameweek Snapshot", snapshot_files)

if selected_snapshot:
    snap_df = pd.read_csv(os.path.join(SNAPSHOT_DIR, selected_snapshot))
    st.markdown(f"### 📊 Snapshot: {selected_snapshot}")
    st.dataframe(snap_df.head(20), use_container_width=True)

# Take weekly snapshot only if not already taken
if not os.path.exists(snapshot_file):
    snapshot_df = filtered[['web_name', 'position', 'team_name', 'now_cost', 'total_points',
                            'points_per_game', 'form', 'selected_by_percent',
                            'value_per_million', 'score']].sort_values(by='score', ascending=False)
    snapshot_df.to_csv(snapshot_file, index=False)
    st.success(f"📁 Gameweek {current_gw} snapshot saved!")
else:
    st.info(f"✅ Snapshot for Gameweek {current_gw} already exists.")

# Display Metrics
st.markdown(f"### 🏆 Top {top_n} Players by Normalized Score")
st.dataframe(
    filtered[['web_name', 'position', 'team_name', 'now_cost', 'total_points',
              'points_per_game', 'form', 'selected_by_percent', 'value_per_million',
              'score_normalized']]
    .sort_values(by='score_normalized', ascending=False)
    .head(top_n),
    use_container_width=True
)

st.markdown("### 🧩 Top 5 Players per Position")
top_by_position = (
    players_df.sort_values(by="score_normalized", ascending=False)
    .groupby("position")
    .head(5)
    .sort_values(by=["position", "score_normalized"], ascending=[True, False])
)

st.dataframe(top_by_position[['web_name', 'position', 'team_name', 'score_normalized']], use_container_width=True)


# --- Visuals ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📈 Score vs Value")
    fig1 = px.scatter(
        filtered,
        x="value_per_million",
        y="score",
        color="position",
        hover_name="web_name",
        title="Score vs Value per Million",
        labels={"value_per_million": "Points / Million", "score_normalized": "Custom Score"}
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### 🔥 Form vs Total Points")
    fig2 = px.scatter(
        filtered,
        x="form",
        y="total_points",
        color="position",
        hover_name="web_name",
        title="Form vs Total Points",
        labels={"form": "Current Form", "total_points": "Total Points"}
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("### 📊 Normalized Score Distribution by Position")
fig_pos = px.box(
    players_df,
    x="position",
    y="score_normalized",
    points="all",  # show all individual player dots
    hover_name="web_name",  # display player name on hover
    color="position",
    title="Score Distribution (Normalized to 0–100)",
    labels={"score_normalized": "Normalized Score (0–100)"}
)
st.plotly_chart(fig_pos, use_container_width=True)

fig = px.bar(
    filtered.sort_values("next_fixture_difficulty"),
    x="web_name",
    y="next_fixture_difficulty",
    color="position",
    title="Next Fixture Difficulty for Top Players"
)
st.plotly_chart(fig)

fixture_matrix_difficulty.fillna(0, inplace=True)
fixture_matrix_labels.fillna("–", inplace=True)

fig = ff.create_annotated_heatmap(
    z=fixture_matrix_difficulty.values.astype(float),
    x=list(fixture_matrix_difficulty.columns.astype(str)),
    y=list(fixture_matrix_difficulty.index),
    annotation_text=fixture_matrix_labels.values.astype(str),
    colorscale=[  # green to red scale
        [0.0, "#2ECC71"],  # green (easy)
        [0.25, "#F1C40F"],  # yellow
        [0.5, "#E67E22"],  # orange
        [0.75, "#E74C3C"],  # red
        [1.0, "#C0392B"]   # dark red (hardest)
    ],
    showscale=True,
    reversescale=False
)

fig.update_layout(
    title="📅 Fixture Difficulty Heatmap (Next 5 GWs)",
    xaxis_title="Gameweek",
    yaxis_title="Team",
    height=700,  # Reduce to make more compact if needed
    font=dict(size=12)
)

st.plotly_chart(fig, use_container_width=True)

# Summary Stats
st.markdown("### 📌 Summary Insights")
avg_score = filtered['score'].mean()
avg_ppg = filtered['points_per_game'].mean()
avg_form = filtered['form'].mean()
avg_ppm = filtered['value_per_million'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg. Score", f"{avg_score:.2f}")
col2.metric("Avg. Points/Game", f"{avg_ppg:.2f}")
col3.metric("Avg. Form", f"{avg_form:.2f}")
col4.metric("Avg. Points/Million", f"{avg_ppm:.2f}")


st.markdown("## 🛠️ Build Your Own Custom Score")

# Select stats
available_stats = [
    'total_points', 'points_per_game', 'form', 'minutes',
    'value_season', 'value_form', 'now_cost', 'selected_by_percent',
    'goals_scored', 'assists',
    'creativity', 'threat', 'ict_index',
    'clean_sheets', 'goals_conceded', 'saves', 'own_goals',
    'yellow_cards', 'red_cards',
    'bonus', 'bps',
    'transfers_in_event', 'transfers_out_event'
]


selected_stats = st.multiselect("Select stats to include in your score:", available_stats)

weights = {}
total_weight = 0

if selected_stats:
    st.markdown("### 🔢 Assign Weights to Each Stat (sum should be ~1.0)")
    for stat in selected_stats:
        weight = st.slider(f"Weight for {stat}", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        weights[stat] = weight
        total_weight += weight

    if total_weight == 0:
        st.warning("Total weight cannot be 0. Please assign weights.")
    else:
        # Calculate custom score
        players_df['custom_score'] = 0
        for stat, wt in weights.items():
            players_df['custom_score'] += players_df[stat].astype(float) * wt

        # Normalize (optional)
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 100))
        players_df['custom_score_normalized'] = scaler.fit_transform(players_df[['custom_score']])

        st.markdown("### 🏆 Top Players by Your Custom Score")
        top_custom = players_df[['web_name', 'position', 'team_name', 'custom_score_normalized'] + selected_stats]\
            .sort_values(by='custom_score_normalized', ascending=False).head(20)
        st.dataframe(top_custom, use_container_width=True)

        # Optional plot
        st.markdown("### 📈 Custom Score Distribution")
        import plotly.express as px
        fig_custom = px.box(
            players_df,
            x="position",
            y="custom_score_normalized",
            points="all",
            color="position",
            hover_name="web_name",
            title="Custom Score by Position (Normalized 0–100)",
            labels={"custom_score_normalized": "Custom Score"}
        )
        st.plotly_chart(fig_custom, use_container_width=True)


st.markdown("## 🧭 Differential Finder")

# Controls
colA, colB, colC, colD = st.columns(4)
max_own = colA.slider("Max ownership %", 0.0, 30.0, 10.0, 0.5)
min_minutes = colB.slider("Min minutes", 0, 3000, 600, 10)
n_gws = colC.slider("Upcoming GWs to consider", 1, 6, 3)
top_n = colD.slider("Show Top N", 5, 50, 20)

pos_options = players_df['position'].unique().tolist()
chosen_pos = st.multiselect("Positions", pos_options, default=pos_options)

# Weights
st.markdown("#### Weights (should roughly sum to 1.0)")
w_col1, w_col2, w_col3, w_col4 = st.columns(4)
w_form = w_col1.slider("Form", 0.0, 1.0, 0.30, 0.01)
w_ict  = w_col2.slider("ICT Index", 0.0, 1.0, 0.25, 0.01)
w_ppg  = w_col3.slider("Points/Game", 0.0, 1.0, 0.25, 0.01)
w_fix  = w_col4.slider("Fixture Ease", 0.0, 1.0, 0.20, 0.01)

# Ensure fixture difficulty is present for next N GWs
fixtures_df = get_fixtures()
players_aug, target_gws = avg_next_fixture_difficulty(players_df, teams, fixtures_df, n_gws=n_gws)
st.caption(f"Using upcoming GWs: {', '.join(map(str, target_gws))}")

# Clean types
for col in ["form", "ict_index", "points_per_game", "selected_by_percent", "minutes"]:
    if col in players_aug.columns:
        players_aug[col] = pd.to_numeric(players_aug[col], errors='coerce')

# Filter for differentials + availability
diff_pool = players_aug[
    (players_aug['position'].isin(chosen_pos)) &
    (players_aug['selected_by_percent'] <= max_own) &
    (players_aug['minutes'] >= min_minutes) &
    (players_aug['status'] == 'a')
].copy()

# Compute Differential Score:
# fixture ease = invert difficulty (1 easy .. 5 hard) -> ease = 6 - diff; averaged already
diff_pool['fixture_ease'] = 6 - diff_pool['avg_next_fixture_difficulty']

# Fill NaNs safely
for c in ["form", "ict_index", "points_per_game", "fixture_ease"]:
    diff_pool[c] = diff_pool[c].fillna(0)

diff_pool['differential_score_raw'] = (
    diff_pool['form'] * w_form +
    diff_pool['ict_index'] * w_ict +
    diff_pool['points_per_game'] * w_ppg +
    diff_pool['fixture_ease'] * w_fix
)

# Normalize 0–100 for display
scaler = MinMaxScaler(feature_range=(0, 100))
diff_pool['differential_score'] = scaler.fit_transform(diff_pool[['differential_score_raw']])

# Table
st.markdown("### 🏁 Top Differentials (Ranked)")
cols_show = [
    'web_name', 'position', 'team_name', 'selected_by_percent',
    'form', 'ict_index', 'points_per_game', 'avg_next_fixture_difficulty',
    'differential_score'
]
table = diff_pool[cols_show].sort_values('differential_score', ascending=False).head(top_n)
st.dataframe(table, use_container_width=True)

# Scatter: ownership vs differential score (hover player)
st.markdown("#### 📈 Ownership vs Differential Score (hover for names)")
fig = px.scatter(
    diff_pool,
    x="selected_by_percent",
    y="differential_score",
    color="position",
    hover_name="web_name",
    hover_data={
        "team_name": True,
        "form": ':.2f',
        "ict_index": ':.2f',
        "points_per_game": ':.2f',
        "avg_next_fixture_difficulty": ':.2f',
        "selected_by_percent": ':.2f'
    },
    labels={
        "selected_by_percent": "Ownership (%)",
        "differential_score": "Differential Score (0–100)"
    },
)
fig.update_yaxes(range=[0, 100])
st.plotly_chart(fig, use_container_width=True)

