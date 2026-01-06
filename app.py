import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
from datetime import datetime, time
import random

# ======================
# APP CONFIG
# ======================
st.set_page_config("DFS Pro Engine", layout="wide")
SALARY_CAP = 50000

# ======================
# SIDEBAR CONTROLS
# ======================
st.sidebar.title("DFS PRO ENGINE")

SPORT = st.sidebar.selectbox(
    "Sport", ["NBA","NFL","MLB","NHL","SOCCER"]
)

SLATE = st.sidebar.selectbox(
    "Slate Type", ["Classic","Showdown"]
)

players_file = st.sidebar.file_uploader(
    "DraftKings Salaries CSV", type="csv"
)

lineups_n = st.sidebar.slider(
    "MME Lineups", 1, 300, 50
)

max_exposure = st.sidebar.slider(
    "Max Player Exposure %", 10, 100, 40
)

max_overlap = st.sidebar.slider(
    "Max Lineup Overlap", 3, 8, 6
)

rand_pct = st.sidebar.slider(
    "Projection Randomness %", 0, 20, 8
)

enable_late = st.sidebar.checkbox("Enable Late Swap")
current_time = None
if enable_late:
    current_time = st.sidebar.time_input("Current Time (ET)")

# ======================
# ROSTER SIZES
# ======================
ROSTER_SIZES = {
    "NBA": 8,
    "NFL": 9,
    "MLB": 10,
    "NHL": 9,
    "SOCCER": 8
}
ROSTER_SIZE = 6 if SLATE=="Showdown" else ROSTER_SIZES[SPORT]

# ======================
# DATA NORMALIZATION (DK SAFE)
# ======================
def normalize(df):
    df.columns = [c.lower().strip() for c in df.columns]

    rename = {
        "name":"player",
        "salary":"salary",
        "position":"position",
        "teamabbrev":"team",
        "avgpointspergame":"projection",
        "game info":"game",
        "gametime":"game_time"
    }
    df = df.rename(columns=rename)

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0)
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce").fillna(0)

    if "ownership" not in df.columns:
        df["ownership"] = 5.0

    if "game_time" in df.columns:
        df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    return df

# ======================
# GAME SELECTOR UI
# ======================
def get_games(df):
    if "game" not in df.columns:
        return []
    return sorted(df["game"].dropna().unique())

# ======================
# STACK RULES
# ======================
def stack_bonus(lineup):
    teams = {}
    for p in lineup:
        teams[p["team"]] = teams.get(p["team"],0)+1

    bonus = 0
    for c in teams.values():
        if SPORT=="MLB" and c>=4:
            bonus += c * 5
        if SPORT in ["NBA","NHL"] and c>=3:
            bonus += c * 4
        if SPORT=="NFL" and c>=2:
            bonus += c * 6
        if SPORT=="SOCCER" and c>=3:
            bonus += c * 4
    return bonus

# ======================
# SIMULATION
# ======================
def simulate_lineup(lineup):
    return sum(
        np.random.normal(p["projection"], p["projection"]*0.30)
        for p in lineup
    ) + stack_bonus(lineup)

def contest_sim(lineups, sims=500):
    results = [{"win":0,"top1":0,"cash":0} for _ in lineups]

    for _ in range(sims):
        scores = [simulate_lineup(l) for l in lineups]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        for r,i in enumerate(ranked):
            if r == 0:
                results[i]["win"] += 1
            if r < max(1,len(scores)//100):
                results[i]["top1"] += 1
            if r < len(scores)//4:
                results[i]["cash"] += 1

    for r in results:
        for k in r:
            r[k] /= sims

    return results

# ======================
# OPTIMIZER (TRUE MME)
# ======================
def build_lineup(df, used, prev_sets, locked):
    prob = LpProblem("DFS", LpMaximize)
    x = LpVariable.dicts("p", df.index, 0, 1, LpBinary)

    rand_proj = {
        i: df.loc[i,"projection"] * (1 + random.uniform(-rand_pct/100, rand_pct/100))
        for i in df.index
    }

    prob += lpSum(
        x[i] * (
            rand_proj[i]
            - df.loc[i,"ownership"]*0.4
            - used.get(df.loc[i,"player"],0)*0.3
        )
        for i in df.index
    )

    prob += lpSum(x[i]*df.loc[i,"salary"] for i in df.index) <= SALARY_CAP
    prob += lpSum(x[i] for i in df.index) == ROSTER_SIZE

    for p in locked:
        prob += lpSum(x[i] for i in df.index if df.loc[i,"player"]==p)==1

    for s in prev_sets:
        prob += lpSum(x[i] for i in df.index if df.loc[i,"player"] in s) <= max_overlap

    prob.solve(PULP_CBC_CMD(msg=False))

    return df.loc[[i for i in df.index if x[i].value()==1]].to_dict("records")

# ======================
# RUN
# ======================
if players_file:
    df = normalize(pd.read_csv(players_file))

    games = get_games(df)
    selected_game = st.selectbox("Game Stack (optional)", ["Any"] + games)

    if selected_game != "Any":
        df = df[df["game"] == selected_game]

    if st.button("BUILD PORTFOLIO"):
        used = {}
        prev_sets = []
        lineups = []

        locked_players = []
        if enable_late and current_time:
            now = datetime.combine(datetime.today(), current_time)
            locked_players = df[df["game_time"] <= now]["player"].tolist()

        for _ in range(lineups_n):
            l = build_lineup(df, used, prev_sets, locked_players)
            if not l:
                continue

            prev_sets.append({p["player"] for p in l})
            for p in l:
                used[p["player"]] = used.get(p["player"],0)+1

            lineups.append(l)

        sims = contest_sim(lineups)

        rows = []
        for i,l in enumerate(lineups):
            r = {
                "Lineup": i+1,
                "Salary": sum(p["salary"] for p in l),
                "Projection": round(sum(p["projection"] for p in l),2),
                "Win %": round(sims[i]["win"]*100,2),
                "Top 1%": round(sims[i]["top1"]*100,2),
                "Cash %": round(sims[i]["cash"]*100,2)
            }
            for j,p in enumerate(l):
                r[f"P{j+1}"] = p["player"]
            rows.append(r)

        out = pd.DataFrame(rows).sort_values("Win %", ascending=False)
        st.dataframe(out, use_container_width=True)
        st.download_button("Download DK CSV", out.to_csv(index=False), "dfs_mme_portfolio.csv")
