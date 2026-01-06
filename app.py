import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
from datetime import datetime

# ======================
# APP CONFIG
# ======================
st.set_page_config(
    page_title="DFS Pro Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# USER TIERS
# ======================
TIERS = {
    "FREE": {"lineups": 3, "sims": False},
    "PRO": {"lineups": 20, "sims": True},
    "ELITE": {"lineups": 50, "sims": True}
}

ACCESS_KEY = st.sidebar.text_input("Access Key", type="password")
TIER = "FREE"
if ACCESS_KEY == "pro123":
    TIER = "PRO"
elif ACCESS_KEY == "elite123":
    TIER = "ELITE"

st.sidebar.success(f"Tier: {TIER}")

# ======================
# SPORT / SLATE
# ======================
SPORT = st.sidebar.selectbox("Sport", ["NBA","NFL","MLB","NHL","SOCCER"])
SLATE = st.sidebar.selectbox("Slate", ["Classic","Showdown"])
SALARY_CAP = 50000

ROSTERS = {
    "NBA": ["PG","SG","SF","PF","C","G","F","UTIL"],
    "NFL": ["QB","RB","RB","WR","WR","WR","TE","FLEX","DST"],
    "MLB": ["P","P","C","1B","2B","3B","SS","OF","OF","OF"],
    "NHL": ["C","C","W","W","W","D","D","G","UTIL"],
    "SOCCER": ["GK","D","D","M","M","F","F","UTIL"]
}

ROSTER = ["CPT","UTIL","UTIL","UTIL","UTIL","UTIL"] if SLATE=="Showdown" else ROSTERS[SPORT]

# ======================
# FILE UPLOADS
# ======================
players_file = st.sidebar.file_uploader("Player Pool CSV", type="csv")
vegas_file = st.sidebar.file_uploader("Vegas Totals CSV", type="csv")

# ======================
# NORMALIZE
# ======================
def normalize(df):
    df.columns = [c.lower().strip() for c in df.columns]

    df = df.rename(columns={
        "name":"player",
        "salary":"salary",
        "position":"position",
        "teamabbrev":"team",
        "avgpointspergame":"projection",
        "game time":"game_time"
    })

    for c in ["salary","projection"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "ownership" not in df.columns:
        df["ownership"] = 5.0
    else:
        df["ownership"] = pd.to_numeric(df["ownership"], errors="coerce").fillna(5)

    if "game_time" in df.columns:
        df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    return df

# ======================
# VEGAS TOTALS
# ======================
team_totals = {}
if vegas_file:
    v = pd.read_csv(vegas_file)
    team_totals = dict(zip(v["team"], v["implied_total"]))

def vegas_bonus(team):
    return team_totals.get(team, 0) * 0.04

# ======================
# CORRELATION BONUS
# ======================
def correlation_bonus(lineup):
    bonus = 0
    teams = {}

    for p in lineup:
        teams.setdefault(p["team"], []).append(p)

    for team, players in teams.items():
        count = len(players)

        if SPORT in ["NBA","CBB"] and count >= 3:
            bonus += count * 2

        if SPORT == "NFL":
            qb = any("QB" in p["position"] for p in players)
            wr = any("WR" in p["position"] or "TE" in p["position"] for p in players)
            if qb and wr:
                bonus += 8

        if SPORT == "MLB" and count >= 4:
            bonus += count * 3

        if SPORT == "NHL" and count >= 2:
            bonus += count * 2

        if SPORT == "SOCCER" and count >= 2:
            bonus += count * 2

    return bonus

# ======================
# LATE SWAP LOCKS
# ======================
def locked_players(df, now):
    locked = []
    if "game_time" not in df.columns:
        return locked

    for _, r in df.iterrows():
        if pd.notna(r["game_time"]) and r["game_time"] <= now:
            locked.append(r["player"])
    return locked

# ======================
# OPTIMIZER
# ======================
def build_lineup(df, locked, excluded):
    prob = LpProblem("DFS", LpMaximize)
    x = LpVariable.dicts("p", df.index, 0, 1, LpBinary)

    prob += lpSum(
        x[i] * (
            df.loc[i,"projection"]
            + vegas_bonus(df.loc[i,"team"])
            - df.loc[i,"ownership"] * 0.35
        )
        for i in df.index
    )

    prob += lpSum(x[i] * df.loc[i,"salary"] for i in df.index) <= SALARY_CAP
    prob += lpSum(x[i] for i in df.index) == len(ROSTER)

    for i in df.index:
        if df.loc[i,"player"] in excluded:
            prob += x[i] == 0
        if df.loc[i,"player"] in locked:
            prob += x[i] == 1

    prob.solve(PULP_CBC_CMD(msg=False))
    lineup = df.loc[[i for i in df.index if x[i].value()==1]].to_dict("records")

    if SLATE == "Showdown" and lineup:
        cpt = max(lineup, key=lambda p: p["projection"])
        cpt["salary"] *= 1.5
        cpt["projection"] *= 1.5

    if sum(p["salary"] for p in lineup) > SALARY_CAP:
        return None

    return lineup

# ======================
# UI CONTROLS
# ======================
st.header("Portfolio Builder")

if players_file:
    df = normalize(pd.read_csv(players_file))

    excluded = st.multiselect("Exclude Players", df["player"].tolist())
    now = st.time_input("Current Time (Late Swap)", datetime.now().time())

    locked = locked_players(df, datetime.combine(datetime.today(), now))

    max_lineups = TIERS[TIER]["lineups"]
    n = st.slider("Lineups", 1, max_lineups, min(10, max_lineups))

    if st.button("BUILD PORTFOLIO"):
        rows = []

        for i in range(n):
            l = build_lineup(df, locked, excluded)
            if not l:
                continue

            r = {
                "Lineup": i + 1,
                "Salary": int(sum(p["salary"] for p in l)),
                "Projection": round(sum(p["projection"] for p in l),2),
                "Correlation": correlation_bonus(l)
            }

            for j,p in enumerate(l):
                r[f"P{j+1}"] = p["player"]

            rows.append(r)

        out = pd.DataFrame(rows).sort_values("Projection", ascending=False)
        st.dataframe(out, use_container_width=True)
        st.download_button("Download DK CSV", out.to_csv(index=False), "dfs_portfolio.csv")
