import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
import random

# ======================
# APP CONFIG
# ======================
st.set_page_config(page_title="DFS PRO ENGINE", layout="wide")

PASSWORD = "dfs123"
if st.text_input("Password", type="password") != PASSWORD:
    st.stop()

# ======================
# SLATE SETUP
# ======================
SPORT = st.selectbox("Sport", ["NBA","NFL","MLB","NHL","CBB","SOCCER"])
SLATE = st.selectbox("Slate Type", ["Classic","Showdown"])
SALARY_CAP = 50000

ROSTERS = {
    "NBA": ["PG","SG","SF","PF","C","G","F","UTIL"],
    "NFL": ["QB","RB","RB","WR","WR","WR","TE","FLEX","DST"],
    "MLB": ["P","P","C","1B","2B","3B","SS","OF","OF","OF"],
    "NHL": ["C","C","W","W","W","D","D","G","UTIL"],
    "CBB": ["G","G","G","F","F","UTIL","UTIL","UTIL"],
    "SOCCER": ["GK","D","D","M","M","F","F","UTIL"]
}

ROSTER = ["CPT","UTIL","UTIL","UTIL","UTIL","UTIL"] if SLATE=="Showdown" else ROSTERS[SPORT]

# ======================
# FILE UPLOAD
# ======================
uploaded = st.file_uploader("Upload DraftKings CSV", type="csv")

# ======================
# SIDEBAR RULES
# ======================
st.sidebar.header("Rules & Controls")

exclude_players = st.sidebar.text_area(
    "Exclude Players (comma-separated)",
    help="Injured / Out players"
)

force_group = st.sidebar.text_area(
    "Force Group (at least one)",
    help="Comma-separated names"
)

avoid_group = st.sidebar.text_area(
    "Avoid Group (never together)",
    help="Comma-separated names"
)

enable_bringback = st.sidebar.checkbox("Enable Bring-Back")
lineups_n = st.sidebar.slider("Lineups", 1, 300, 50)
max_overlap = st.sidebar.slider("Max Player Overlap", 2, 8, 6)
sims = st.sidebar.slider("Contest Sims", 300, 3000, 800)

# ======================
# NORMALIZE DK CSV
# ======================
def normalize(df):
    df.columns = [c.lower().strip() for c in df.columns]

    rename_map = {
        "name":"player",
        "salary":"salary",
        "avgpointspergame":"projection",
        "teamabbrev":"team",
        "position":"position"
    }
    df = df.rename(columns=rename_map)

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0)
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce").fillna(0)
    df["ownership"] = 5.0

    return df

# ======================
# POSITION VALIDATION
# ======================
def valid(pos, slot):
    if slot in ["UTIL","FLEX"]:
        return True
    if SLATE=="Showdown":
        return True
    return slot in pos

# ======================
# STACK BONUS
# ======================
def stack_bonus(lineup):
    teams = {}
    bonus = 0
    for p in lineup:
        teams[p["team"]] = teams.get(p["team"],0)+1

    for c in teams.values():
        if SPORT=="MLB" and c>=3: bonus+=5
        if SPORT in ["NBA","CBB","NHL"] and c>=2: bonus+=3
        if SPORT=="NFL" and c>=2: bonus+=4
    return bonus

# ======================
# CONTEST SIMULATION
# ======================
def simulate(lineup):
    score = 0
    for p in lineup:
        mu = p["projection"]
        score += np.random.normal(mu, mu*0.30)
    return score + stack_bonus(lineup)

def ev_score(lineup, sims):
    scores = sorted([simulate(lineup) for _ in range(sims)], reverse=True)
    ev = 0
    for i,s in enumerate(scores[:int(len(scores)*0.25)]):
        ev += s * 0.01
    return ev

# ======================
# OPTIMIZER
# ======================
def build_lineup(df, used_lineups):
    prob = LpProblem("DFS", LpMaximize)
    x = LpVariable.dicts("p", df.index, 0, 1, LpBinary)

    # Randomized projection for diversity
    rand_proj = {
        i: df.loc[i,"projection"] * random.uniform(0.85,1.15)
        for i in df.index
    }

    prob += lpSum(
        x[i]*(rand_proj[i] - df.loc[i,"ownership"]*0.35)
        for i in df.index
    )

    # Salary
    prob += lpSum(x[i]*df.loc[i,"salary"] for i in df.index) <= SALARY_CAP
    prob += lpSum(x[i] for i in df.index) == len(ROSTER)

    # Positions
    for slot in set(ROSTER):
        prob += lpSum(
            x[i] for i in df.index if valid(df.loc[i,"position"],slot)
        ) >= ROSTER.count(slot)

    # Exclusions
    excl = [p.strip() for p in exclude_players.split(",") if p.strip()]
    for p in excl:
        prob += lpSum(
            x[i] for i in df.index if df.loc[i,"player"]==p
        ) == 0

    # Force / Avoid
    forced = [p.strip() for p in force_group.split(",") if p.strip()]
    avoided = [p.strip() for p in avoid_group.split(",") if p.strip()]

    if forced:
        prob += lpSum(
            x[i] for i in df.index if df.loc[i,"player"] in forced
        ) >= 1

    if avoided:
        prob += lpSum(
            x[i] for i in df.index if df.loc[i,"player"] in avoided
        ) <= 1

    # Lineup diversity
    for prev in used_lineups:
        prob += lpSum(
            x[i] for i in prev
        ) <= max_overlap

    prob.solve(PULP_CBC_CMD(msg=False))

    idxs = [i for i in df.index if x[i].value()==1]
    return idxs

# ======================
# RUN
# ======================
if uploaded:
    df = normalize(pd.read_csv(uploaded))

    if st.button("BUILD PORTFOLIO"):
        used = []
        lineups = []

        for _ in range(lineups_n):
            idxs = build_lineup(df, used)
            if not idxs:
                break
            used.append(idxs)
            lineup = df.loc[idxs].to_dict("records")
            lineups.append(lineup)

        rows = []
        for i,l in enumerate(lineups):
            rows.append({
                "Lineup": i+1,
                "Salary": sum(p["salary"] for p in l),
                "Projection": round(sum(p["projection"] for p in l),2),
                "EV": round(ev_score(l, sims),2),
                **{f"P{j+1}":p["player"] for j,p in enumerate(l)}
            })

        out = pd.DataFrame(rows).sort_values("EV", ascending=False)

        st.dataframe(out, use_container_width=True)
        st.download_button(
            "Download DK Upload CSV",
            out[[c for c in out.columns if c.startswith("P")]].to_csv(index=False),
            "dk_upload.csv"
        )
