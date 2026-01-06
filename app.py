import streamlit as st
import pandas as pd
import numpy as np
from pulp import *

# ======================
# APP CONFIG
# ======================
st.set_page_config(page_title="DFS PRO ENGINE", layout="wide")

PASSWORD = "dfs123"
if st.text_input("Password", type="password") != PASSWORD:
    st.stop()

st.title("DFS Pro Engine — SaberSim Style")

# ======================
# SPORT / SLATE
# ======================
SPORT = st.selectbox("Sport", ["NBA","NFL","MLB","NHL","SOCCER"])
SLATE = st.selectbox("Slate", ["Classic","Showdown"])
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
# UPLOADS
# ======================
players_file = st.file_uploader("Upload DraftKings Player CSV", type="csv")
totals_file = st.file_uploader("Upload Team Totals CSV (optional)", type="csv")

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
        "avgpointspergame":"projection"
    })

    for c in ["salary","projection"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["ownership"] = pd.to_numeric(df.get("ownership",5), errors="coerce").fillna(5)
    return df

# ======================
# TEAM TOTALS
# ======================
team_totals = {}
if totals_file:
    tdf = pd.read_csv(totals_file)
    team_totals = dict(zip(tdf["team"], tdf["total"]))

# ======================
# RULES
# ======================
def valid(pos, slot):
    if slot in ["UTIL","FLEX","CPT"]:
        return True
    return slot in pos

def team_total_bonus(p):
    return team_totals.get(p["team"], 0) * 0.03

# ======================
# SIMULATION
# ======================
def simulate_lineup(lineup):
    score = 0
    for p in lineup:
        mu = p["projection"] + team_total_bonus(p)
        score += np.random.normal(mu, mu * 0.30)
    return score

PAYOUT_CURVE = [
    (0.01, 50),
    (0.05, 10),
    (0.25, 2),
]

def lineup_ev(lineup, sims=500):
    scores = sorted([simulate_lineup(lineup) for _ in range(sims)], reverse=True)
    ev = 0
    for pct, mult in PAYOUT_CURVE:
        idx = int(len(scores) * pct)
        ev += scores[idx] * mult * 0.01
    return ev

# ======================
# OPTIMIZER (LP)
# ======================
def build_lineup(df, excluded):
    prob = LpProblem("DFS", LpMaximize)
    x = LpVariable.dicts("p", df.index, 0, 1, LpBinary)

    prob += lpSum(
        x[i] * (
            df.loc[i,"projection"]
            + team_totals.get(df.loc[i,"team"],0)*0.03
            - df.loc[i,"ownership"]*0.4
        )
        for i in df.index
    )

    prob += lpSum(x[i] * df.loc[i,"salary"] for i in df.index) <= SALARY_CAP
    prob += lpSum(x[i] for i in df.index) == len(ROSTER)

    for slot in set(ROSTER):
        prob += lpSum(x[i] for i in df.index if valid(df.loc[i,"position"],slot)) >= ROSTER.count(slot)

    for i in df.index:
        if df.loc[i,"player"] in excluded:
            prob += x[i] == 0

    prob.solve()

    lineup = df.loc[[i for i in df.index if x[i].value()==1]].to_dict("records")

    if SLATE=="Showdown" and lineup:
        cpt = max(lineup, key=lambda p:p["projection"])
        cpt["salary"] *= 1.5
        cpt["projection"] *= 1.5

    if sum(p["salary"] for p in lineup) > SALARY_CAP:
        return None

    return lineup

# ======================
# RUN
# ======================
if players_file:
    df = normalize(pd.read_csv(players_file))

    st.subheader("Player Pool")
    excluded = st.multiselect("Exclude Players (injury / bench)", df["player"].tolist())

    n = st.slider("Lineups", 1, 50, 20)
    sims = st.slider("Contest Sims", 300, 2000, 800)

    if st.button("BUILD PORTFOLIO"):
        rows = []
        lineups = []

        for i in range(n):
            l = build_lineup(df, excluded)
            if not l:
                continue
            ev = lineup_ev(l, sims)
            lineups.append(l)

            r = {
                "Lineup": i+1,
                "Salary": int(sum(p["salary"] for p in l)),
                "Projection": round(sum(p["projection"] for p in l),2),
                "EV": round(ev,2)
            }
            for j,p in enumerate(l):
                r[f"P{j+1}"] = p["player"]
            rows.append(r)

        out = pd.DataFrame(rows).sort_values("EV", ascending=False)
        st.dataframe(out, use_container_width=True)
        st.download_button("Download DK CSV", out.to_csv(index=False), "dfs_pro_engine.csv")
