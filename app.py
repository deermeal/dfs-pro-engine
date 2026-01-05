import streamlit as st
import pandas as pd
import numpy as np
from pulp import *

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="DFS PRO ENGINE", layout="wide")

PASSWORD = "dfs123"
if st.text_input("Password", type="password") != PASSWORD:
    st.stop()

SALARY_CAP = 50000

# ======================
# SPORT / SLATE
# ======================
SPORT = st.selectbox("Sport", ["NBA","NFL","CBB","NHL","MLB","SOCCER"])
SLATE = st.selectbox("Slate Type", ["Classic","Showdown"])

ROSTERS = {
    "NBA": ["PG","SG","SF","PF","C","G","F","UTIL"],
    "NFL": ["QB","RB","RB","WR","WR","WR","TE","FLEX","DST"],
    "CBB": ["G","G","G","F","F","UTIL","UTIL","UTIL"],
    "NHL": ["C","C","W","W","W","D","D","G","UTIL"],
    "MLB": ["P","P","C","1B","2B","3B","SS","OF","OF","OF"],
    "SOCCER": ["GK","D","D","M","M","F","F","UTIL"]
}

ROSTER = ["CPT","UTIL","UTIL","UTIL","UTIL","UTIL"] if SLATE=="Showdown" else ROSTERS[SPORT]

# ======================
# UPLOAD
# ======================
uploaded = st.file_uploader("Upload DraftKings CSV", type="csv")

def normalize(df):
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={
        "name":"player",
        "avgpointspergame":"projection",
        "salary":"salary",
        "position":"position",
        "teamabbrev":"team"
    })

    for c in ["salary","projection"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["ownership"] = 5.0  # default if none provided
    return df

# ======================
# RULE HELPERS
# ======================
def valid_position(player_pos, roster_slot):
    if roster_slot in ["UTIL","FLEX"]:
        return True
    if roster_slot == "CPT":
        return True
    return roster_slot in player_pos

def stack_bonus(lineup):
    teams = {}
    bonus = 0

    for p in lineup:
        teams[p["team"]] = teams.get(p["team"], 0) + 1

    for team, count in teams.items():
        if SPORT == "MLB" and count >= 3:
            bonus += count * 4
        elif SPORT in ["NBA","CBB","NHL"] and count >= 2:
            bonus += count * 2
        elif SPORT in ["NFL","SOCCER"] and count >= 2:
            bonus += count * 3

    return bonus

# ======================
# SIMULATION + EV
# ======================
def simulate_lineup(lineup):
    score = 0
    for p in lineup:
        mu = p["projection"]
        score += np.random.normal(mu, mu * 0.30)
    return score + stack_bonus(lineup)

PAYOUTS = [
    (0.01, 50),
    (0.05, 10),
    (0.25, 2)
]

def ev_score(lineup, sims=800):
    results = sorted(
        [simulate_lineup(lineup) for _ in range(sims)],
        reverse=True
    )
    ev = 0
    for pct, payout in PAYOUTS:
        idx = int(len(results) * pct)
        ev += results[idx] * payout * 0.01
    return ev

# ======================
# OPTIMIZER
# ======================
def build_lineup(df, exposure, max_lineups, excluded):
    prob = LpProblem("DFS", LpMaximize)
    x = LpVariable.dicts("p", df.index, 0, 1, LpBinary)

    # Objective: projection + leverage
    prob += lpSum(
        x[i] * (
            df.loc[i,"projection"]
            - df.loc[i,"ownership"] * 0.4
        )
        for i in df.index
    )

    # Salary (CPT costs 1.5x)
    prob += lpSum(
        x[i] * (
            df.loc[i,"salary"] * (1.5 if SLATE=="Showdown" else 1)
        )
        for i in df.index
    ) <= SALARY_CAP

    # Roster size
    prob += lpSum(x[i] for i in df.index) == len(ROSTER)

    # Position constraints
    for slot in set(ROSTER):
        prob += lpSum(
            x[i]
            for i in df.index
            if valid_position(df.loc[i,"position"], slot)
        ) >= ROSTER.count(slot)

    # Exposure control
    for player, count in exposure.items():
        if count / max_lineups > 0.6:
            prob += lpSum(
                x[i] for i in df.index if df.loc[i,"player"] == player
            ) == 0

    # Exclusions (injury / out)
    for i in df.index:
        if df.loc[i,"player"] in excluded:
            prob += x[i] == 0

    prob.solve(PULP_CBC_CMD(msg=False))

    lineup = df.loc[
        [i for i in df.index if x[i].value() == 1]
    ].to_dict("records")

    # Apply CPT multiplier AFTER optimization
    if SLATE == "Showdown" and lineup:
        cpt = max(lineup, key=lambda p: p["projection"])
        cpt["salary"] *= 1.5
        cpt["projection"] *= 1.5

    return lineup

# ======================
# RUN APP
# ======================
if uploaded:
    df = normalize(pd.read_csv(uploaded))

    excluded_players = st.multiselect(
        "Exclude Players (Injury / Out / Fade)",
        sorted(df["player"].unique())
    )

    n = st.slider("Number of Lineups", 1, 50, 20)
    sims = st.slider("Simulations per Lineup", 300, 2000, 800)
    max_exp = st.slider("Max Exposure %", 10, 100, 40)

    if st.button("BUILD LINEUPS"):
        exposure = {}
        portfolios = []

        for _ in range(n):
            lineup = build_lineup(df, exposure, n, excluded_players)
            for p in lineup:
                exposure[p["player"]] = exposure.get(p["player"], 0) + 1
            portfolios.append(lineup)

        rows = []
        for i, l in enumerate(portfolios):
            rows.append({
                "Lineup": i + 1,
                "Salary": round(sum(p["salary"] for p in l), 0),
                "Projection": round(sum(p["projection"] for p in l), 2),
                "Stack Bonus": stack_bonus(l),
                "EV": round(ev_score(l, sims), 2),
                **{f"P{j+1}": p["player"] for j,p in enumerate(l)}
            })

        out = pd.DataFrame(rows).sort_values("EV", ascending=False)
        st.dataframe(out, use_container_width=True)

        st.download_button(
            "Download DraftKings CSV",
            out.to_csv(index=False),
            "dfs_pro_engine_lineups.csv"
        )
