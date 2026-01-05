import streamlit as st
import pandas as pd
import numpy as np
from pulp import *

# ==================================================
# APP CONFIG
# ==================================================
st.set_page_config(
    page_title="DFS PRO ENGINE",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# AUTH
# ==================================================
PASSWORD = "dfs123"
pw = st.sidebar.text_input("🔒 Password", type="password")
if pw != PASSWORD:
    st.stop()

# ==================================================
# GLOBALS
# ==================================================
SALARY_CAP = 50000

# ==================================================
# SIDEBAR – SLATE SETTINGS
# ==================================================
st.sidebar.header("🎯 Slate Setup")

SPORT = st.sidebar.selectbox(
    "Sport",
    ["NBA", "NFL", "CBB", "NHL", "MLB", "SOCCER"]
)

SLATE = st.sidebar.selectbox(
    "Slate Type",
    ["Classic", "Showdown"]
)

NUM_LINEUPS = st.sidebar.slider("Lineups", 1, 50, 20)
MAX_EXPOSURE = st.sidebar.slider("Max Player Exposure %", 10, 100, 40)
SIMS = st.sidebar.slider("Contest Simulations", 300, 3000, 1000, step=100)

# ==================================================
# ROSTERS
# ==================================================
ROSTERS = {
    "NBA": ["PG","SG","SF","PF","C","G","F","UTIL"],
    "NFL": ["QB","RB","RB","WR","WR","WR","TE","FLEX","DST"],
    "CBB": ["G","G","G","F","F","UTIL","UTIL","UTIL"],
    "NHL": ["C","C","W","W","W","D","D","G","UTIL"],
    "MLB": ["P","P","C","1B","2B","3B","SS","OF","OF","OF"],
    "SOCCER": ["GK","D","D","M","M","F","F","UTIL"]
}

ROSTER = ["CPT","UTIL","UTIL","UTIL","UTIL","UTIL"] if SLATE=="Showdown" else ROSTERS[SPORT]

# ==================================================
# MAIN TITLE
# ==================================================
st.title("📊 DFS PRO ENGINE")
st.caption("Stokastic-style DFS Optimizer • DraftKings Legal")

# ==================================================
# UPLOAD
# ==================================================
uploaded = st.file_uploader("📥 Upload DraftKings Salary CSV", type=["csv"])

# ==================================================
# DATA CLEAN
# ==================================================
def normalize(df):
    df.columns = [c.lower().strip() for c in df.columns]

    df = df.rename(columns={
        "name": "player",
        "salary": "salary",
        "avgpointspergame": "projection",
        "position": "position",
        "teamabbrev": "team"
    })

    for c in ["salary","projection"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "ownership" not in df.columns:
        df["ownership"] = 5.0

    return df

# ==================================================
# RULES
# ==================================================
def valid_pos(player_pos, slot):
    if slot in ["UTIL","FLEX"]:
        return True
    if slot == "CPT":
        return True
    return slot in player_pos

def stack_bonus(lineup):
    teams = {}
    bonus = 0

    for p in lineup:
        teams[p["team"]] = teams.get(p["team"], 0) + 1

    for _, count in teams.items():
        if SPORT == "MLB" and count >= 3:
            bonus += count * 4
        elif SPORT in ["NBA","CBB","NHL"] and count >= 2:
            bonus += count * 2
        elif SPORT in ["NFL","SOCCER"] and count >= 2:
            bonus += count * 3

    return bonus

# ==================================================
# SIMULATION / EV
# ==================================================
def simulate_lineup(lineup):
    total = 0
    for p in lineup:
        mu = p["projection"]
        total += np.random.normal(mu, mu * 0.30)
    return total + stack_bonus(lineup)

PAYOUT_CURVE = [
    (0.01, 50),
    (0.05, 10),
    (0.25, 2)
]

def ev_score(lineup, sims):
    scores = sorted(
        [simulate_lineup(lineup) for _ in range(sims)],
        reverse=True
    )

    ev = 0
    for pct, mult in PAYOUT_CURVE:
        idx = int(len(scores) * pct)
        ev += scores[idx] * mult * 0.01

    return ev

# ==================================================
# OPTIMIZER
# ==================================================
def build_lineup(df, exposure, excluded):
    prob = LpProblem("DFS", LpMaximize)
    x = LpVariable.dicts("p", df.index, 0, 1, LpBinary)

    # Objective: projection + leverage
    prob += lpSum(
        x[i] * (df.loc[i,"projection"] - df.loc[i,"ownership"] * 0.4)
        for i in df.index
    )

    # Salary cap (CPT costs 1.5x)
    prob += lpSum(
        x[i] * (df.loc[i,"salary"] * (1.5 if SLATE=="Showdown" else 1))
        for i in df.index
    ) <= SALARY_CAP

    # Roster size
    prob += lpSum(x[i] for i in df.index) == len(ROSTER)

    # Position constraints
    for slot in set(ROSTER):
        prob += lpSum(
            x[i] for i in df.index
            if valid_pos(df.loc[i,"position"], slot)
        ) >= ROSTER.count(slot)

    # Exposure caps
    for player, count in exposure.items():
        if count / NUM_LINEUPS > MAX_EXPOSURE / 100:
            prob += lpSum(
                x[i] for i in df.index if df.loc[i,"player"] == player
            ) == 0

    # Exclusions
    for i in df.index:
        if df.loc[i,"player"] in excluded:
            prob += x[i] == 0

    prob.solve(PULP_CBC_CMD(msg=False))

    lineup = df.loc[
        [i for i in df.index if x[i].value() == 1]
    ].to_dict("records")

    # Apply CPT multiplier AFTER
    if SLATE == "Showdown" and lineup:
        cpt = max(lineup, key=lambda p: p["projection"])
        cpt["salary"] *= 1.5
        cpt["projection"] *= 1.5

    return lineup

# ==================================================
# RUN
# ==================================================
if uploaded:
    df = normalize(pd.read_csv(uploaded))

    excluded = st.multiselect(
        "🚫 Exclude Players (Injury / Fade)",
        sorted(df["player"].unique())
    )

    if st.button("🚀 BUILD PORTFOLIO"):
        exposure = {}
        portfolio = []

        for _ in range(NUM_LINEUPS):
            lineup = build_lineup(df, exposure, excluded)
            for p in lineup:
                exposure[p["player"]] = exposure.get(p["player"], 0) + 1
            portfolio.append(lineup)

        rows = []
        for i, l in enumerate(portfolio):
            rows.append({
                "Lineup": i+1,
                "Salary": round(sum(p["salary"] for p in l),0),
                "Projection": round(sum(p["projection"] for p in l),2),
                "Stack Bonus": stack_bonus(l),
                "EV": round(ev_score(l, SIMS),2),
                **{f"P{j+1}": p["player"] for j,p in enumerate(l)}
            })

        out = pd.DataFrame(rows).sort_values("EV", ascending=False)

        st.subheader("🏆 Optimized Lineups")
        st.dataframe(out, use_container_width=True)

        st.download_button(
            "⬇️ Download DraftKings CSV",
            out.to_csv(index=False),
            "dfs_pro_engine_lineups.csv"
        )
