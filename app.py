import streamlit as st
import pandas as pd
import numpy as np
from pulp import *
import random
from datetime import datetime

# ======================
# APP CONFIG
# ======================
st.set_page_config("DFS PRO ENGINE", layout="wide")

# ---------- USER TIERS ----------
TIER = st.sidebar.selectbox(
    "User Tier",
    ["FREE","PRO","ELITE"],
    help="Stripe-ready tier gate"
)

LINEUP_CAP = {
    "FREE": 20,
    "PRO": 100,
    "ELITE": 300
}[TIER]

# ======================
# SLATE CONFIG
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
# FILE UPLOADS
# ======================
players_file = st.file_uploader("Upload DK Players CSV", type="csv")
vegas_file = st.file_uploader(
    "Upload Vegas Totals CSV (optional)",
    type="csv",
    help="team, implied_total"
)

# ======================
# SIDEBAR RULES
# ======================
st.sidebar.header("Stacking & Rules")

STACK_TEMPLATE = st.sidebar.selectbox(
    "Team Stack Template",
    ["NONE","3-2","4-1","5-2"]
)

enable_bringback = st.sidebar.checkbox("Require Bring-Back")

exclude_players = st.sidebar.text_area("Exclude Players")
force_group = st.sidebar.text_area("Force Group")
avoid_group = st.sidebar.text_area("Avoid Group")

late_swap_time = st.sidebar.time_input(
    "Late Swap Lock Time",
    datetime.now().time()
)

lineups_n = st.sidebar.slider(
    "Lineups",
    1,
    LINEUP_CAP,
    min(50, LINEUP_CAP)
)

sims = st.sidebar.slider("Contest Sims", 300, 3000, 800)
max_overlap = st.sidebar.slider("Max Overlap", 2, 8, 6)

# ======================
# NORMALIZE DATA
# ======================
def normalize(df):
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={
        "name":"player",
        "salary":"salary",
        "avgpointspergame":"projection",
        "teamabbrev":"team",
        "position":"position",
        "game info":"game"
    })
    for c in ["salary","projection"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["ownership"] = 5.0
    return df

# ======================
# VEGAS TOTALS
# ======================
def apply_vegas(df, vegas):
    vegas = dict(zip(vegas["team"], vegas["implied_total"]))
    df["vegas"] = df["team"].map(vegas).fillna(0)
    df["projection"] *= (1 + df["vegas"] / 100)
    return df

# ======================
# POSITION CHECK
# ======================
def valid(pos, slot):
    if slot in ["UTIL","FLEX","CPT"]:
        return True
    return slot in pos

# ======================
# STACK CONSTRAINT
# ======================
def enforce_stack(prob, x, df):
    if STACK_TEMPLATE == "NONE":
        return
    team_counts = {
        t: lpSum(x[i] for i in df.index if df.loc[i,"team"]==t)
        for t in df["team"].unique()
    }

    if STACK_TEMPLATE == "3-2":
        prob += lpSum(v >= 3 for v in team_counts.values()) >= 1
    if STACK_TEMPLATE == "4-1":
        prob += lpSum(v >= 4 for v in team_counts.values()) >= 1
    if STACK_TEMPLATE == "5-2":
        prob += lpSum(v >= 5 for v in team_counts.values()) >= 1

# ======================
# OPTIMIZER
# ======================
def build_lineup(df, used):
    prob = LpProblem("DFS", LpMaximize)
    x = LpVariable.dicts("p", df.index, 0, 1, LpBinary)

    prob += lpSum(
        x[i] * (
            df.loc[i,"projection"] * random.uniform(0.9,1.1)
            - df.loc[i,"ownership"] * 0.4
        )
        for i in df.index
    )

    prob += lpSum(x[i]*df.loc[i,"salary"] for i in df.index) <= SALARY_CAP
    prob += lpSum(x[i] for i in df.index) == len(ROSTER)

    for slot in set(ROSTER):
        prob += lpSum(
            x[i] for i in df.index if valid(df.loc[i,"position"],slot)
        ) >= ROSTER.count(slot)

    enforce_stack(prob, x, df)

    for prev in used:
        prob += lpSum(x[i] for i in prev) <= max_overlap

    prob.solve(PULP_CBC_CMD(msg=False))
    return [i for i in df.index if x[i].value()==1]

# ======================
# RUN
# ======================
if players_file:
    df = normalize(pd.read_csv(players_file))

    if vegas_file:
        vegas = pd.read_csv(vegas_file)
        df = apply_vegas(df, vegas)

    if st.button("BUILD PORTFOLIO"):
        used = []
        rows = []

        for _ in range(lineups_n):
            idxs = build_lineup(df, used)
            if not idxs:
                break
            used.append(idxs)
            l = df.loc[idxs].to_dict("records")

            rows.append({
                "Salary": sum(p["salary"] for p in l),
                "Projection": round(sum(p["projection"] for p in l),2),
                **{f"P{i+1}":p["player"] for i,p in enumerate(l)}
            })

        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)

        st.download_button(
            "Download DK CSV",
            out[[c for c in out.columns if c.startswith("P")]].to_csv(index=False),
            "dk_upload.csv"
        )
