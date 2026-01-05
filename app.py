import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="DFS PRO ENGINE", layout="wide")

PASSWORD = "dfs123"
pw = st.text_input("Password", type="password")
if pw != PASSWORD:
    st.stop()

SALARY_CAP = 50000

# =========================
# SPORT / SLATE
# =========================
SPORT = st.selectbox("Sport", ["NBA", "NFL", "MLB", "NHL"])
SLATE = st.selectbox("Slate Type", ["Classic", "Showdown"])

ROSTERS = {
    "NBA": ["PG","SG","SF","PF","C","G","F","UTIL"],
    "NFL": ["QB","RB","RB","WR","WR","WR","TE","FLEX","DST"],
    "MLB": ["P","P","C","1B","2B","3B","SS","OF","OF","OF"],
    "NHL": ["C","C","W","W","W","D","D","G","UTIL"]
}

ROSTER = ["CPT","UTIL","UTIL","UTIL","UTIL","UTIL"] if SLATE == "Showdown" else ROSTERS[SPORT]

# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader("Upload DraftKings CSV", type=["csv"])

def normalize(df):
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={
        "name":"player",
        "position":"position",
        "salary":"salary",
        "avgpointspergame":"projection",
        "teamabbrev":"team"
    })

    for c in ["salary","projection"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "ownership" not in df.columns:
        df["ownership"] = 5.0

    return df

# =========================
# RULES
# =========================
def valid(pos, slot):
    if slot in ["UTIL","FLEX","CPT"]:
        return True
    return slot in pos

def stack_bonus(lineup):
    teams = {}
    bonus = 0
    for p in lineup:
        teams[p["team"]] = teams.get(p["team"], 0) + 1

    for cnt in teams.values():
        if SPORT == "MLB" and cnt >= 3:
            bonus += cnt * 4
        if SPORT in ["NBA","NHL"] and cnt >= 2:
            bonus += cnt * 2
        if SPORT == "NFL" and cnt >= 2:
            bonus += cnt * 3

    return bonus

# =========================
# SIMULATION / EV
# =========================
def simulate_lineup(lineup):
    total = 0
    for p in lineup:
        mu = p["projection"]
        total += np.random.normal(mu, mu * 0.30)
    return total + stack_bonus(lineup)

def ev_score(lineup, sims=500):
    scores = [simulate_lineup(lineup) for _ in range(sims)]
    scores.sort(reverse=True)
    return np.mean(scores[:max(1, int(len(scores)*0.1))])

# =========================
# OPTIMIZER (LP)
# =========================
def build_lineup(df, excluded, exposure, max_exp):
    prob = LpProblem("DFS", LpMaximize)
    x = LpVariable.dicts("p", df.index, 0, 1, LpBinary)

    prob += lpSum(
        x[i] * (
            df.loc[i,"projection"]
            - df.loc[i,"ownership"] * 0.4
        )
        for i in df.index
    )

    prob += lpSum(x[i] * df.loc[i,"salary"] for i in df.index) <= SALARY_CAP
    prob += lpSum(x[i] for i in df.index) == len(ROSTER)

    for slot in set(ROSTER):
        prob += lpSum(
            x[i] for i in df.index if valid(df.loc[i,"position"], slot)
        ) >= ROSTER.count(slot)

    for i in df.index:
        if df.loc[i,"player"] in excluded:
            prob += x[i] == 0

    for p, cnt in exposure.items():
        if cnt / max_exp > 0.6:
            prob += lpSum(
                x[i] for i in df.index if df.loc[i,"player"] == p
            ) == 0

    prob.solve()

    lineup = df.loc[[i for i in df.index if x[i].value() == 1]].to_dict("records")

    if SLATE == "Showdown" and lineup:
        cpt = max(lineup, key=lambda p: p["projection"])
        cpt["salary"] *= 1.5
        cpt["projection"] *= 1.5

    if sum(p["salary"] for p in lineup) > SALARY_CAP:
        return None

    return lineup

# =========================
# UI CONTROLS
# =========================
if uploaded:
    df = normalize(pd.read_csv(uploaded))
    st.success("CSV Loaded")

    excluded = st.multiselect(
        "Exclude Players (injury / out)",
        sorted(df["player"].unique())
    )

    n = st.slider("Lineups", 1, 50, 20)
    sims = st.slider("Simulations", 300, 2000, 800)
    max_exp = st.slider("Max Exposure %", 10, 100, 40)

    if st.button("BUILD PORTFOLIO"):
        exposure = {}
        results = []

        for _ in range(n):
            lineup = build_lineup(df, excluded, exposure, n)
            if not lineup:
                continue

            for p in lineup:
                exposure[p["player"]] = exposure.get(p["player"], 0) + 1

            ev = ev_score(lineup, sims)

            row = {
                "Salary": int(sum(p["salary"] for p in lineup)),
                "Projection": round(sum(p["projection"] for p in lineup), 2),
                "EV": round(ev, 2)
            }

            for i, p in enumerate(lineup):
                row[f"P{i+1}"] = p["player"]

            results.append(row)

        out = pd.DataFrame(results).sort_values("EV", ascending=False)
        st.dataframe(out, use_container_width=True)
        st.download_button(
            "Download DraftKings CSV",
            out.to_csv(index=False),
            "dfs_pro_engine_lineups.csv"
        )
