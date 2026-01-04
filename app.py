import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="DFS PRO ENGINE", layout="wide")

# =========================
# PASSWORD
# =========================
PASSWORD = "dfs123"
pw = st.text_input("Enter Password", type="password")
if pw != PASSWORD:
    st.warning("Enter password to continue")
    st.stop()

# =========================
# SPORT + SLATE TYPE
# =========================
SPORT = st.selectbox(
    "Sport",
    ["NBA", "NFL", "CBB", "NHL", "MLB", "SOCCER"]
)

SLATE = st.selectbox(
    "Slate Type",
    ["Classic", "Showdown"]
)

st.caption(f"Sport: {SPORT} | Slate: {SLATE}")

# =========================
# ROSTERS
# =========================
def get_roster(sport, slate):
    if slate == "Showdown":
        return ["CPT", "UTIL", "UTIL", "UTIL", "UTIL", "UTIL"]

    return {
        "NBA": ["PG","SG","SF","PF","C","G","F","UTIL"],
        "NFL": ["QB","RB","RB","WR","WR","WR","TE","FLEX","DST"],
        "CBB": ["G","G","G","F","F","UTIL","UTIL","UTIL"],
        "NHL": ["C","C","W","W","W","D","D","G","UTIL"],
        "MLB": ["P","P","C","1B","2B","3B","SS","OF","OF","OF"],
        "SOCCER": ["GK","D","D","M","M","F","F","UTIL"]
    }[sport]

ROSTER = get_roster(SPORT, SLATE)
SALARY_CAP = 50000

# =========================
# UI
# =========================
st.title("🔥 DFS PRO ENGINE (Stable Cloud Build)")
uploaded = st.file_uploader("Upload DraftKings Salaries CSV", type=["csv"])

# =========================
# NORMALIZE CSV
# =========================
def normalize(df):
    df.columns = [c.lower().strip() for c in df.columns]

    df = df.rename(columns={
        "name": "player",
        "salary": "salary",
        "avgpointspergame": "projection",
        "position": "position",
        "teamabbrev": "team",
        "game time": "game_time"
    })

    required = ["player", "position", "salary", "projection"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0)
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce").fillna(0)
    df["ownership"] = 5.0

    if "game_time" in df.columns:
        df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    return df

# =========================
# POSITION VALIDATION
# =========================
def valid(pos, slot):
    pos = str(pos)

    if slot in ["UTIL", "FLEX"]:
        return True

    if slot == "CPT":
        return True

    if slot == "G":
        return any(x in pos for x in ["PG","SG","G"])
    if slot == "F":
        return any(x in pos for x in ["SF","PF","F"])

    return slot in pos

# =========================
# LATE SWAP
# =========================
def get_locked_players(df, current_time):
    locked = set()
    if "game_time" not in df.columns or not current_time:
        return locked

    now = datetime.combine(datetime.today(), current_time)

    for _, r in df.iterrows():
        if pd.notna(r["game_time"]) and r["game_time"] <= now:
            locked.add(r["player"])
    return locked

# =========================
# EXPOSURE
# =========================
def exposure_ok(lineup, exposure, max_pct, total):
    for p in lineup:
        used = exposure.get(p["player"], 0)
        if total > 0 and used / total > max_pct:
            return False
    return True

def update_exposure(lineup, exposure):
    for p in lineup:
        exposure[p["player"]] = exposure.get(p["player"], 0) + 1

# =========================
# LINEUP BUILDER
# =========================
def build_lineup(df, locked_players):
    lineup = []
    used = set(locked_players)
    salary = 0
    pool = df.sample(frac=1).to_dict("records")

    for slot in ROSTER:
        for p in pool:
            if p["player"] in used:
                continue
            if not valid(p["position"], slot):
                continue
            if salary + p["salary"] > SALARY_CAP:
                continue

            lineup.append({**p, "slot": slot})
            used.add(p["player"])
            salary += p["salary"]
            break

    return lineup if len(lineup) == len(ROSTER) else None

# =========================
# CORRELATION
# =========================
def correlation_bonus(lineup):
    teams = {}
    bonus = 0

    for p in lineup:
        teams[p["team"]] = teams.get(p["team"], 0) + 1

    for c in teams.values():
        if c >= 3:
            bonus += c * 2

    return bonus

# =========================
# SIMULATION
# =========================
def simulate_lineup(lineup):
    total = 0
    for p in lineup:
        mu = p["projection"]
        sigma = mu * 0.30
        total += np.random.normal(mu, sigma)
    return total + correlation_bonus(lineup)

def run_simulation(lineups, sims, field):
    results = [{"win":0,"top1":0,"cash":0} for _ in lineups]

    for _ in range(sims):
        scores = [simulate_lineup(l) for l in lineups]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        for i, idx in enumerate(ranked):
            if i == 0:
                results[idx]["win"] += 1
            if i < max(1, field // 100):
                results[idx]["top1"] += 1
            if i < max(1, field // 4):
                results[idx]["cash"] += 1

    for r in results:
        r["win"] /= sims
        r["top1"] /= sims
        r["cash"] /= sims

    return results

FIELD_MAP = {
    "<1k": 500,
    "1k–10k": 5000,
    "10k–100k": 25000,
    "100k+": 100000
}

# =========================
# APP RUN
# =========================
if uploaded:
    df = normalize(pd.read_csv(uploaded))
    st.success("CSV Loaded")

    n = st.slider("Lineups", 1, 150, 20)
    max_exposure = st.slider("Max Player Exposure (%)", 10, 100, 40) / 100

    enable_late = st.checkbox("Enable Late Swap")
    current_time = st.time_input("Current Time") if enable_late else None

    run_sim = st.checkbox("Run Contest Simulation")
    sims = st.slider("Simulations", 200, 3000, 1000, step=200)
    field_size = st.selectbox("Contest Size", list(FIELD_MAP.keys()))

    if st.button("BUILD"):
        exposure = {}
        lineups = []
        tries = 0

        locked_players = get_locked_players(df, current_time)

        while len(lineups) < n and tries < n * 25:
            tries += 1
            l = build_lineup(df, locked_players)
            if not l:
                continue
            if not exposure_ok(l, exposure, max_exposure, n):
                continue

            update_exposure(l, exposure)
            lineups.append(l)

        sim_results = None
        if run_sim:
            sim_results = run_simulation(lineups, sims, FIELD_MAP[field_size])

        rows = []
        for i, l in enumerate(lineups):
            r = {
                "Lineup": i + 1,
                "Salary": sum(p["salary"] for p in l),
                "Projection": round(sum(p["projection"] for p in l), 2),
                "Correlation": correlation_bonus(l)
            }

            if sim_results:
                r["Win %"] = round(sim_results[i]["win"] * 100, 2)
                r["Top 1%"] = round(sim_results[i]["top1"] * 100, 2)
                r["Cash %"] = round(sim_results[i]["cash"] * 100, 2)

            for p in l:
                r[p["slot"]] = p["player"]

            rows.append(r)

        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)
        st.download_button("Download CSV", out.to_csv(index=False), "dfs_lineups.csv")
