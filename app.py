import streamlit as st
import pandas as pd
import random
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="DFS PRO ENGINE", layout="wide")

PASSWORD = "dfs123"
pw = st.text_input("Enter Password", type="password")
if pw != PASSWORD:
    st.warning("Enter password to continue")
    st.stop()

# =========================
# SPORT SELECTION
# =========================
SPORT = st.selectbox("Sport", ["NBA", "NHL", "MLB"])
st.caption(f"Current Sport: {SPORT}")

if SPORT == "NBA":
    SALARY_CAP = 50000
    ROSTER = ["PG","SG","SF","PF","C","G","F","UTIL"]
elif SPORT == "NHL":
    SALARY_CAP = 50000
    ROSTER = ["C","C","W","W","W","D","D","G","UTIL"]
else:
    SALARY_CAP = 50000
    ROSTER = ["P","P","C","1B","2B","3B","SS","OF","OF","OF"]

st.title("🏀 DFS PRO ENGINE (Cloud Ready)")
uploaded = st.file_uploader("Upload DraftKings Salaries CSV", type=["csv"])

# =========================
# NORMALIZE DK CSV
# =========================
def normalize(df):
    df.columns = [c.lower().strip() for c in df.columns]

    df = df.rename(columns={
        "name": "player",
        "salary": "salary",
        "avgpointspergame": "projection",
        "position": "position",
        "teamabbrev": "team"
    })

    required = ["player", "position", "salary", "projection"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0)
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce").fillna(0)

    return df

# =========================
# POSITION VALIDATION
# =========================
def valid(pos, slot):
    pos = str(pos)

    if slot == "UTIL":
        return True

    if SPORT == "NBA":
        if slot == "G":
            return "PG" in pos or "SG" in pos
        if slot == "F":
            return "SF" in pos or "PF" in pos
        return slot in pos

    if SPORT == "NHL":
        return slot in pos

    if SPORT == "MLB":
        if slot == "OF":
            return "OF" in pos
        return slot in pos

    return False

# =========================
# LINEUP BUILDER
# =========================
def build_lineup(df):
    lineup = []
    used = set()
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
# #3 CORRELATION (SAFE)
# =========================
def correlation_bonus(lineup):
    teams = {}
    bonus = 0

    for p in lineup:
        teams[p["team"]] = teams.get(p["team"], 0) + 1

    for count in teams.values():
        if SPORT == "NBA" and count >= 3:
            bonus += count * 2
        if SPORT == "MLB" and count >= 3:
            bonus += count * 4
        if SPORT == "NHL" and count >= 2:
            bonus += count * 3

    return bonus

# =========================
# #4 CONTEST SIMULATION
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
# APP LOGIC
# =========================
if uploaded:
    df = normalize(pd.read_csv(uploaded))
    st.success("DraftKings CSV Loaded")

    n = st.slider("Number of Lineups", 1, 150, 20)

    run_sim = st.checkbox("Run Contest Simulation")
    sims = st.slider("Simulations", 200, 3000, 1000, step=200)
    field_size = st.selectbox("Contest Size", list(FIELD_MAP.keys()))

    if st.button("BUILD"):
        lineups = []
        tries = 0

        while len(lineups) < n and tries < n * 25:
            tries += 1
            l = build_lineup(df)
            if l:
                lineups.append(l)

        rows = []
        sim_results = None

        if run_sim:
            sim_results = run_simulation(lineups, sims, FIELD_MAP[field_size])

        for i, l in enumerate(lineups):
            r = {
                "Lineup": i + 1,
                "Salary": sum(p["salary"] for p in l),
                "Projection": round(sum(p["projection"] for p in l), 2),
                "Correlation Bonus": correlation_bonus(l)
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
        st.download_button("Download Lineups CSV", out.to_csv(index=False), "dfs_lineups.csv")
