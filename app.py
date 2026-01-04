import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="DFS PRO ENGINE", layout="wide")
st.caption(f"Current Sport: {SPORT}")

st.markdown("""
<style>
button {font-size:18px !important; padding:12px !important;}
</style>
""", unsafe_allow_html=True)

PASSWORD = "dfs123"  # change later

pw = st.text_input("Enter Password", type="password")
if pw != PASSWORD:
    st.warning("Enter password to continue")
    st.stop()


SPORT = st.selectbox("Sport", ["NBA", "NHL", "MLB"])

if SPORT == "NBA":
    SALARY_CAP = 50000
    ROSTER = ["PG","SG","SF","PF","C","G","F","UTIL"]

elif SPORT == "NHL":
    SALARY_CAP = 50000
    ROSTER = ["C","C","W","W","W","D","D","G","UTIL"]

elif SPORT == "MLB":
    SALARY_CAP = 50000
    ROSTER = ["P","P","C","1B","2B","3B","SS","OF","OF","OF"]


st.title("🏀 DFS PRO ENGINE (Cloud Ready)")

uploaded = st.file_uploader("Upload DraftKings CSV", type=["csv"])

def normalize(df):
    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]

    rename_map = {
        "sal": "salary",
        "salary($)": "salary",
        "dk salary": "salary",

        "fpts": "projection",
        "proj": "projection",
        "fantasy points": "projection",

        "pos": "position",
        "positions": "position",

        "batting_order": "order",
        "game time": "game_time",

        "teamabbr": "team",
        "oppabbr": "opp",
    }

    df = df.rename(columns=rename_map)

    REQUIRED = ["player", "position", "salary", "projection"]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0)
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce").fillna(0)

    if "ownership" in df.columns:
        df["ownership"] = pd.to_numeric(df["ownership"], errors="coerce").fillna(5)
    else:
        df["ownership"] = 5.0

    if "game_time" in df.columns:
        df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    return df



def valid(pos, slot):
    pos = str(pos)

    if slot == "UTIL":
        return pos not in ["P", "G"]

    if SPORT == "MLB":
        if slot == "OF":
            return "OF" in pos
        return slot in pos

    if SPORT == "NBA":
        if slot == "G":
            return "PG" in pos or "SG" in pos
        if slot == "F":
            return "SF" in pos or "PF" in pos
        return slot in pos

    if SPORT == "NHL":
        return slot in pos


from datetime import datetime

def get_locked_players(df, current_time):
    locked = []
    now = datetime.combine(datetime.today(), current_time)

    for _, row in df.iterrows():
        if "game_time" in df.columns and pd.notna(row["game_time"]):
            if row["game_time"] <= now:
                locked.append({
                    "player": row["player"],
                    "position": row["position"],
                    "salary": row["salary"],
                    "projection": row["projection"],
                    "slot": None
                })
    return locked
def exposure_ok(lineup, exposure, max_pct, total):
    for p in lineup:
        used = exposure.get(p["player"], 0)
        if total > 0 and used / total > max_pct:
            return False
    return True

def update_exposure(lineup, exposure):
    for p in lineup:
        exposure[p["player"]] = exposure.get(p["player"], 0) + 1


def build(df, locked_players=[]):
    lineup = []
    salary = sum(p["salary"] for p in locked_players)
    used = {p["player"] for p in locked_players}

    lineup.extend(locked_players)

    pool = df.sample(frac=1).to_dict("records")

    for slot in ROSTER:
        if any(p.get("slot") == slot for p in lineup):
            continue

        for p in pool:
            if p["player"] in used:
                continue
            if not valid(p["position"], slot):
                continue
            if salary + p["salary"] > SALARY_CAP:
                continue

            lineup.append({**p, "slot": slot})
            salary += p["salary"]
            used.add(p["player"])
            break

    return lineup if len(lineup) == len(ROSTER) else None

if SPORT == "NHL":
    if not nhl_goalie_ok(l):
        continue

if uploaded:
    df = normalize(pd.read_csv(uploaded))
    st.success("CSV Loaded")

    n = st.slider("Lineups",1,150,20)
    st.markdown("### ⏱️ Late Swap (NBA Only)")
enable_late = st.checkbox("Enable Late Swap")
st.markdown("### 🎲 Contest Simulation")
run_sim = st.checkbox("Run Contest Simulation")
sims = st.slider("Simulations", 100, 3000, 1000, step=100)
field_size = st.selectbox("Contest Size", ["<1k", "1k–10k", "10k–100k", "100k+"])


current_time = None
if enable_late:
    current_time = st.time_input("Current Time (ET)")

    if st.button("BUILD"):
        exposure = {}
all_lineups = []
tries = 0

while len(all_lineups) < n and tries < n * 15:
    tries += 1

    locked = []
    if enable_late and current_time:
        locked = get_locked_players(df, current_time)

    l = build(df, locked)
    if not l:
        continue

    if not exposure_ok(l, exposure, max_exposure / 100, n):
        continue

    update_exposure(l, exposure)
    all_lineups.append(l)
scored = []

for i, l in enumerate(all_lineups):
    sim_data = sim_results[i] if run_sim else None
    score = portfolio_score(l, sim_data)
    scored.append((score, l))

scored.sort(reverse=True)
final_lineups = [l for _, l in scored[:final_keep]]

        rows=[]
        for i,l in enumerate(lineups):
            r={"Lineup":i+1,"Salary":sum(p["salary"] for p in l),
               "Projection":sum(p["projection"] for p in l)}
            for p in l: r[p["slot"]] = p["player"]
            rows.append(r)

        out=pd.DataFrame(rows)
        st.dataframe(out,use_container_width=True)
        st.download_button("Download CSV", out.to_csv(index=False), "lineups.csv")
FIELD_MAP = {
    "<1k": 500,
    "1k–10k": 5000,
    "10k–100k": 25000,
    "100k+": 100000
}
import numpy as np

def simulate_player_outcome(p):
    # NBA variance ~ 30%
    return np.random.normal(p["projection"], p["projection"] * 0.30)

def run_contest_sim(lineups, sims, field):
    results = [{"win":0, "top1":0, "cash":0} for _ in lineups]

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
 if run_sim:
    field = FIELD_MAP[field_size]
    sim_results = run_contest_sim(lineups, sims, field)
st.markdown("### 🎛️ Portfolio Controls")
max_exposure = st.slider("Max Player Exposure (%)", 10, 100, 40)
final_keep = st.slider("Final Lineups to Keep", 1, 150, 20)

    for i, r in enumerate(sim_results):
        rows[i]["Win %"] = round(r["win"] * 100, 2)
        rows[i]["Top 1%"] = round(r["top1"] * 100, 2)
        rows[i]["Cash %"] = round(r["cash"] * 100, 2)

def portfolio_score(lineup, sim=None):
    proj = sum(p["projection"] for p in lineup)
    own = sum(p.get("ownership", 6) for p in lineup)

    leverage = max(0, 120 - own) * 0.06
    corr = 0

    if SPORT == "MLB":
        corr = mlb_stack_bonus(lineup)
        proj += sum(order_bonus(p) for p in lineup)

    if SPORT == "NHL":
        corr = nhl_stack_bonus(lineup)

    sim_bonus = 0
    if sim:
        sim_bonus = sim.get("top1", 0) * 50 + sim.get("win", 0) * 100

    return proj + leverage + corr + sim_bonus

st.markdown("### 📊 Exposure Summary")

exp_rows = []
for player, count in exposure.items():
    exp_rows.append({
        "Player": player,
        "Exposure %": round(count / len(all_lineups) * 100, 1)
    })

st.dataframe(pd.DataFrame(exp_rows).sort_values("Exposure %", ascending=False))

def nhl_goalie_ok(lineup):
    goalies = [p for p in lineup if "G" in str(p["position"])]
    if len(goalies) != 1:
        return False

    g = goalies[0]
    for p in lineup:
        if p["team"] == g.get("opp"):
            return False
    return True

def nhl_stack_bonus(lineup):
    stacks = {}
    for p in lineup:
        key = (p.get("team"), p.get("line"))
        stacks[key] = stacks.get(key, 0) + 1

    bonus = 0
    for (_, _), count in stacks.items():
        if count >= 3:
            bonus += count * 4
    return bonus

def mlb_pitcher_ok(lineup):
    pitchers = [p for p in lineup if "P" in str(p["position"])]

    if len(pitchers) != 2:
        return False

    for h in lineup:
        if "P" not in h["position"]:
            for p in pitchers:
                if h.get("opp") == p.get("team"):
                    return False
    return True

def mlb_stack_bonus(lineup):
    teams = {}
    for p in lineup:
        if "P" not in p["position"]:
            teams[p["team"]] = teams.get(p["team"], 0) + 1

    bonus = 0
    for count in teams.values():
        if count >= 3:
            bonus += count * 5
    return bonus
if SPORT == "MLB":
    if not mlb_pitcher_ok(l):
        continue

def order_bonus(p):
    try:
        return max(0, 9 - int(p.get("order", 9))) * 0.8
    except:
        return 0
