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

PASSWORD = "dfs123"
pw = st.text_input("Enter Password", type="password")
if pw != PASSWORD:
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
    df = df.rename(columns={
        "sal":"salary",
        "fpts":"projection",
        "pos":"position",
        "game time":"game_time"
    })

    for c in ["salary","projection"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "game_time" in df.columns:
        df["game_time"] = pd.to_datetime(df["game_time"], errors="coerce")

    return df


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
        lineups = []
tries = 0

while len(lineups) < n and tries < 500:
    tries += 1

    locked = []
    if enable_late and current_time:
        locked = get_locked_players(df, current_time)

    l = build(df, locked)
    if l:
        lineups.append(l)

        rows=[]
        for i,l in enumerate(lineups):
            r={"Lineup":i+1,"Salary":sum(p["salary"] for p in l),
               "Projection":sum(p["projection"] for p in l)}
            for p in l: r[p["slot"]] = p["player"]
            rows.append(r)

        out=pd.DataFrame(rows)
        st.dataframe(out,use_container_width=True)
        st.download_button("Download CSV", out.to_csv(index=False), "lineups.csv")
