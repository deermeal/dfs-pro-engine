import streamlit as st
import pandas as pd
import numpy as np
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary

# =====================
# CONFIG
# =====================
st.set_page_config(page_title="DFS PRO ENGINE", layout="wide")

PASSWORD = "dfs123"
pw = st.text_input("Enter Password", type="password")
if pw != PASSWORD:
    st.stop()

# =====================
# SPORT / SLATE
# =====================
SPORT = st.selectbox("Sport", ["NBA","NFL","CBB","NHL","MLB","SOCCER"])
SLATE = st.selectbox("Slate Type", ["Classic","Showdown"])

SALARY_CAP = 50000

ROSTERS = {
    "NBA": ["PG","SG","SF","PF","C","G","F","UTIL"],
    "NFL": ["QB","RB","RB","WR","WR","WR","TE","FLEX","DST"],
    "CBB": ["G","G","G","F","F","UTIL","UTIL","UTIL"],
    "NHL": ["C","C","W","W","W","D","D","G","UTIL"],
    "MLB": ["P","P","C","1B","2B","3B","SS","OF","OF","OF"],
    "SOCCER": ["GK","D","D","M","M","F","F","UTIL"]
}

ROSTER = ["CPT","UTIL","UTIL","UTIL","UTIL","UTIL"] if SLATE=="Showdown" else ROSTERS[SPORT]

st.title("🔥 DFS PRO ENGINE — OPTIMIZED")

# =====================
# UPLOAD
# =====================
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

    df["ownership"] = 5.0
    return df

def valid(pos, slot):
    if slot in ["UTIL","FLEX","CPT"]:
        return True
    return slot in pos

# =====================
# CORRELATION
# =====================
def correlation_bonus(lineup):
    teams = {}
    bonus = 0
    for p in lineup:
        teams[p["team"]] = teams.get(p["team"],0)+1
    for c in teams.values():
        if c >= 3:
            bonus += c * 2
    return bonus

# =====================
# SIMULATION
# =====================
def simulate_lineup(lineup):
    total = 0
    for p in lineup:
        mu = p["projection"]
        total += np.random.normal(mu, mu*0.30)
    return total + correlation_bonus(lineup)

def run_sim(lineups, sims, field):
    results = [{"win":0,"cash":0} for _ in lineups]
    for _ in range(sims):
        scores = [simulate_lineup(l) for l in lineups]
        rank = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for i,idx in enumerate(rank):
            if i==0:
                results[idx]["win"]+=1
            if i < max(1,field//4):
                results[idx]["cash"]+=1
    for r in results:
        r["win"]/=sims
        r["cash"]/=sims
    return results

FIELD_MAP = {"<1k":500,"1k–10k":5000,"10k–100k":25000,"100k+":100000}

# =====================
# OPTIMIZER
# =====================
def optimize(df):
    prob = LpProblem("DFS", LpMaximize)

    players = list(df.index)
    x = LpVariable.dicts("p", players, 0, 1, LpBinary)

    prob += lpSum(
        x[i] * (df.loc[i,"projection"]
        - df.loc[i,"ownership"]*0.4)
        for i in players
    )

    prob += lpSum(x[i]*df.loc[i,"salary"] for i in players) <= SALARY_CAP
    prob += lpSum(x[i] for i in players) == len(ROSTER)

    for slot in set(ROSTER):
        prob += lpSum(x[i] for i in players if valid(df.loc[i,"position"],slot)) >= ROSTER.count(slot)

    prob.solve()
    return df.loc[[i for i in players if x[i].value()==1]]

# =====================
# RUN
# =====================
if uploaded:
    df = normalize(pd.read_csv(uploaded))
    n = st.slider("Lineups",1,50,10)
    sims = st.slider("Simulations",200,3000,1000,200)
    contest = st.selectbox("Contest Size", FIELD_MAP.keys())

    if st.button("BUILD"):
        lineups = []
        for _ in range(n):
            lineup_df = optimize(df.sample(frac=1))
            lineup = lineup_df.to_dict("records")
            lineups.append(lineup)

        sim = run_sim(lineups, sims, FIELD_MAP[contest])

        rows=[]
        for i,l in enumerate(lineups):
            r={
                "Lineup":i+1,
                "Salary":sum(p["salary"] for p in l),
                "Projection":round(sum(p["projection"] for p in l),2),
                "Correlation":correlation_bonus(l),
                "Win %":round(sim[i]["win"]*100,2),
                "Cash %":round(sim[i]["cash"]*100,2),
            }
            for idx,p in enumerate(l):
                r[f"P{idx+1}"]=p["player"]
            rows.append(r)

        out=pd.DataFrame(rows).sort_values("Win %",ascending=False)
        st.dataframe(out,use_container_width=True)
        st.download_button("Download CSV", out.to_csv(index=False), "dfs_lineups.csv")
