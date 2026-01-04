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

# ======================
# SPORT / SLATE
# ======================
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
    df["ownership"] = 5.0
    return df

# ======================
# RULES
# ======================
def valid(pos, slot):
    if slot in ["UTIL","FLEX","CPT"]:
        return True
    return slot in pos

def stack_bonus(lineup):
    teams = {}
    bonus = 0
    for p in lineup:
        teams[p["team"]] = teams.get(p["team"],0)+1
    for t,c in teams.items():
        if SPORT=="MLB" and c>=3:
            bonus += c*4
        if SPORT in ["NBA","CBB","NHL"] and c>=2:
            bonus += c*2
        if SPORT=="NFL" and c>=2:
            bonus += c*3
        if SPORT=="SOCCER" and c>=2:
            bonus += c*3
    return bonus

# ======================
# SIMULATION
# ======================
def simulate(lineup):
    total=0
    for p in lineup:
        mu=p["projection"]
        total+=np.random.normal(mu,mu*0.30)
    return total+stack_bonus(lineup)

PAYOUTS = [
    (0.01,50),
    (0.05,10),
    (0.25,2),
]

def ev_score(lineup,sims=500):
    scores=[simulate(lineup) for _ in range(sims)]
    scores.sort(reverse=True)
    ev=0
    for pct,pay in PAYOUTS:
        idx=int(len(scores)*pct)
        ev+=scores[idx]*pay*0.01
    return ev

# ======================
# OPTIMIZER (PORTFOLIO)
# ======================
def build_lineup(df, exposure, max_exp):
    prob=LpProblem("DFS",LpMaximize)
    x=LpVariable.dicts("p",df.index,0,1,LpBinary)

    prob+=lpSum(
        x[i]*(df.loc[i,"projection"]
        -df.loc[i,"ownership"]*0.4)
        for i in df.index
    )

    prob+=lpSum(x[i]*df.loc[i,"salary"] for i in df.index)<=SALARY_CAP
    prob+=lpSum(x[i] for i in df.index)==len(ROSTER)

    for slot in set(ROSTER):
        prob+=lpSum(x[i] for i in df.index if valid(df.loc[i,"position"],slot))>=ROSTER.count(slot)

    for p,c in exposure.items():
        if c/max_exp>0.6:
            prob+=lpSum(x[i] for i in df.index if df.loc[i,"player"]==p)==0

    prob.solve()
    lineup=df.loc[[i for i in df.index if x[i].value()==1]].to_dict("records")

    if SLATE=="Showdown":
        cpt=max(lineup,key=lambda p:p["projection"])
        cpt["salary"]*=1.5
        cpt["projection"]*=1.5

    return lineup

# ======================
# RUN
# ======================
if uploaded:
    df=normalize(pd.read_csv(uploaded))
    n=st.slider("Lineups",1,50,20)
    sims=st.slider("Simulations",300,2000,800)
    max_exp=st.slider("Max Exposure %",10,100,40)

    if st.button("BUILD"):
        exposure={}
        lineups=[]
        for _ in range(n):
            l=build_lineup(df,exposure,n)
            for p in l:
                exposure[p["player"]]=exposure.get(p["player"],0)+1
            lineups.append(l)

        rows=[]
        for i,l in enumerate(lineups):
            ev=ev_score(l,sims)
            r={
                "Lineup":i+1,
                "Salary":sum(p["salary"] for p in l),
                "Projection":round(sum(p["projection"] for p in l),2),
                "Stack Bonus":stack_bonus(l),
                "EV":round(ev,2)
            }
            for j,p in enumerate(l):
                r[f"P{j+1}"]=p["player"]
            rows.append(r)

        out=pd.DataFrame(rows).sort_values("EV",ascending=False)
        st.dataframe(out,use_container_width=True)
        st.download_button("Download CSV",out.to_csv(index=False),"dfs_portfolio.csv")
