import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="DFS PRO ENGINE", layout="wide")

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
        "pos":"position"
    })
    for c in ["salary","projection"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def valid(pos, slot):
    if slot=="UTIL": return True
    if slot=="G": return "PG" in pos or "SG" in pos
    if slot=="F": return "SF" in pos or "PF" in pos
    return slot in pos

def build(df):
    lineup, sal, used = [], 0, set()
    pool = df.sample(frac=1).to_dict("records")
    for slot in ROSTER:
        for p in pool:
            if p["player"] in used: continue
            if not valid(p["position"], slot): continue
            if sal + p["salary"] > SALARY_CAP: continue
            lineup.append({**p,"slot":slot})
            sal += p["salary"]
            used.add(p["player"])
            break
    return lineup if len(lineup)==8 else None

if uploaded:
    df = normalize(pd.read_csv(uploaded))
    st.success("CSV Loaded")

    n = st.slider("Lineups",1,150,20)
    if st.button("BUILD"):
        lineups = []
        tries = 0
        while len(lineups)<n and tries<500:
            l = build(df)
            tries+=1
            if l: lineups.append(l)

        rows=[]
        for i,l in enumerate(lineups):
            r={"Lineup":i+1,"Salary":sum(p["salary"] for p in l),
               "Projection":sum(p["projection"] for p in l)}
            for p in l: r[p["slot"]] = p["player"]
            rows.append(r)

        out=pd.DataFrame(rows)
        st.dataframe(out,use_container_width=True)
        st.download_button("Download CSV", out.to_csv(index=False), "lineups.csv")
