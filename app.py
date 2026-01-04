import streamlit as st
import pandas as pd
import random

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="DFS PRO ENGINE", layout="wide")

PASSWORD = "dfs123"
pw = st.text_input("Enter Password", type="password")
if pw != PASSWORD:
    st.warning("Enter password to continue")
    st.stop()

# ----------------------------
# SPORT SELECTION
# ----------------------------
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
uploaded = st.file_uploader("Upload DraftKings CSV", type=["csv"])

# ----------------------------
# NORMALIZE CSV
# ----------------------------
def normalize(df):
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={
        "sal":"salary",
        "fpts":"projection",
        "pos":"position"
    })

    required = ["player","position","salary","projection"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    df["salary"] = pd.to_numeric(df["salary"], errors="coerce").fillna(0)
    df["projection"] = pd.to_numeric(df["projection"], errors="coerce").fillna(0)
    return df

def valid(pos, slot):
    if slot == "UTIL":
        return True
    return slot in pos

def build(df):
    lineup = []
    salary = 0
    used = set()
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
            salary += p["salary"]
            used.add(p["player"])
            break

    return lineup if len(lineup) == len(ROSTER) else None

# ----------------------------
# BUILD LINEUPS
# ----------------------------
if uploaded:
    df = normalize(pd.read_csv(uploaded))
    st.success("CSV Loaded")

    n = st.slider("Lineups", 1, 150, 20)

    if st.button("BUILD"):
        lineups = []
        tries = 0

        while len(lineups) < n and tries < n * 20:
            tries += 1
            l = build(df)
            if l:
                lineups.append(l)

        rows = []
        for i, l in enumerate(lineups):
            r = {
                "Lineup": i + 1,
                "Salary": sum(p["salary"] for p in l),
                "Projection": sum(p["projection"] for p in l)
            }
            for p in l:
                r[p["slot"]] = p["player"]
            rows.append(r)

        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)
        st.download_button("Download CSV", out.to_csv(index=False), "lineups.csv")
