import streamlit as st
import os

st.set_page_config(layout="wide", page_title="Scouting Pro")

st.sidebar.title("⚽ Scouting Pro")
st.sidebar.divider()

fuente = st.sidebar.radio(
    "Fuente de datos",
    options=["📊 Wyscout", "🏟 Jugadores", "📋 Equipos"],
    captions=["Excel Wyscout", "CSV partido a partido", "CSV partido a partido"],
)

st.sidebar.divider()
st.sidebar.caption("⚠️ Los ratings no son comparables entre fuentes.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if fuente == "📊 Wyscout":
    st.title("⚽ Scouting & Role Scoring Engine — Wyscout")
    with open(os.path.join(BASE_DIR, "app_wyscout.py"), encoding="utf-8") as f:
        src = f.read()
    src = src.replace('st.set_page_config(layout="wide", page_title="Scouting Wyscout Pro")', "")
    src = src.replace('st.set_page_config(layout="wide", page_title="Scouting Pro")', "")
    src = src.replace('st.title("⚽ Scouting & Role Scoring Engine v1.2")', "")
    src = src.replace('st.title("⚽ Scouting & Role Scoring Engine v2.0")', "")
    exec(src, {"__name__": "__main__"})

elif fuente == "🏟 Jugadores":
    st.title("🏟 Scouting & Role Scoring Engine")
    with open(os.path.join(BASE_DIR, "app_rfef.py"), encoding="utf-8") as f:
        src = f.read()
    exec(src, {"__name__": "__main__"})

else:
    st.title("📋 Análisis de Equipos")
    with open(os.path.join(BASE_DIR, "app_equipos.py"), encoding="utf-8") as f:
        src = f.read()
    src = src.replace('st.set_page_config', '# st.set_page_config')
    src = src.replace('st.title("🏟️ AppScout — Análisis de Equipos")', "")
    exec(src, {"__name__": "__main__"})
