import streamlit as st

st.set_page_config(page_title="GeoEyeAI", layout="wide")

if "lang" not in st.session_state:
    st.session_state["lang"] = "pt"

idioma = st.sidebar.selectbox("Language / Idioma", ["Português", "English"], 
                             index=0 if st.session_state["lang"] == "pt" else 1)
st.session_state["lang"] = "pt" if idioma == "Português" else "en"
lang = st.session_state["lang"]

titulos = {
    "pt": {"home": "Início", "rel": "Relatório de Ortofoto - Segmentação"},
    "en": {"home": "Home", "rel": "Orthoimage Report - Segmentation"}
}

pg = st.navigation([
    st.Page("pages/home.py", title=titulos[lang]["home"], icon="🏠", default=True),
    st.Page("pages/relatorio_orto.py", title=titulos[lang]["rel"], icon="🏗️")
])
pg.run()