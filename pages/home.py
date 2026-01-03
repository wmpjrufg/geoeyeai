# Exemplo dentro do pages/home.py
import streamlit as st

if st.session_state["lang"] == "pt":
    st.title("Bem-vindo")
    st.write("Esta é a página inicial do GeoEyeAI.")
else:
    st.title("Welcome")
    st.write("This is the home page of GeoEyeAI.")