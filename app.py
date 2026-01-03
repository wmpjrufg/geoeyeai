import streamlit as st

# Configuração da página (Sempre a primeira linha)
st.set_page_config(page_title="GeoEyeAI", layout="wide")

# Inicializa o estado do idioma se não existir
if "lang" not in st.session_state:
    st.session_state["lang"] = "pt"

# Cria o seletor na barra lateral (Aparecerá em todas as páginas)
idioma_selecionado = st.sidebar.selectbox("Language / Idioma", ["Português", "English"], index=0 if st.session_state["lang"] == "pt" else 1)
if idioma_selecionado == "Português":
    st.session_state["lang"] = "pt"
else:
    st.session_state["lang"] = "en"
lang = st.session_state["lang"]

# Dicionário com os Títulos do Menu
titulos_menu = {
                    "pt": {
                            "home": "Início",
                            "relatorio_orto": "Relatório de inspeção Ortofoto"
                          },
                    "en": {
                            "home": "Home",
                            "relatorio_orto": "Orthoimage Inspection Report"
                          }
                }

# Definição das páginas usando os títulos dinâmicos
home_page = st.Page("pages/home.py", title=titulos_menu[lang]["home"], icon="🏠", default=True)
relatorio_orto_page = st.Page("pages/relatorio_orto.py", title=titulos_menu[lang]["relatorio_orto"], icon="🏗️")

# Executa a navegação
pg = st.navigation([home_page, relatorio_orto_page])
pg.run()
