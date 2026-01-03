import streamlit as st
import numpy as np
import pandas as pd

from geo import *

# Home
lang = st.session_state.get("lang", "pt")

# # Cria o dicionário com os textos DESSA página específica
# textos = {
#             "pt": {
#                     "titulo": "Verificação de uma Longarina – Flexão Pura (NBR 7190)",
#                     "entrada_tipo_secao": "Tipo de seção",
#                     "tipo_secao": ["Retangular", "Circular"],
#                     "base": "Base (cm)",
#                     "altura": "Altura (cm)",
#                     "diametro": "Diâmetro (cm)",
#                     "entrada_comprimento": "Comprimento da viga (m)",
#                     "carga_permanente": "Carga permanente (kN/m)",
#                     "trem_tipo": "Tipo de trem",
#                     "tipo_tb": ["TB-240", "TB-450"],
#                     "classe_carregamento": "Classe de carregamento",
#                     "classe_carregamento_opcoes": ["Permanente", "Longa duração", "Média duração", "Curta duração", "Instantânea"],
#                     "classe_madeira": "Classe de madeira", 
#                     "classe_madeira_opcoes": ["Madeira natural","Madeira recomposta"],
#                     "classe_umidade": "Classe de umidade",
#                     "gamma_g": "γg – Coeficiente parcial de segurança da carga permanente",
#                     "gamma_q": "γq – Coeficiente parcial de segurança da carga variável",
#                     "gamma_w": "γw – Coeficiente parcial de segurança do material",
#                     "f_ck": "Resistência característica à compressão paralela às fibras (MPa)",
#                     "f_tk": "Resistência característica à tração paralela às fibras (MPa)",
#                     "e_modflex": "Módulo de elasticidade à flexão (GPa)",
#                     "botao": "Verificar viga",
#                     "resultado": "A resistência calculada é: "
#                   },
#             "en": {
#                     "titulo": "Verification of a Stringer – Pure Bending (NBR 7190)",
#                     "entrada_tipo_secao": "Section type",
#                     "tipo_secao": ["Rectangular", "Circular"],
#                     "base": "Width (cm)",
#                     "altura": "Height (cm)",
#                     "diametro": "Diameter (cm)",
#                     "entrada_comprimento": "Beam length (m)",
#                     "carga_permanente": "Dead load (kN/m)",
#                     "trem_tipo": "Load train type",
#                     "tipo_tb": ["TB-240", "TB-450"],
#                     "classe_carregamento": "Load duration class",
#                     "classe_carregamento_opcoes": ["Dead", "Long-term", "Medium-term", "Short-term", "Instantaneous"],
#                     "classe_madeira": "Wood class",
#                     "classe_madeira_opcoes": ["Natural wood", "Engineered wood"],
#                     "classe_umidade": "Moisture class",
#                     "gamma_g": "γg – Partial safety factor for dead load",
#                     "gamma_q": "γq – Partial safety factor for variable load",
#                     "gamma_w": "γw – Partial safety factor for material",
#                     "f_ck": "Characteristic compressive strength parallel to grain (MPa)",
#                     "f_tk": "Characteristic tensile strength parallel to grain (MPa)",
#                     "e_modflex": "Modulus of elasticity in bending (GPa)",
#                     "botao": "Check beam",
#                     "resultado": "The calculated resistance is: "
#                   }
#          }   
# t = textos[lang]

# # Calculadora da página
# st.header(t["titulo"])
# tipo_secao = st.selectbox(t["entrada_tipo_secao"], t["tipo_secao"])
# if tipo_secao in ["Retangular", "Rectangular"]:
#     b_cm = st.number_input(t["base"], min_value=0.05, value=20.)
#     h_cm = st.number_input(t["altura"], min_value=0.05, value=50.)
#     b = b_cm / 100
#     h = h_cm / 100
#     geo = {"b_w": b, "h": h}
# else:
#     d_cm = st.number_input(t["diametro"], min_value=0.05, value=30.)
#     d = d_cm / 100
#     geo = {"d": d}
# carga_pp = 8.0
# l = st.number_input(t["entrada_comprimento"], min_value=3.0, value=6.0)
# col1, col2 = st.columns(2)
# with col1:
#     p_gk = st.number_input(t["carga_permanente"], value=carga_pp)    
# with col2:
#     p_qk = st.selectbox(t["trem_tipo"], t["tipo_tb"])
# classe_carregamento = st.selectbox(t["classe_carregamento"], t["classe_carregamento_opcoes"]).lower()
# if classe_carregamento == "permanent":
#     classe_carregamento = "Permanente"
# elif classe_carregamento == "long-term":
#     classe_carregamento = "Longa duração"
# elif classe_carregamento == "medium-term":
#     classe_carregamento = "Média duração"
# elif classe_carregamento == "short-term":
#     classe_carregamento = "Curta duração"
# elif classe_carregamento == "instantaneous":
#     classe_carregamento = "Instantânea"
# classe_madeira = st.selectbox(t["classe_madeira"], t["classe_madeira_opcoes"]).lower()
# if classe_madeira == "natural wood":
#     classe_madeira = "madeira natural"
# else: 
#     classe_madeira = "madeira recomposta"
# classe_umidade = st.selectbox(t["classe_umidade"], [1, 2, 3, 4])
# gamma_g = st.number_input(t["gamma_g"], value=1.40, step=0.1)
# gamma_q = st.number_input(t["gamma_q"], value=1.40, step=0.1)
# gamma_w = st.number_input(t["gamma_w"], value=1.40, step=0.1)
# f_ck = st.number_input(t["f_ck"], value=20.0)
# f_tk = st.number_input(t["f_tk"], value=15.0)
# e_modflex = st.number_input(t["e_modflex"], value=12.0)
# f_ck *=  1000  # Converte MPa para kN/m²
# f_tk *= 1000  # Converte MPa para kN/m²
# e_modflex *= 1E9  # Converte GPa para kN/m²

# if st.button(t["botao"]):
    
#     # Análise final
#     res_m, res_delta = checagem_longarina_madeira_flexao(geo, p_gk, p_qk, l, classe_carregamento, classe_madeira, classe_umidade, gamma_g, gamma_q, gamma_w, f_ck, f_tk, e_modflex)
    
#     # Tabela de Flexão
#     st.subheader("Flexão (ELU)")
#     # Converte o dicionário para tabela (orient='index' deixa as chaves nas linhas)
#     df_m = pd.DataFrame.from_dict(res_m, orient='index', columns=['Valor'])
#     st.table(df_m)

#     # Tabela de Flecha
#     st.subheader("Flecha (ELS)")
#     df_delta = pd.DataFrame.from_dict(res_delta, orient='index', columns=['Valor'])
#     st.table(df_delta)
    
#     # Mensagem final resumida
#     if res_m['analise'] == 'OK' and res_delta['analise'] == 'OK':
#         st.success("Resultado Final: APROVADO")
#     else:
#         st.error("Resultado Final: REPROVADO")
