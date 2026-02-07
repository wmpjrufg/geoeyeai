import streamlit as st
from PIL import Image
import os

# Garante que a variável de idioma exista
if "lang" not in st.session_state:
    st.session_state["lang"] = "pt"

lang = st.session_state["lang"]

# =========================================================
# DICIONÁRIO DE TEXTOS (PT/EN)
# =========================================================
textos = {
    "pt": {
        "univ": "Universidade Federal de Catalão (UFCAT)",
        "grupo": "Grupo de Pesquisa de Estudos em Engenharia",
        "titulo_principal": "Bem-vindo à Plataforma GeoEyeAI",
        "subtitulo": "Monitoramento Geotécnico Inteligente via Visão Computacional",
        "intro_titulo": "O que é esta ferramenta?",
        "intro_texto": """
        Esta ferramenta consiste em um software avançado de aprendizado de máquina (Deep Learning) projetado para auxiliar engenheiros geotécnicos. 
        Ao fazer o upload de um **ortomosaico** de alta resolução (arquivos .tiff de até 7GB), o sistema utiliza redes neurais de **Segmentação Semântica** para identificar e mapear automaticamente potenciais anomalias na estrutura.
        """,
        "funcionalidades_titulo": "Principais Funcionalidades:",
        "func_1": "**Análise Automatizada:** Detecção de patologias como erosões, trincas, rupturas e presença de água.",
        "func_2": "**Processamento Otimizado:** O sistema fatia imagens gigantescas em blocos menores para análise detalhada sem perder resolução.",
        "func_3": "**Validação Humana:** Interface interativa para que o engenheiro confirme, descarte ou edite as predições da IA.",
        "func_4": "**Relatórios Oficiais:** Geração automática de relatórios em PDF com mapas de anomalias e estatísticas.",
        "requisitos_titulo": "Requisitos de Entrada:",
        "req_texto": "• Arquivos **GeoTIFF (.tif)** com bandas RGB.\n• Alta resolução espacial para identificação de texturas.\n• Capacidade de processamento de arquivos grandes.",
        "equipe_titulo": "Equipe de Desenvolvimento:",
        "footer": "Desenvolvido para pesquisa e otimização de inspeções em barragens e taludes."
    },
    "en": {
        "univ": "Federal University of Catalão (UFCAT)",
        "grupo": "Engineering Studies Research Group",
        "titulo_principal": "Welcome to GeoEyeAI Platform",
        "subtitulo": "Intelligent Geotechnical Monitoring via Computer Vision",
        "intro_titulo": "What is this tool?",
        "intro_texto": """
        This tool consists of advanced machine learning (Deep Learning) software designed to assist geotechnical engineers. 
        By uploading a high-resolution **orthomosaic** (.tiff files up to 7GB), the system uses **Semantic Segmentation** neural networks 
        to automatically identify and map potential anomalies in the structure.
        """,
        "funcionalidades_titulo": "Key Features:",
        "func_1": "**Automated Analysis:** Detection of pathologies such as erosion, cracks, ruptures, and water presence.",
        "func_2": "**Optimized Processing:** The system slices massive images into smaller blocks for detailed analysis without resolution loss.",
        "func_3": "**Human Validation:** Interactive interface for engineers to confirm, discard, or edit AI predictions.",
        "func_4": "**Official Reports:** Automatic generation of PDF reports with anomaly maps and statistics.",
        "requisitos_titulo": "Input Requirements:",
        "req_texto": "• **GeoTIFF (.tif)** files with RGB bands.\n• High spatial resolution for texture identification.\n• Large file processing capability.",
        "equipe_titulo": "Development Team:",
        "footer": "Developed for research and optimization of dam and slope inspections."
    }
}

t = textos[lang]

# =========================================================
# LAYOUT DA PÁGINA
# =========================================================

# 1. Cabeçalho Institucional (Logo + Nome)
col_logo, col_header = st.columns([1, 4])

with col_logo:
    # Defina o nome do arquivo local e a URL de backup
    arquivo_local = "logo_ufcat.png"
    url_logo = "https://files.cercomp.ufg.br/weby/up/519/o/UFCAT_-_Identidade_Visual_Completa.png"

    # Verifica se o arquivo existe na pasta
    if os.path.exists(arquivo_local):
        st.image(arquivo_local, use_container_width=True)
    else:
        # Se não achar o arquivo local, usa a URL da internet
        st.image(url_logo, use_container_width=True)

with col_header:
    st.markdown(f"### {t['univ']}")
    st.markdown(f"##### {t['grupo']}")


st.divider()

# 2. Título Principal
st.title(t["titulo_principal"])
st.subheader(t["subtitulo"])

# 3. Descrição do Software
st.markdown(f"### {t['intro_titulo']}")
st.info(t["intro_texto"])

# 4. Colunas de Funcionalidades e Requisitos
c1, c2 = st.columns(2)

with c1:
    st.markdown(f"#### {t['funcionalidades_titulo']}")
    st.markdown(f"- {t['func_1']}")
    st.markdown(f"- {t['func_2']}")
    st.markdown(f"- {t['func_3']}")
    st.markdown(f"- {t['func_4']}")

with c2:
    st.markdown(f"#### {t['requisitos_titulo']}")
    st.warning(t["req_texto"])
    
    st.caption("Exemplo de fluxo: Upload ➝ IA ➝ Validação ➝ PDF")

# 5. Equipe de Desenvolvimento (NOVO)
st.divider()
st.markdown(f"#### {t['equipe_titulo']}")

col_team1, col_team2, col_team3 = st.columns(3)

with col_team1:
    st.markdown("**Cauã Cerqueira Netto**")
with col_team2:
    st.markdown("**Savio Castanhede**")
with col_team3:
    st.markdown("**Wanderley Malaquias**")

st.divider()
st.caption(f"© {t['footer']}")