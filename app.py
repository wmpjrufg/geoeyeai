import streamlit as st
from fastai.vision.all import *
import gdown
import os
import numpy as np
import torch

# ==============================================================================
# 1. CONFIGURAÇÕES TÉCNICAS (Forçar CPU)
# ==============================================================================
torch.device('cpu')
defaults.device = torch.device('cpu')

# OBRIGATÓRIO: Funções que o seu modelo espera encontrar
def get_y_fn(x): pass
def pixel_accuracy(inp, targ): pass
def mask_converter(x): pass

# ==============================================================================
# 2. CONFIGURAÇÕES DO MODELO
# ==============================================================================
FILE_ID = '1eklLbrDgvhuvK2erXltaOoijInLigzcB'
MODEL_PATH = 'modelo_final_1024.pkl'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Baixando modelo do Google Drive...'):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    
    # Com as versões fixas no requirements, o load_learner funciona direto
    return load_learner(MODEL_PATH, cpu=True)

# ==============================================================================
# 3. INTERFACE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="GeoEye AI Predictor", layout="wide")

st.title("🛰️ GeoEye AI - Análise de Solo")
st.markdown("Identificação automática de **Água** e **Erosão**.")

try:
    learn = load_model()
    st.sidebar.success("✅ Modelo carregado (Versão Legada)")
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")

uploaded_file = st.sidebar.file_uploader("Suba uma imagem de satélite", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file:
    col1, col2 = st.columns(2)
    img = PILImage.create(uploaded_file)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(img, use_container_width=True)

    if st.button('🚀 Executar Análise'):
        with st.spinner('Processando pixels na CPU...'):
            img_input = img.reshape(512, 512)
            
            with torch.no_grad():
                pred, _, _ = learn.predict(img_input)
            
            with col2:
                st.subheader("Resultado da Segmentação")
                mask_np = np.array(pred)
                
                display_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
                display_mask[mask_np == 1] = [61, 61, 245]   # Azul
                display_mask[mask_np == 2] = [221, 255, 51]  # Amarelo
                
                st.image(display_mask, use_container_width=True)
                
                total = mask_np.size
                agua = (mask_np == 1).sum()
                erosao = (mask_np == 2).sum()
                
                m1, m2 = st.columns(2)
                m1.metric("Área de Água", f"{(agua/total)*100:.2f}%")
                m2.metric("Área de Erosão", f"{(erosao/total)*100:.2f}%")