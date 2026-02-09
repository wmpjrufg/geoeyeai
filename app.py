import sys
from types import ModuleType

# ==============================================================================
# 1. PATCH DE COMPATIBILIDADE ULTRA (ANTES DE TUDO)
# Isso neutraliza o erro "ImportError: Could not import __path__"
# ==============================================================================
def patch_fastai():
    # Criamos módulos falsos para enganar o carregador do Pickle
    d = ModuleType("fastcore.dispatch")
    t = ModuleType("fastcore.transform")
    
    # Adicionamos os atributos que o carregador procura
    sys.modules["fastcore.dispatch"] = d
    sys.modules["fastcore.transform"] = t

patch_fastai()

# Agora sim os imports normais
import streamlit as st
from fastai.vision.all import *
import gdown
import os
import numpy as np
import torch

# ==============================================================================
# 2. CONFIGURAÇÕES TÉCNICAS
# ==============================================================================
torch.device('cpu')
defaults.device = torch.device('cpu')

# Funções que o seu modelo espera encontrar
def get_y_fn(x): pass
def pixel_accuracy(inp, targ): pass
def mask_converter(x): pass

FILE_ID = '1dtvjL8Pva7kTJwd7ssY-CMJF7elhFhj4'
MODEL_PATH = 'modelo_final_1024.pkl'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Baixando modelo do Google Drive...'):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    return load_learner(MODEL_PATH, cpu=True)

# ==============================================================================
# 3. INTERFACE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="GeoEye AI Predictor", layout="wide")

st.title("🛰️ GeoEye AI - Análise de Solo")
st.markdown("Identificação automática de **Água** e **Erosão**.")

try:
    learn = load_model()
    st.sidebar.success("✅ Modelo carregado com sucesso")
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.info("Dica: Se o erro for de 'fastcore', tente reiniciar o app no painel do Streamlit.")

uploaded_file = st.sidebar.file_uploader("Suba uma imagem de satélite", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file:
    col1, col2 = st.columns(2)
    img = PILImage.create(uploaded_file)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(img, use_container_width=True)

    if st.button('🚀 Executar Análise (CPU)'):
        with st.spinner('Processando pixels...'):
            # Redimensionamos para 512 para garantir velocidade na CPU
            img_input = img.reshape(512, 512)
            
            with torch.no_grad():
                # O _ ignora o pred_idx e o probs que causariam erro de RAM
                pred, _, _ = learn.predict(img_input)
            
            with col2:
                st.subheader("Resultado da Segmentação")
                mask_np = np.array(pred)
                
                # Cores: Azul (Água) e Amarelo (Erosão)
                display_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
                display_mask[mask_np == 1] = [61, 61, 245]   
                display_mask[mask_np == 2] = [221, 255, 51]  
                
                st.image(display_mask, caption="Mapa de Classes", use_container_width=True)
                
                # Estatísticas
                total = mask_np.size
                agua = (mask_np == 1).sum()
                erosao = (mask_np == 2).sum()
                
                m1, m2 = st.columns(2)
                m1.metric("Área de Água", f"{(agua/total)*100:.2f}%")
                m2.metric("Área de Erosão", f"{(erosao/total)*100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.write("GeoEyeAI Project - 2026")