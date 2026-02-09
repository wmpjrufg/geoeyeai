import streamlit as st
from fastai.vision.all import *
import gdown
import os
import numpy as np
import torch

# ==============================================================================
# 1. FORÇAR USO DE CPU (IMPORTANTE PARA DEPLOY ONLINE)
# ==============================================================================
torch.device('cpu')
defaults.device = torch.device('cpu')

# ==============================================================================
# 2. NAMESPACE (DECLARAÇÃO DAS FUNÇÕES DO TREINO)
# O load_learner exige que estas funções existam com o mesmo nome do treino.
# ==============================================================================
def get_y_fn(x): pass
def pixel_accuracy(inp, targ): pass
def mask_converter(arquivo_da_mascara): pass

# ==============================================================================
# 3. CONFIGURAÇÕES DO GOOGLE DRIVE
# ==============================================================================
FILE_ID = '1dtvjL8Pva7kTJwd7ssY-CMJF7elhFhj4'
MODEL_PATH = 'modelo_final_1024.pkl'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Baixando modelo do Google Drive...'):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, MODEL_PATH, quiet=False)
    
    # Carrega o modelo forçando ele para a CPU
    return load_learner(MODEL_PATH, cpu=True)

# ==============================================================================
# 4. INTERFACE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="GeoEye AI Predictor", layout="wide")

st.title("🛰️ GeoEye AI - Análise de Solo")
st.markdown("Identificação automática de **Água** e **Erosão** via celular.")

# Carregar modelo
try:
    learn = load_model()
    st.sidebar.success("✅ Modelo VGG16 carregado na CPU")
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")

# Upload da Imagem
uploaded_file = st.sidebar.file_uploader("Suba uma imagem de satélite", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file:
    col1, col2 = st.columns(2)
    
    # Abrir imagem
    img = PILImage.create(uploaded_file)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(img, use_container_width=True)

    if st.button('🚀 Executar Análise (CPU)'):
        with st.spinner('Processando pixels... Por ser CPU, isso pode levar alguns segundos.'):
            
            # Redimensionar para 512 para não travar a memória RAM da CPU
            img_input = img.reshape(512, 512)
            
            # Predição ignorando o terceiro parâmetro (probs) para evitar erro de 274GB
            with torch.no_grad():
                pred, _, _ = learn.predict(img_input)
            
            with col2:
                st.subheader("Resultado da Segmentação")
                
                # Converter máscara para numpy
                mask_np = np.array(pred)
                
                # Criar imagem colorida para o Streamlit
                # Cores originais: Azul (61,61,245) e Amarelo (221,255,51)
                display_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
                display_mask[mask_np == 1] = [61, 61, 245]   # Água
                display_mask[mask_np == 2] = [221, 255, 51]  # Erosão
                
                st.image(display_mask, caption="Mapa de Classes", use_container_width=True)
                
                # Cálculo de Estatísticas
                total_pixels = mask_np.size
                agua = (mask_np == 1).sum()
                erosao = (mask_np == 2).sum()
                
                # Exibir métricas
                m1, m2 = st.columns(2)
                m1.metric("Área de Água", f"{(agua/total_pixels)*100:.2f}%")
                m2.metric("Área de Erosão", f"{(erosao/total_pixels)*100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.write("GeoEyeAI Project - 2024")