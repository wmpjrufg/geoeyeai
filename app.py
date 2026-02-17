import streamlit as st
from fastai.vision.all import *
import gdown
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# ==============================================================================
# 1. CONFIGURAÇÕES TÉCNICAS (Forçar CPU)
# ==============================================================================
defaults.device = torch.device('cpu')
torch.set_num_threads(max(1, os.cpu_count() or 1))  # opcional: usar threads CPU

# ==============================================================================
# 2. CONFIGURAÇÕES DO MODELO (AGORA: PESOS .PTH)
# ==============================================================================
FILE_ID = '1Z27-d2GdiLLzXroexjtgIuT6i0XzW_0p'       # <-- troque
WEIGHTS_PATH = 'modelo_weights.pth'       # arquivo leve

# parâmetros do treino (precisam bater)
meus_codes = ['Background', 'Agua', 'Erosao', 'Trinca', 'Ruptura']
TAMANHO_IMG = 768
arch = resnet18  # mesmo backbone do treino

@st.cache_resource
def load_model_cpu():
    # 1) Baixa o .pth (leve)
    if not os.path.exists(WEIGHTS_PATH):
        with st.spinner('Baixando pesos do Google Drive...'):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, WEIGHTS_PATH, quiet=False)

    # 2) Cria DLS dummy (só para montar o pipeline do fastai)
    dummy_img = PILImage.create(np.zeros((TAMANHO_IMG, TAMANHO_IMG, 3), dtype=np.uint8))
    dummy_msk = PILMask.create(np.zeros((TAMANHO_IMG, TAMANHO_IMG), dtype=np.uint8))

    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=meus_codes)),
        get_items=lambda _: [dummy_img],
        get_y=lambda x: dummy_msk,
        splitter=FuncSplitter(lambda o: False),
        item_tfms=[Resize(TAMANHO_IMG, method='pad', pad_mode='zeros',
                          resamples=(Image.BILINEAR, Image.NEAREST))],
        batch_tfms=[Normalize.from_stats(*imagenet_stats)]
    )
    dls = dblock.dataloaders(Path('.'), bs=1, num_workers=0)

    # 3) Reconstrói modelo e carrega pesos
    learn = unet_learner(dls, arch, pretrained=False)

    sd = torch.load(WEIGHTS_PATH, map_location='cpu')
    learn.model.load_state_dict(sd)
    learn.model.eval()
    learn.model.cpu()
    learn.model.float()  # segurança para CPU

    return learn

# ==============================================================================
# 3. INTERFACE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="GeoEye AI Predictor", layout="wide")

st.title("🛰️ GeoEye AI - Análise de Solo")
st.markdown("Identificação automática de **Água**, **Erosão**, **Trinca** e **Ruptura**.")

try:
    learn = load_model_cpu()
    st.sidebar.success("✅ Modelo carregado (CPU / Pesos .PTH)")
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

uploaded_file = st.sidebar.file_uploader("Suba uma imagem de satélite", type=["jpg", "png", "jpeg", "tif"])


if uploaded_file:
    col1, col2 = st.columns(2)
    img = PILImage.create(uploaded_file)

    with col1:
        st.subheader("Imagem Original")
        st.image(img, use_container_width=True)

    if st.button('🚀 Executar Análise'):
        with st.spinner('Processando pixels na CPU...'):
            # Mantém sua chamada original
            pred, pred_idx, probs = learn.predict(img)

        with col2:
            st.subheader("Resultado da Segmentação")
            mask_np = np.array(pred)

            display_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
            display_mask[mask_np == 1] = [61, 61, 245]   # Água
            display_mask[mask_np == 2] = [221, 255, 51]  # Erosão
            display_mask[mask_np == 3] = [252, 128, 7]   # Trinca
            display_mask[mask_np == 4] = [36, 179, 83]   # Ruptura

            st.image(display_mask, use_container_width=True)

            total = mask_np.size

            st.markdown("### 📊 Estatísticas de Área")
            cols = st.columns(4)

            classes = {
                "Água": 1,
                "Erosão": 2,
                "Trinca": 3,
                "Ruptura": 4
            }

            for i, (nome, id_cl) in enumerate(classes.items()):
                qtd = (mask_np == id_cl).sum()
                porcentagem = (qtd / total) * 100
                cols[i].metric(nome, f"{porcentagem:.2f}%")
