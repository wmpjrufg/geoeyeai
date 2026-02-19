import streamlit as st
from fastai.vision.all import *
import gdown
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# ==============================================================================
# 1. CONFIGURAÇÕES TÉCNICAS (CPU)
# ==============================================================================
defaults.device = torch.device('cpu')
torch.set_num_threads(max(1, os.cpu_count() or 1))

# ==============================================================================
# 2. CATÁLOGO DE MODELOS (DICIONÁRIO)
#    - Você só edita aqui: nome -> file_id + arch
#    - cut_fn será aplicado automaticamente se for mobilenet
# ==============================================================================
MODELS = {
                "MobileNetV2": {
                                    "file_id": "1OottSdIkgImYBmRXh81xgWtCqXpQtE1n",
                                    "arch": "mobilenet_v2",
                                    "img_size": 768,
                                    "codes": ['Background', 'Agua', 'Erosao', 'Trinca', 'Ruptura']
                                },
                "ResNet18": {
                                "file_id": "1MlDj-2B5Wad9O4pSqrLnaPWpR-R1xZGk",
                                "arch": "resnet18",
                                "img_size": 768,
                                "codes": ['Background', 'Agua', 'Erosao', 'Trinca', 'Ruptura']
                            },
                "ResNet34": {
                                "file_id": "1PIEUBRvRtPDLopizvUwq66d-m5TTvTU8",
                                "arch": "resnet34",
                                "img_size": 768,
                                "codes": ['Background', 'Agua', 'Erosao', 'Trinca', 'Ruptura']
                            },
                "ResNet50": {
                                    "file_id": "11-tpfad51B71PB4m-hXfwQXbuUjkQEEQ",
                                    "arch": "resnet50",
                                    "img_size": 768,
                                    "codes": ['Background', 'Agua', 'Erosao', 'Trinca', 'Ruptura']
                                },
                "MobileNetV3small": {
                                    "file_id": "1xEhMtTqK_VXae14ZYe6JWNTylc0eIR9Q",
                                    "arch": "mobilenet_v3_small",
                                    "img_size": 768,
                                    "codes": ['Background', 'Agua', 'Erosao', 'Trinca', 'Ruptura']
                                },
                "MobileNetV3large": {
                                    "file_id": "1_je_PzTVqyOyCLKiFKwY1vrayLEnS4UD",
                                    "arch": "mobilenet_v3_large",
                                    "img_size": 768,
                                    "codes": ['Background', 'Agua', 'Erosao', 'Trinca', 'Ruptura']
                                },
        }

# ==============================================================================
# 3. FUNÇÕES UTILITÁRIAS
# ==============================================================================

def _resolve_arch(arch_name: str):
    """Mapeia string -> função de arquitetura do torchvision/fastai."""
    arch_name = arch_name.lower().strip()

    # fastai já expõe resnet18, resnet34, etc.
    if arch_name == "resnet18":
        return resnet18

    # MobileNet: usar torchvision.models
    # (fastai usa torchvision por baixo; isso funciona bem com unet_learner)
    from torchvision.models import (
        mobilenet_v2,
        mobilenet_v3_small,
        mobilenet_v3_large
    )

    if arch_name == "mobilenet_v2":
        return mobilenet_v2
    if arch_name == "mobilenet_v3_small":
        return mobilenet_v3_small
    if arch_name == "mobilenet_v3_large":
        return mobilenet_v3_large

    raise ValueError(f"Arquitetura não suportada: {arch_name}")

def _weights_path_for(model_key: str) -> str:
    return "weights_current.pth"

def _build_dummy_dls(img_size: int, codes: list[str]):
    """Cria DLS dummy para montar o pipeline do fastai (mínima intervenção)."""
    dummy_img = PILImage.create(np.zeros((img_size, img_size, 3), dtype=np.uint8))
    dummy_msk = PILMask.create(np.zeros((img_size, img_size), dtype=np.uint8))

    dblock = DataBlock(
        blocks=(ImageBlock, MaskBlock(codes=codes)),
        get_items=lambda _: [dummy_img],
        get_y=lambda x: dummy_msk,
        splitter=FuncSplitter(lambda o: False),
        item_tfms=[Resize(img_size, method='pad', pad_mode='zeros',
                          resamples=(Image.BILINEAR, Image.NEAREST))],
        batch_tfms=[Normalize.from_stats(*imagenet_stats)]
    )
    dls = dblock.dataloaders(Path('.'), bs=1, num_workers=0)
    return dls

@st.cache_resource
def load_model_cpu(model_key: str):
    """Carrega modelo escolhido (CPU) replicando arch/cut do treino."""
    if model_key not in MODELS:
        raise ValueError("Modelo selecionado não existe no dicionário MODELS.")

    cfg = MODELS[model_key]
    file_id = cfg["file_id"]
    arch_name = cfg["arch"]
    img_size = int(cfg.get("img_size", 768))
    codes = cfg.get("codes", ['Background', 'Agua', 'Erosao', 'Trinca', 'Ruptura'])

    weights_path = _weights_path_for(model_key)

    # sempre substitui
    if os.path.exists(weights_path):
        os.remove(weights_path)

    # 1) Baixa o .pth
    if not os.path.exists(weights_path):
        with st.spinner(f'Baixando pesos do Google Drive ({model_key})...'):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, weights_path, quiet=False)

    # 2) DLS dummy (pipeline)
    dls = _build_dummy_dls(img_size, codes)

    # 3) Resolve arch e cut (se mobilenet)
    arch = _resolve_arch(arch_name)
    cut_fn = (lambda m: m.features) if "mobilenet" in arch_name.lower() else None

    # 4) Reconstrói learner e carrega pesos
    learn = unet_learner(
        dls,
        arch,
        pretrained=False,  # em deploy, quem manda é o state_dict do .pth
        cut=cut_fn
    )

    sd = torch.load(weights_path, map_location='cpu')
    learn.model.load_state_dict(sd, strict=True)  # garante compatibilidade com treino
    learn.model.eval()
    learn.model.cpu()
    learn.model.float()

    # Retorna também config, para usar codes/img_size se precisar
    return learn, cfg

# ==============================================================================
# 4. INTERFACE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="GeoEye AI Predictor", layout="wide")

st.title("🛰️ GeoEye AI - Inspeção em Geotecnia")
st.markdown("Identificação automática de **Água**, **Erosão**, **Trinca** e **Ruptura**.")

# Sidebar: seleção do modelo
st.sidebar.header("⚙️ Configurações")
model_key = st.sidebar.selectbox("Selecione o modelo", list(MODELS.keys()), index=0)

# Carrega modelo
try:
    learn, cfg = load_model_cpu(model_key)
    st.sidebar.success(f"✅ Modelo carregado: {model_key} (CPU / .PTH)")
except Exception as e:
    st.error(f"Erro ao carregar o modelo '{model_key}': {e}")
    st.stop()

uploaded_file = st.sidebar.file_uploader(
    "Suba uma imagem de satélite",
    type=["jpg", "png", "jpeg", "tif", "tiff"]
)

# ==============================================================================
# 5. PREDICT
# ==============================================================================
if uploaded_file:
    col1, col2 = st.columns(2)
    img = PILImage.create(uploaded_file)

    with col1:
        st.subheader("Imagem Original")
        st.image(img, use_container_width=True)

    if st.button('🚀 Executar Análise'):
        with st.spinner('Processando pixels na CPU...'):
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
                qtd = int((mask_np == id_cl).sum())
                porcentagem = (qtd / total) * 100
                cols[i].metric(nome, f"{porcentagem:.2f}%")
