import streamlit as st
from fastai.vision.all import *
import gdown
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import zipfile
import io
import datetime
import tempfile
from fpdf import FPDF

# ==============================================================================
# 1. CONFIGURAÇÕES TÉCNICAS (CPU)
# ==============================================================================
defaults.device = torch.device('cpu')
torch.set_num_threads(max(1, os.cpu_count() or 1))

# ==============================================================================
# 2. CATÁLOGO DE MODELOS (DICIONÁRIO)
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
                "MobileNetV3small": {
                                    "file_id": "1xEhMtTqK_VXae14ZYe6JWNTylc0eIR9Q",
                                    "arch": "mobilenet_v3_small",
                                    "img_size": 768,
                                    "codes": ['Background', 'Agua', 'Erosao', 'Trinca', 'Ruptura']
                                },
        }

# ==============================================================================
# 3. FUNÇÕES UTILITÁRIAS
# ==============================================================================

def _resolve_arch(arch_name: str):
    arch_name = arch_name.lower().strip()
    if arch_name == "resnet18":
        return resnet18
    from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
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
    if model_key not in MODELS:
        raise ValueError("Modelo selecionado não existe no dicionário MODELS.")
    cfg = MODELS[model_key]
    file_id = cfg["file_id"]
    arch_name = cfg["arch"]
    img_size = int(cfg.get("img_size", 768))
    codes = cfg.get("codes", ['Background', 'Agua', 'Erosao', 'Trinca', 'Ruptura'])

    weights_path = _weights_path_for(model_key)
    if os.path.exists(weights_path):
        os.remove(weights_path)
    if not os.path.exists(weights_path):
        with st.spinner(f'Baixando pesos do Google Drive ({model_key})...'):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, weights_path, quiet=False)

    dls = _build_dummy_dls(img_size, codes)
    arch = _resolve_arch(arch_name)
    cut_fn = (lambda m: m.features) if "mobilenet" in arch_name.lower() else None

    learn = unet_learner(dls, arch, pretrained=False, cut=cut_fn)
    sd = torch.load(weights_path, map_location='cpu')
    learn.model.load_state_dict(sd, strict=True)
    learn.model.eval()
    learn.model.cpu()
    learn.model.float()
    return learn, cfg

# ==============================================================================
# 4. GERAÇÃO DE RELATÓRIO PDF (ADICIONADO)
# ==============================================================================
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 8, 'Universidade Federal de Catalão (UFCAT)', ln=True)
        self.set_font('Arial', '', 10)
        self.cell(0, 5, 'Grupo de Pesquisa de Estudos em Engenharia', ln=True)
        self.cell(0, 5, 'Projeto: Inspeção Automatizada de Barragens', ln=True)
        self.cell(0, 5, f'Data do Relatório: {datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}', ln=True)
        self.ln(2)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 8, 'Relatório de Análise das Predições', ln=True)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Página {self.page_no()}', 0, 0, 'C')

def generate_pdf(records):
    # O filtro de confirmados já é feito antes de passar pra essa função
    pdf = PDFReport()
    pdf.add_page()
    for i, rec in enumerate(records):
        # O Overlay já está pronto no nosso código original
        img_final = rec['img_overlay'].convert("RGB")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            img_final.save(tmp_img, quality=85)
            tmp_filename = tmp_img.name

        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, f"Arquivo de Imagem: {rec['filename']}", ln=True)
        y_start = pdf.get_y()
        pdf.image(tmp_filename, x=10, y=y_start, w=90)
        
        pdf.set_xy(105, y_start) 
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, "Anomalias Detectadas (Área):", ln=True)
        pdf.set_font('Arial', '', 10)
        
        # Filtra quais classes foram encontradas > 0%
        achou_algo = False
        for classe, pct in rec['stats'].items():
            if float(pct.replace('%', '')) > 0:
                achou_algo = True
                pdf.set_x(105) 
                pdf.cell(5, 6, "-", 0, 0)
                pdf.cell(0, 6, f"{classe}: {pct}", ln=True)
                
        if not achou_algo:
            pdf.set_x(105)
            pdf.cell(0, 6, "Nenhuma (Ok)", ln=True)
        
        pdf.set_y(y_start + 92) 
        comentario = rec.get('comentario', 'Sem observações adicionais.')
        if not comentario.strip():
            comentario = 'Sem observações adicionais.'
            
        pdf.set_font('Arial', '', 9)
        pdf.multi_cell(0, 5, comentario)
        pdf.ln(5) 
        
        if pdf.get_y() > 240: pdf.add_page()
        os.remove(tmp_filename)
        
    return bytes(pdf.output())

# ==============================================================================
# 5. INTERFACE STREAMLIT
# ==============================================================================
st.set_page_config(page_title="GeoEye AI Predictor", layout="wide")

st.title("🛰️ GeoEye AI - Inspeção em Geotecnia")
st.markdown("Identificação automática de **Água**, **Erosão**, **Trinca** e **Ruptura**.")

st.sidebar.header("⚙️ Configurações")
model_key = st.sidebar.selectbox("Selecione o modelo", list(MODELS.keys()), index=0)

try:
    learn, cfg = load_model_cpu(model_key)
    st.sidebar.success(f"✅ Modelo carregado: {model_key} (CPU / .PTH)")
except Exception as e:
    st.error(f"Erro ao carregar o modelo '{model_key}': {e}")
    st.stop()

uploaded_file = st.sidebar.file_uploader(
    "Suba uma imagem de satélite ou um arquivo ZIP",
    type=["jpg", "png", "jpeg", "tif", "tiff", "zip"]
)

# ==============================================================================
# 6. PREDICT, OVERLAY, CARROSSEL E RELATÓRIO
# ==============================================================================
if uploaded_file:
    if "file_id" not in st.session_state or st.session_state.file_id != uploaded_file.file_id:
        st.session_state.file_id = uploaded_file.file_id
        st.session_state.results = []
        st.session_state.current_idx = 0

    images_to_process = []
    if uploaded_file.name.lower().endswith('.zip'):
        with zipfile.ZipFile(uploaded_file, 'r') as z:
            for filename in z.namelist():
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    img_data = z.read(filename)
                    img = PILImage.create(io.BytesIO(img_data)).convert('RGB')
                    images_to_process.append((filename, img))
    else:
        img = PILImage.create(uploaded_file).convert('RGB')
        images_to_process.append((uploaded_file.name, img))

    if not images_to_process:
        st.warning("Nenhuma imagem válida encontrada no arquivo enviado.")
    else:
        if st.button('🚀 Executar Análise'):
            st.session_state.results = []
            st.session_state.current_idx = 0
            
            progress_text = "Processando imagens e gerando overlays..."
            my_bar = st.progress(0, text=progress_text)
            
            for i, (filename, img_pil_rgb) in enumerate(images_to_process):
                pred, pred_idx, probs = learn.predict(img_pil_rgb)
                mask_np_square = np.array(pred).astype(np.uint8)
                
                orig_w, orig_h = img_pil_rgb.size
                model_size = mask_np_square.shape[0]
                
                mask_pil_square = Image.fromarray(mask_np_square)

                if orig_w > orig_h: 
                    ratio = model_size / orig_w
                    new_h = int(orig_h * ratio)
                    pad_top = (model_size - new_h) // 2
                    crop_box = (0, pad_top, model_size, pad_top + new_h)
                else: 
                    ratio = model_size / orig_h
                    new_w = int(orig_w * ratio)
                    pad_left = (model_size - new_w) // 2
                    crop_box = (pad_left, 0, pad_left + new_w, model_size)
                
                mask_pil_cropped = mask_pil_square.crop(crop_box)
                mask_pil_final = mask_pil_cropped.resize((orig_w, orig_h), resample=Image.NEAREST)
                mask_np_final = np.array(mask_pil_final)

                rgba_mask_np = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
                alpha_value = 160 

                rgba_mask_np[mask_np_final == 1] = [61, 61, 245, alpha_value] 
                rgba_mask_np[mask_np_final == 2] = [221, 255, 51, alpha_value] 
                rgba_mask_np[mask_np_final == 3] = [252, 128, 7, alpha_value]  
                rgba_mask_np[mask_np_final == 4] = [36, 179, 83, alpha_value]  

                img_pil_rgba = img_pil_rgb.convert("RGBA")
                mask_pil_rgba = Image.fromarray(rgba_mask_np, 'RGBA')
                overlay_img = Image.alpha_composite(img_pil_rgba, mask_pil_rgba)

                total = mask_np_final.size
                stats = {}
                classes = {"Água": 1, "Erosão": 2, "Trinca": 3, "Ruptura": 4}
                for nome, id_cl in classes.items():
                    if total > 0:
                        qtd = int((mask_np_final == id_cl).sum())
                        porcentagem = (qtd / total) * 100
                    else:
                        porcentagem = 0.0
                    stats[nome] = f"{porcentagem:.2f}%"

                # Adiciona status e comentario para preencher o PDF depois
                st.session_state.results.append({
                    "filename": filename,
                    "img_original": img_pil_rgb,
                    "img_overlay": overlay_img, 
                    "stats": stats,
                    "status": "Pendente",
                    "comentario": ""
                })
                
                my_bar.progress((i + 1) / len(images_to_process), text=f"Processando: {filename}")
            
            my_bar.empty()

        if st.session_state.results:
            total_imgs = len(st.session_state.results)
            
            if total_imgs > 1:
                st.markdown("---")
                col_prev, col_idx, col_next = st.columns([1, 2, 1])
                with col_prev:
                    if st.button("⬅️ Voltar"):
                        st.session_state.current_idx = (st.session_state.current_idx - 1) % total_imgs
                with col_idx:
                    st.markdown(f"<h4 style='text-align: center;'>Imagem {st.session_state.current_idx + 1} de {total_imgs}</h4>", unsafe_allow_html=True)
                with col_next:
                    if st.button("Próximo ➡️"):
                        st.session_state.current_idx = (st.session_state.current_idx + 1) % total_imgs
                st.markdown("---")

            idx = st.session_state.current_idx
            result_data = st.session_state.results[idx]
            
            st.caption(f"**Arquivo em exibição:** {result_data['filename']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Imagem Original")
                st.image(result_data["img_original"], use_container_width=True)

            with col2:
                st.subheader("Overlay (Detecção)")
                st.image(result_data["img_overlay"], use_container_width=True)

            st.markdown("### 📊 Estatísticas de Área")
            cols = st.columns(4)
            for i, (nome, val) in enumerate(result_data["stats"].items()):
                cols[i].metric(nome, val)

            # --- CONTROLES DO RELATÓRIO PDF ---
            st.markdown("---")
            st.markdown(f"**Status atual:** {result_data['status']}")
            
            v_cols = st.columns([1, 1, 2])
            with v_cols[0]:
                if st.button("✅ Confirmar Registro", type="primary"):
                    st.session_state.results[idx]['status'] = 'Confirmado'
                    st.rerun()
            with v_cols[1]:
                if st.button("🗑️ Descartar Registro"):
                    st.session_state.results[idx]['status'] = 'Excluido'
                    st.rerun()
                    
            comentario_input = st.text_area(
                "Observações Técnicas (Opcional - aparecerá no PDF):", 
                value=result_data.get('comentario', ''), height=80, key=f"obs_{idx}"
            )
            if comentario_input != result_data.get('comentario', ''):
                st.session_state.results[idx]['comentario'] = comentario_input

            # --- SESSÃO FINAL PARA EXPORTAR ---
            st.divider()
            st.subheader("📄 Relatório de Saída")
            
            confirmados = [r for r in st.session_state.results if r['status'] == 'Confirmado']
            st.info(f"{len(confirmados)} registros confirmados para o PDF.")
            
            if st.button("Gerar Arquivo PDF"):
                if not confirmados:
                    st.error("Confirme ao menos um registro para gerar o PDF.")
                else:
                    with st.spinner("Gerando PDF..."):
                        pdf_bytes = generate_pdf(confirmados)
                        st.download_button(
                            label="Baixar Relatório PDF",
                            data=pdf_bytes,
                            file_name=f"Relatorio_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
        
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Preview da Imagem")
                if len(images_to_process) > 1:
                    st.caption(f"Arquivo: {images_to_process[0][0]} (+{len(images_to_process)-1} imagens aguardando análise)")
                st.image(images_to_process[0][1], use_container_width=True)