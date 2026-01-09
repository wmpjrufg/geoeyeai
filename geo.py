import streamlit as st
import pandas as pd
from PIL import Image
import tempfile
import os
import gc # Garbage Collector para liberar RAM
from geo import * # Recupera o idioma da sessão
lang = st.session_state.get("lang", "pt")

textos_ia = {
    "pt": {
        "titulo": "Geração de Relatório de Inspeção por Ortofoto",
        "btn_analise": "Iniciar Análise Neural",
        "btn_confirmar": "Confirmar Registro",
        "btn_descartar": "Descartar Registro",
        "txt_processando": "Analisando blocos... Isso pode levar alguns minutos.",
        "txt_pendente": "Aguardando carregamento da ortofoto para processamento.",
        "status_identificado": "Anomalias identificadas"
    },
    "en": {
        "titulo": "Orthoimage Inspection Report Generation",
        "btn_analise": "Start Neural Analysis",
        "btn_confirmar": "Confirm Record",
        "btn_descartar": "Discard Record",
        "txt_processando": "Analyzing blocks... This may take a few minutes.",
        "txt_pendente": "Waiting for orthoimage upload to process.",
        "status_identificado": "Identified anomalies"
    }
}
ti = textos_ia[lang]

st.header(ti["titulo"])

# Inicialização segura do estado
if 'results_data' not in st.session_state:
    st.session_state.results_data = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

with st.sidebar:
    st.divider()
    up_file = st.file_uploader("Upload Ortofoto (TIF)", type=["tif", "tiff"])
    
    if up_file and st.button(ti["btn_analise"]):
        # 1. Carregar modelo apenas quando necessário
        model = load_segmentation_model()
        
        # 2. Processamento de arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tfile:
            tfile.write(up_file.getvalue())
            temp_path = tfile.name

        try:
            # 3. Extração e Split
            with st.status(ti["txt_processando"]) as status:
                rgb_image = extract_img_geotrans_Tiff(temp_path)
                blocks, offsets = split_image(rgb_image, SIZE_INPUT)
                
                # Deletar imagem original da RAM imediatamente após o split
                del rgb_image
                gc.collect()

                results = []
                total = len(blocks)
                
                # 4. Loop de Predição Otimizado
                for idx, (block, offset) in enumerate(zip(blocks, offsets)):
                    # Adicionamos uma pequena margem para evitar processar blocos puramente brancos/pretos (vazios)
                    if np.mean(block) > 250 or np.mean(block) < 5:
                        continue

                    prediction = model.predict(np.expand_dims(block, axis=0), verbose=0)
                    pred_mask = prediction.squeeze()
                    
                    # Filtro de confiança: 0.5
                    if np.any(pred_mask[:, :, :-1] > 0.5):
                        m_rgb, class_str = create_mask_overlay(pred_mask)
                        results.append({
                            'original': block, 
                            'mask_rgb': m_rgb, 
                            'y': offset[0], 
                            'x': offset[1],
                            'tipo': class_str, 
                            'status': 'Pendente'
                        })
                    
                    if idx % 100 == 0:
                        status.update(label=f"Processando: {idx}/{total} - {len(results)} {ti['status_identificado']}")

                st.session_state.results_data = results
                st.session_state.current_index = 0
                status.update(label="Análise Concluída!", state="complete")
        
        except Exception as e:
            st.error(f"Erro no processamento: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Limpeza final de memória
            gc.collect()
            st.rerun()

# --- ÁREA DE EXIBIÇÃO ---
if st.session_state.results_data:
    res = st.session_state.results_data
    idx = st.session_state.current_index
    item = res[idx]

    col_nav = st.columns([1, 2, 1])
    with col_nav[0]:
        if st.button("⬅️") and idx > 0:
            st.session_state.current_index -= 1
            st.rerun()
    with col_nav[1]:
        st.write(f"**Registro {idx+1} de {len(res)}** | Status: {item['status']}")
    with col_nav[2]:
        if st.button("➡️") and idx < len(res) - 1:
            st.session_state.current_index += 1
            st.rerun()

    v_cols = st.columns([1, 1, 2])
    with v_cols[0]:
        if st.button(ti["btn_confirmar"], use_container_width=True):
            st.session_state.results_data[idx]['status'] = 'Confirmado'
            st.rerun()
    with v_cols[1]:
        if st.button(ti["btn_descartar"], use_container_width=True):
            st.session_state.results_data[idx]['status'] = 'Excluido'
            st.rerun()
    with v_cols[2]:
        opac = st.slider("Overlay", 0.0, 1.0, 0.5)

    img_o = Image.fromarray(item['original'])
    img_m = Image.fromarray(item['mask_rgb'])
    c1, c2 = st.columns(2)
    c1.image(img_o, use_container_width=True, caption=f"Original (Y:{item['y']} X:{item['x']})")
    c2.image(Image.blend(img_o, img_m, alpha=opac), use_container_width=True, caption=item['tipo'])
else:
    st.info(ti["txt_pendente"])