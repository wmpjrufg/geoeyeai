import streamlit as st
import pandas as pd
from PIL import Image
import tempfile
import os
import gc
import numpy as np
from geo import * # CORREÇÃO AQUI: Quebra de linha adicionada
lang = st.session_state.get("lang", "pt")

textos = {
    "pt": {
        "titulo": "Geração de Relatório de Inspeção por Ortofoto - Segmentação",
        "btn_analise": "Iniciar Predição",
        "btn_confirmar": "Confirmar Registro",
        "btn_descartar": "Descartar Registro",
        "txt_processando": "Analisando blocos e liberando memória...",
        "txt_pendente": "Aguardando carregamento da ortofoto.",
        "status_identificado": "Anomalias encontradas",
        "opacidade": "Opacidade da Máscara",
        "relatorio_header": "Relatório de Saída",
        "btn_download": "Baixar Relatório CSV",
        "txt_confirmadas": "anomalias confirmadas"
    },
    "en": {
        "titulo": "Orthoimage Inspection Report Generation - Segmentation",
        "btn_analise": "Start Prediction",
        "btn_confirmar": "Confirm Record",
        "btn_descartar": "Discard Record",
        "txt_processando": "Analyzing blocks and clearing memory...",
        "txt_pendente": "Waiting for orthoimage upload.",
        "status_identificado": "Anomalies found",
        "opacidade": "Mask Opacity",
        "relatorio_header": "Output Report",
        "btn_download": "Download CSV Report",
        "txt_confirmadas": "confirmed anomalies"
    }
}
t = textos[lang]

st.header(t["titulo"])

if 'results_data' not in st.session_state:
    st.session_state.results_data = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# --- BARRA LATERAL (Upload e Relatório) ---
with st.sidebar:
    st.divider()
    up_file = st.file_uploader("Upload Ortofoto (TIF)", type=["tif", "tiff"])
    
    if up_file and st.button(t["btn_analise"]):
        st.session_state.results_data = [] 
        gc.collect()
        
        model = load_segmentation_model()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tfile:
            tfile.write(up_file.getvalue())
            temp_path = tfile.name

        try:
            with st.status(t["txt_processando"]) as status_box:
                rgb_image = extract_img_geotrans_Tiff(temp_path)
                blocks, offsets = split_image(rgb_image, SIZE_INPUT)
                
                del rgb_image 
                gc.collect()

                results = []
                total = len(blocks)
                
                for idx, (block, offset) in enumerate(zip(blocks, offsets)):
                    # Filtro de blocos vazios
                    if np.mean(block) > 245 or np.mean(block) < 10: 
                        continue

                    prediction = model.predict(np.expand_dims(block, axis=0), verbose=0)
                    pred_mask = prediction.squeeze()
                    
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
                    
                    # Atualiza status a cada 50 blocos para não travar a UI
                    if idx % 50 == 0:
                         status_box.update(label=f"Progresso: {idx}/{total}")

                st.session_state.results_data = results
                st.session_state.current_index = 0
                status_box.update(label="Analise Concluida!", state="complete")
        
        except Exception as e:
            st.error(f"Erro: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            gc.collect()
            st.rerun()

    # --- LÓGICA DE DOWNLOAD ---
    st.divider()
    st.header(t["relatorio_header"])
    
    # Filtra apenas os confirmados
    confirmados = [r for r in st.session_state.results_data if r['status'] == 'Confirmado']
    
    if confirmados:
        df = pd.DataFrame([{
            'Pixel_Y': r['y'], 
            'Pixel_X': r['x'], 
            'Tipo_Anomalia': r['tipo']
        } for r in confirmados])
        
        st.success(f"{len(confirmados)} {t['txt_confirmadas']}")
        
        st.download_button(
            label=t["btn_download"],
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="inspecao_geoeye.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Nenhuma confirmação efetuada.")

# --- ÁREA DE EXIBIÇÃO ---
if st.session_state.results_data:
    res = st.session_state.results_data
    idx = st.session_state.current_index
    item = res[idx]

    c_nav = st.columns([1, 2, 1])
    with c_nav[0]:
        if st.button("⬅️") and idx > 0:
            st.session_state.current_index -= 1
            st.rerun()
    with c_nav[1]:
        st.write(f"**{idx+1} / {len(res)}** | {item['status']}")
    with c_nav[2]:
        if st.button("➡️") and idx < len(res) - 1:
            st.session_state.current_index += 1
            st.rerun()

    v_cols = st.columns([1, 1, 2])
    with v_cols[0]:
        if st.button(t["btn_confirmar"], use_container_width=True):
            st.session_state.results_data[idx]['status'] = 'Confirmado'
            st.rerun()
    with v_cols[1]:
        if st.button(t["btn_descartar"], use_container_width=True):
            st.session_state.results_data[idx]['status'] = 'Excluido'
            st.rerun()
    with v_cols[2]:
        opac = st.slider(t["opacidade"], 0.0, 1.0, 0.5)

    img_o = Image.fromarray(item['original'])
    img_m = Image.fromarray(item['mask_rgb'])
    c1, c2 = st.columns(2)
    c1.image(img_o, use_container_width=True, caption=f"Y:{item['y']} X:{item['x']}")
    c2.image(Image.blend(img_o, img_m, alpha=opac), use_container_width=True, caption=item['tipo'])
else:
    st.info(t["txt_pendente"])