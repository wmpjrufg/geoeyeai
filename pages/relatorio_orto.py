import streamlit as st
import pandas as pd
from PIL import Image
import tempfile
import os
import gc
import numpy as np
from geo import * # Certifique-se de que o arquivo geo.py está na mesma pasta e ATUALIZADO

# Configuração de idioma
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

# Inicialização das variáveis de estado
if 'results_data' not in st.session_state:
    st.session_state.results_data = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# =========================================================
# 1. SEÇÃO DE UPLOAD E CONTROLE (Página Principal)
# =========================================================
st.divider()
up_file = st.file_uploader("Upload Ortofoto (TIF)", type=["tif", "tiff"])

if up_file and st.button(t["btn_analise"]):
    st.session_state.results_data = [] 
    gc.collect()
    
    # Carrega o modelo (função do seu geo.py)
    model = load_segmentation_model()
    
    # Salva arquivo temporário para processamento
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
                # Filtro simples para ignorar blocos vazios/brancos/pretos
                if np.mean(block) > 245 or np.mean(block) < 10: 
                    continue

                prediction = model.predict(np.expand_dims(block, axis=0), verbose=0)
                pred_mask = prediction.squeeze()
                
                if np.any(pred_mask[:, :, :-1] > 0.5):
                    # AGORA RECEBE RGBA (4 canais)
                    m_rgba, class_str = create_mask_overlay(pred_mask)
                    
                    results.append({
                        'original': block, 
                        'mask_rgba': m_rgba, # Guardamos como RGBA
                        'y': offset[0], 
                        'x': offset[1], 
                        'tipo': class_str, 
                        'status': 'Pendente'
                    })
                
                # Atualiza o status visual a cada 50 iterações
                if idx % 50 == 0:
                      status_box.update(label=f"Progresso: {idx}/{total}")

            st.session_state.results_data = results
            st.session_state.current_index = 0
            status_box.update(label="Analise Concluida!", state="complete")
    
    except Exception as e:
        st.error(f"Erro no processamento: {e}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        gc.collect()
        st.rerun()

# =========================================================
# 2. ÁREA DE REVISÃO E VISUALIZAÇÃO
# =========================================================
if st.session_state.results_data:
    st.divider()
    res = st.session_state.results_data
    idx = st.session_state.current_index
    item = res[idx]

    # --- Controles de Navegação ---
    c_nav = st.columns([1, 2, 1])
    with c_nav[0]:
        if st.button("⬅️") and idx > 0:
            st.session_state.current_index -= 1
            st.rerun()
    with c_nav[1]:
        # Mostra contador e status atual
        st.markdown(f"<div style='text-align: center;'><b>{idx+1} / {len(res)}</b><br>Status: {item['status']}</div>", unsafe_allow_html=True)
    with c_nav[2]:
        if st.button("➡️") and idx < len(res) - 1:
            st.session_state.current_index += 1
            st.rerun()

    # --- Botões de Ação ---
    v_cols = st.columns([1, 1, 2])
    with v_cols[0]:
        if st.button(t["btn_confirmar"], use_container_width=True):
            st.session_state.results_data[idx]['status'] = 'Confirmado'
            st.rerun()
    with v_cols[1]:
        if st.button(t["btn_descartar"], use_container_width=True):
            st.session_state.results_data[idx]['status'] = 'Excluido'
            st.rerun()
    
    # --- Slider de Opacidade ---
    with v_cols[2]:
        # Slider padrão
        opac = st.slider(t["opacidade"], 0.0, 1.0, 0.6)

    # --- Lógica de Visualização (SEMPRE ATIVA) ---
    
    # 1. Prepara a imagem Original (Converte para RGBA para permitir camadas)
    img_o = Image.fromarray(item['original']).convert("RGBA")
    
    c1, c2 = st.columns(2)
    c1.image(img_o, use_container_width=True, caption=f"Original (Y:{item['y']} X:{item['x']})")
    
    with c2:
        # AQUI FOI REMOVIDO O CHECKBOX E O IF/ELSE
        
        # 2. Pega a máscara RGBA salva
        # Nota: 'mask_rgba' vem do geo.py atualizado.
        mask_data = item['mask_rgba'].copy() # Copia para não alterar o original na memória
        
        # 3. Aplica a opacidade SOMENTE no canal Alpha (transparência)
        # Onde era transparente (0), continua transparente.
        # Onde era sólido (255), vira translúcido.
        mask_data[:, :, 3] = (mask_data[:, :, 3] * opac).astype(np.uint8)
        
        # Cria imagem da máscara
        mask_img = Image.fromarray(mask_data)
        
        # 4. Sobrepõe a máscara na original (Alpha Composite)
        # Isso garante que a máscara fique por cima sem escurecer o fundo
        img_final = Image.alpha_composite(img_o, mask_img)
        
        caption_txt = f"Predição: {item['tipo']}"
            
        st.image(img_final, use_container_width=True, caption=caption_txt)

    # =========================================================
    # 3. RELATÓRIO E DOWNLOAD
    # =========================================================
    st.divider()
    st.subheader(t["relatorio_header"])
    
    # Filtra apenas os itens com status 'Confirmado'
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
        st.info("Confirme ao menos um registro para gerar o relatório.")

else:
    # Mensagem de espera inicial
    st.info(t["txt_pendente"])
