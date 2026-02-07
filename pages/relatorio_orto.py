import streamlit as st
import pandas as pd
from PIL import Image
import tempfile
import os
import gc
import datetime
import numpy as np
from streamlit_drawable_canvas import st_canvas
from fpdf import FPDF
from geo import * # Configuração de idioma e a nova função reconstruct_from_blocks

lang = st.session_state.get("lang", "pt")

# ... (Seu dicionário de textos continua igual) ...
textos = {
    "pt": {
        "titulo": "Geração de Relatório de Inspeção - Segmentação",
        "btn_analise": "Iniciar Predição Completa",
        "btn_confirmar": "Confirmar Registro",
        "btn_descartar": "Descartar Registro",
        "txt_processando": "Processando todos os blocos...",
        "relatorio_header": "Relatório de Saída",
        "btn_download_pdf": "Baixar Relatório PDF",
        "lbl_comentario": "Observações Técnicas (Opcional)",
        "opacidade": "Opacidade",
        "txt_pendente": "Aguardando carregamento da ortofoto.",
        "txt_confirmadas": "anomalias confirmadas",
        "btn_reconstruir": "Gerar Ortofoto Final (Mosaico)",
        "txt_reconstruindo": "Montando ortofoto final... Isso pode demorar.",
        "btn_down_orto": "Baixar Ortofoto Mapeada (.tif)"
    },
    "en": {
        "titulo": "Inspection Report Generation",
        "btn_analise": "Start Full Prediction",
        "btn_confirmar": "Confirm Record",
        "btn_descartar": "Discard Record",
        "txt_processando": "Processing all blocks...",
        "relatorio_header": "Output Report",
        "btn_download_pdf": "Download PDF Report",
        "lbl_comentario": "Technical Observations",
        "opacidade": "Opacity",
        "txt_pendente": "Waiting for orthoimage upload.",
        "txt_confirmadas": "confirmed anomalies",
        "btn_reconstruir": "Generate Final Orthophoto (Mosaic)",
        "txt_reconstruindo": "Assembling final orthophoto... This may take a while.",
        "btn_down_orto": "Download Mapped Orthophoto (.tif)"
    }
}
t = textos[lang]

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# --- CLASSE PARA GERAR O PDF (Igual ao anterior) ---
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
    # Filtra apenas registros que têm alguma anomalia ou comentário para o PDF não ficar gigante
    # Se quiser imprimir TUDO no PDF, remova esse filtro.
    records_to_print = [r for r in records if r['tipo'] != "" or r['comentario'] != ""]
    
    pdf = PDFReport()
    pdf.add_page()
    for i, rec in enumerate(records_to_print):
        img_o = Image.fromarray(rec['original']).convert("RGBA")
        mask_data = rec['mask_rgba'].copy()
        mask_data[:, :, 3] = (mask_data[:, :, 3] * 0.7).astype(np.uint8) 
        mask_img = Image.fromarray(mask_data)
        img_final = Image.alpha_composite(img_o, mask_img).convert("RGB")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            img_final.save(tmp_img, quality=85)
            tmp_filename = tmp_img.name

        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, f"Bloco (Y:{rec['y']} X:{rec['x']})", ln=True)
        y_start = pdf.get_y()
        pdf.image(tmp_filename, x=10, y=y_start, w=90)
        
        pdf.set_xy(105, y_start) 
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, "Anomalias Detectadas:", ln=True)
        pdf.set_font('Arial', '', 10)
        
        tipo_str = rec['tipo'].strip()
        if not tipo_str:
            pdf.set_x(105)
            pdf.cell(0, 6, "Nenhuma (Ok)", ln=True)
        else:
            classes = tipo_str.split(', ')
            for cls in classes:
                if cls.strip():
                    pdf.set_x(105) 
                    pdf.cell(5, 6, "-", 0, 0)
                    pdf.cell(0, 6, cls.strip().capitalize(), ln=True)
        
        pdf.set_y(y_start + 92) 
        comentario = rec.get('comentario', 'Sem observações adicionais.')
        pdf.set_font('Arial', '', 9)
        pdf.multi_cell(0, 5, comentario)
        pdf.ln(5) 
        if pdf.get_y() > 240: pdf.add_page()
        os.remove(tmp_filename)
        
    return bytes(pdf.output())

# =========================================================
# INICIO DO APP
# =========================================================

st.header(t["titulo"])

if 'results_data' not in st.session_state:
    st.session_state.results_data = []
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'original_shape' not in st.session_state:
    st.session_state.original_shape = None

# 1. UPLOAD
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
            
            # Salva o shape original para reconstrução depois!
            st.session_state.original_shape = rgb_image.shape
            
            blocks, offsets = split_image(rgb_image, SIZE_INPUT)
            del rgb_image 
            gc.collect()

            results = []
            total = len(blocks)
            for idx, (block, offset) in enumerate(zip(blocks, offsets)):
                # Filtro de blocos totalmente brancos/pretos (bordas inválidas) pode ser mantido
                # Mas o filtro de anomalia foi removido.
                if np.mean(block) > 250 or np.mean(block) < 5: 
                    # Se for bloco vazio/ruido de borda, não adiciona
                    continue 

                pred = model.predict(np.expand_dims(block, axis=0), verbose=0).squeeze()
                
                # Independente se achou algo ou não, CRIAMOS o registro
                # Se pred for tudo zero, m_rgba será transparente e class_str vazio.
                m_rgba, class_str = create_mask_overlay(pred)
                
                # Define status inicial
                status_inicial = 'Pendente'
                if class_str == "":
                    status_inicial = 'Ok (Automático)' # Já marca como OK se não achou nada
                
                results.append({
                    'original': block, 
                    'mask_rgba': m_rgba, 
                    'y': offset[0], 'x': offset[1], 
                    'tipo': class_str, 
                    'status': status_inicial,
                    'comentario': '' 
                })
                
                if idx % 50 == 0: status_box.update(label=f"Progresso: {idx}/{total}")

            st.session_state.results_data = results
            st.session_state.current_index = 0
            status_box.update(label="Concluído!", state="complete")
    except Exception as e:
        st.error(f"Erro: {e}")
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)
        st.rerun()

# 2. REVISÃO
if st.session_state.results_data:
    res = st.session_state.results_data
    idx = st.session_state.current_index
    item = res[idx]

    # Navegação
    st.divider()
    c_nav = st.columns([1, 2, 1])
    with c_nav[0]:
        if st.button("⬅️") and idx > 0:
            st.session_state.current_index -= 1
            st.rerun()
    with c_nav[1]:
        st.markdown(f"<div style='text-align: center;'><b>{idx+1} / {len(res)}</b><br>Status: {item['status']}</div>", unsafe_allow_html=True)
    with c_nav[2]:
        if st.button("➡️") and idx < len(res) - 1:
            st.session_state.current_index += 1
            st.rerun()

    st.divider()

    # Controles
    v_cols = st.columns([1, 1, 1.5, 1.5])
    with v_cols[0]:
        if st.button(t["btn_confirmar"], type="primary"):
            st.session_state.results_data[idx]['status'] = 'Confirmado'
            st.rerun()
    with v_cols[1]:
        if st.button(t["btn_descartar"]):
            st.session_state.results_data[idx]['status'] = 'Excluido'
            st.rerun()
    with v_cols[2]:
        opac = st.slider(t["opacidade"], 0.0, 1.0, 0.6, label_visibility="collapsed")
    with v_cols[3]:
        modo_edicao = st.checkbox("✏️ Editar", value=False)

    # --- LÓGICA DE EDIÇÃO (Mantida igual) ---
    stroke_width = 3
    stroke_color = "#FF0000"
    fill_color_rgba = "rgba(0,0,0,0)"
    classe_selecionada = CLASSES[0]

    if modo_edicao:
        with st.sidebar:
            st.divider()
            st.header("Ferramentas de Edição")
            st.markdown("### ➕ Adicionar Classificação")
            classe_selecionada = st.selectbox("Selecione a Classe:", options=CLASSES)
            cor_rgb = COLORS[classe_selecionada]
            stroke_color = rgb_to_hex(cor_rgb)
            fill_color_rgba = f"rgba({cor_rgb[0]}, {cor_rgb[1]}, {cor_rgb[2]}, 0.5)"
            st.markdown(f"Cor Atual: <span style='color:{stroke_color}'>⬤</span>", unsafe_allow_html=True)
            stroke_width = st.slider("Espessura da Borda:", 1, 10, 3)

            st.divider()
            if st.button("🗑️ Limpar Tudo (Resetar)", type="secondary", use_container_width=True):
                item['mask_rgba'] = np.zeros((512, 512, 4), dtype=np.uint8)
                item['tipo'] = ""
                st.success("Limpo!")
                st.rerun()

    # Visualização
    img_o = Image.fromarray(item['original']).convert("RGBA")
    mask_visual = item['mask_rgba'].copy()
    mask_visual[:, :, 3] = (mask_visual[:, :, 3] * opac).astype(np.uint8)
    img_combined = Image.alpha_composite(img_o, Image.fromarray(mask_visual))

    c1, c2 = st.columns(2)
    c1.image(img_o, use_column_width=True, caption=f"Original")
    
    with c2:
        lbl = item['tipo'] if item['tipo'] else "Sem anomalias"
        if not modo_edicao:
            st.image(img_combined, use_column_width=True, caption=f"Predição: {lbl}")
        else:
            canvas = st_canvas(
                fill_color=fill_color_rgba,
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=img_combined,
                update_streamlit=True,
                height=512, width=512,
                drawing_mode="rect", key=f"canvas_{idx}"
            )
            
            if st.button("💾 Salvar Adição", type="primary"):
                if canvas.image_data is not None:
                    drawn = canvas.image_data[:, :, 3] > 0
                    c_rgb = COLORS[classe_selecionada]
                    item['mask_rgba'][drawn, :3] = c_rgb
                    item['mask_rgba'][drawn, 3] = 255 
                    
                    classes_presentes = []
                    for cls_name, cls_color in COLORS.items():
                        match_color = np.all(item['mask_rgba'][:, :, :3] == cls_color, axis=-1)
                        match_alpha = item['mask_rgba'][:, :, 3] > 0
                        if np.any(match_color & match_alpha):
                            classes_presentes.append(cls_name)
                    item['tipo'] = ", ".join(classes_presentes)
                    st.success("Atualizado!")
                    st.rerun()

    # Comentários
    st.markdown(f"**{t['lbl_comentario']}**")
    comentario_input = st.text_area("Descreva a anomalia (aparecerá no PDF):", 
                                    value=item.get('comentario', ''), height=100, key=f"obs_{idx}")
    if comentario_input != item.get('comentario', ''):
        st.session_state.results_data[idx]['comentario'] = comentario_input

    # =========================================================
    # 3. EXPORTAÇÃO (PDF)
    # =========================================================
    st.divider()
    st.subheader(t["relatorio_header"])
    
    confirmados = [r for r in st.session_state.results_data if r['status'] == 'Confirmado']
    
    c_pdf, c_orto = st.columns(2)
    
    with c_pdf:
        st.info(f"{len(confirmados)} registros confirmados para o PDF.")
        if st.button("📄 Gerar Arquivo PDF"):
            if not confirmados:
                st.error("Confirme ao menos um registro com anomalia.")
            else:
                with st.spinner("Gerando PDF..."):
                    pdf_bytes = generate_pdf(confirmados)
                    st.download_button(
                        label=t["btn_download_pdf"],
                        data=pdf_bytes,
                        file_name=f"Relatorio_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf"
                    )

    # =========================================================
    # 4. GERAÇÃO DE ORTOFOTO FINAL (NOVO)
    # =========================================================
    with c_orto:
        st.success("Reconstrução da Imagem")
        if st.button(t["btn_reconstruir"]):
            with st.spinner(t["txt_reconstruindo"]):
                # Chama a função nova do geo.py
                full_image = reconstruct_from_blocks(
                    st.session_state.original_shape,
                    st.session_state.results_data,
                    opacity=0.7 # Opacidade fixa de 0.7 para ficar visível
                )
                
                # Salva em memória para download
                # Usamos TIFF para manter qualidade, ou PNG se preferir
                success, encoded_img = cv2.imencode('.tif', cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
                
                if success:
                    st.download_button(
                        label=t["btn_down_orto"],
                        data=encoded_img.tobytes(),
                        file_name=f"Ortofoto_Mapeada_{datetime.datetime.now().strftime('%Y%m%d')}.tif",
                        mime="image/tiff",
                        type="primary"
                    )
                else:
                    st.error("Erro ao codificar a imagem final.")

else:
    st.info(t["txt_pendente"])