# app.py
import io
import zipfile
import random
from datetime import datetime
from typing import List, Tuple

import streamlit as st
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ================= CONFIG =================
ALLOWED_EXT = (".png", ".jpg", ".jpeg")
LABEL_POOL = ["Erosão", "Água", "Trinca"]

LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris."
)

FIXED_HEADER = {
    "university": "Universidade Federal de Catalão (UFCAT)",
    "group": "Grupo de Pesquisa de Estudos em Engenharia",
    "dissertation": "Dissertação de Mestrado: bla bla bla",
    "authors": "Savio; Wanderlei; Caua; bla bla bla",
    "report_title": "Relatório de Análise das Predições",
}

DISCLAIMER_TEXT = (
    "Disclaimer: This report was generated using an experimental software tool currently under development. "
    "The labels and descriptions are preliminary and intended solely for research, testing, and demonstration purposes."
)


# ================= HELPERS =================
def pick_labels(rng, pool, k_min=1, k_max=2):
    k = rng.randint(k_min, min(k_max, len(pool)))
    return rng.sample(pool, k)


def extract_images_from_zip(zip_bytes):
    images = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            if not info.is_dir() and info.filename.lower().endswith(ALLOWED_EXT):
                images.append((info.filename, z.read(info)))
    images.sort(key=lambda x: x[0].lower())
    return images


def wrap_text(c, text, max_width, font="Helvetica", size=10):
    c.setFont(font, size)
    words, lines, current = text.split(), [], ""
    for w in words:
        test = (current + " " + w).strip()
        if c.stringWidth(test, font, size) <= max_width:
            current = test
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def build_report_pdf(images, seed=42):
    rng = random.Random(seed)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    margin = 1.6 * cm
    gutter = 0.6 * cm
    col_left = (W - 2 * margin - gutter) * 0.62
    col_right = (W - 2 * margin - gutter) * 0.38
    row_h = 7.2 * cm

    def header(y):
        x = margin
        c.setFont("Helvetica-Bold", 14)
        c.drawString(x, y, FIXED_HEADER["university"]); y -= 0.55*cm
        c.setFont("Helvetica", 11)
        c.drawString(x, y, FIXED_HEADER["group"]); y -= 0.45*cm
        c.drawString(x, y, FIXED_HEADER["dissertation"]); y -= 0.45*cm
        c.drawString(x, y, f"Autores: {FIXED_HEADER['authors']}"); y -= 0.6*cm
        c.setFont("Helvetica-Bold", 13)
        c.drawString(x, y, FIXED_HEADER["report_title"]); y -= 0.4*cm
        c.setFont("Helvetica", 9)
        c.drawString(x, y, datetime.now().strftime("Generated: %Y-%m-%d %H:%M:%S"))
        y -= 0.45*cm

        c.setFont("Helvetica", 8)
        for line in wrap_text(c, DISCLAIMER_TEXT, W - 2*margin, size=8):
            c.drawString(x, y, line)
            y -= 0.35*cm

        c.line(margin, y, W - margin, y)
        return y - 0.6*cm

    y = header(H - margin)

    for i, (name, data) in enumerate(images, 1):
        if y - row_h < margin:
            c.showPage()
            y = header(H - margin)

        img = Image.open(io.BytesIO(data)).convert("RGB")
        iw, ih = img.size
        scale = min(col_left / iw, (5.2*cm) / ih)

        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, y, f"Image {i}: {name}")

        img_y = y - 0.35*cm
        c.drawImage(
            ImageReader(img),
            margin,
            img_y - ih*scale,
            width=iw*scale,
            height=ih*scale,
            mask="auto"
        )

        labels = pick_labels(rng, LABEL_POOL)
        xr = margin + col_left + gutter

        c.setFont("Helvetica-Bold", 11)
        c.drawString(xr, y, "Predicted label(s)")
        yr = y - 0.55*cm
        c.setFont("Helvetica", 11)
        for lab in labels:
            c.drawString(xr, yr, f"• {lab}")
            yr -= 0.55*cm

        ly = (img_y - ih*scale) - 0.4*cm
        c.setFont("Helvetica", 10)
        for line in wrap_text(c, LOREM, col_left)[:4]:
            c.drawString(margin, ly, line)
            ly -= 0.45*cm

        y -= row_h

    c.save()
    buf.seek(0)
    return buf.getvalue()


# ================= STREAMLIT UI =================
st.set_page_config(page_title="ZIP → PDF Report (UFCAT)", layout="centered")
st.title("ZIP Photos → PDF Report")

st.caption(
    "Experimental research tool under development. "
    "Automatically generates a structured PDF report from uploaded images."
)

seed = st.number_input("Random seed (labels)", 0, 10_000_000, 42, 1)

zip_file = st.file_uploader("Upload ZIP file with images", type=["zip"])

if zip_file and st.button("Generate PDF report", type="primary"):
    images = extract_images_from_zip(zip_file.getvalue())

    if not images:
        st.error("No valid images found in the ZIP.")
    else:
        pdf = build_report_pdf(images, seed)
        st.success("Report generated successfully.")
        st.download_button(
            "Download PDF",
            data=pdf,
            file_name="relatorio_predicoes.pdf",
            mime="application/pdf",
        )
