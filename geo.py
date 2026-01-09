import os
import numpy as np
import cv2
import rasterio
import tensorflow as tf
import segmentation_models as sm
import gc
import streamlit as st

# --- CONFIGURACAO CRITICA DE AMBIENTE ---
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
os.environ['SM_FRAMEWORK'] = 'tf.keras'

# CONFIGURACOES TECNICAS
MODEL_PATH = './models/model_30epochs_vgg16_2025-12-29.h5'
BACKBONE = 'vgg16'
SIZE_INPUT = 512 
CLASSES = ['agua', 'erosao', 'ruptura', 'trinca']
COLORS = {
    'agua': [61, 61, 245],
    'erosao': [221, 255, 51],
    'trinca': [252, 128, 7],
    'ruptura': [36, 179, 83]
}

def setup_tf_memory():
    """Configura o TensorFlow para evitar estouro de memoria."""
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

setup_tf_memory()

def extract_img_geotrans_Tiff(filepath_tiff: str) -> np.ndarray:
    with rasterio.open(filepath_tiff) as src:
        # Le as 3 bandas iniciais
        data = src.read((1, 2, 3))
        img = np.transpose(data, (1, 2, 0))
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img

def split_image(image: np.ndarray, block_size: int):
    h, w, c = image.shape
    blocks, offsets = [], []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i + block_size, j:j + block_size]
            bh, bw = block.shape[:2]
            if bh < block_size or bw < block_size:
                block = cv2.copyMakeBorder(block, 0, block_size - bh, 0, block_size - bw, 
                                         cv2.BORDER_CONSTANT, value=[0]*c)
            blocks.append(block)
            offsets.append((i, j))
    return blocks, offsets

@st.cache_resource
def load_segmentation_model():
    n_classes = len(CLASSES) + 1
    model = sm.Unet(BACKBONE, classes=n_classes, activation='softmax', encoder_weights=None)
    model.load_weights(MODEL_PATH)
    return model

def create_mask_overlay(pred_mask):
    mask_indices = np.argmax(pred_mask, axis=-1)
    mask_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    detected_classes = []
    for i, class_name in enumerate(CLASSES):
        if np.any(mask_indices == i):
            mask_rgb[mask_indices == i] = COLORS[class_name]
            detected_classes.append(class_name)
    return mask_rgb, ", ".join(detected_classes)