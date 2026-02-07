import os
import sys

# --- 1. CRITICAL CONFIGURATION (MUST BE FIRST) ---
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
os.environ['SM_FRAMEWORK'] = 'tf.keras'

# --- 2. IMPORTS ---
import numpy as np
import cv2
import rasterio
import tensorflow as tf
from tensorflow import keras

# --- 3. COMPATIBILITY PATCH ---
if not hasattr(keras.utils, 'generic_utils'):
    import keras.utils
    keras.utils.generic_utils = keras.utils

import segmentation_models as sm
import gc
import streamlit as st

# --- REST OF YOUR CODE ---
MODEL_PATH = './models/model_30epochs_resnet50_2026-01-23.h5'
BACKBONE = 'resnet50'
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
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            
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
    
    # Cria uma matriz de 4 canais (RGBA) preenchida com ZEROS
    mask_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    
    detected_classes = []
    for i, class_name in enumerate(CLASSES):
        if np.any(mask_indices == i):
            mask_rgba[mask_indices == i, :3] = COLORS[class_name]
            mask_rgba[mask_indices == i, 3] = 255
            detected_classes.append(class_name)
            
    return mask_rgba, ", ".join(detected_classes)

# --- NOVA FUNÇÃO DE RECONSTRUÇÃO ---
def reconstruct_from_blocks(original_shape, results_data, opacity=0.6):
    """
    Reconstroi a imagem completa fundindo o original + máscaras editadas.
    original_shape: (height, width, channels) da imagem original completa.
    results_data: lista de dicionários contendo 'original', 'mask_rgba', 'y', 'x'.
    opacity: opacidade da máscara na imagem final.
    """
    full_h, full_w, _ = original_shape
    
    # Cria canvas RGB para o resultado final
    final_canvas = np.zeros((full_h, full_w, 3), dtype=np.uint8)
    
    for item in results_data:
        y, x = item['y'], item['x']
        
        # 1. Pega o bloco original e a máscara
        # mask_rgba é (512, 512, 4)
        block_rgb = item['original']
        mask_rgba = item['mask_rgba']
        
        # 2. Aplica a fusão (Alpha Blending) no bloco
        # Converte para float para calcular
        bg = block_rgb.astype(float)
        fg = mask_rgba[:, :, :3].astype(float)
        alpha = (mask_rgba[:, :, 3] / 255.0) * opacity # Aplica opacidade global
        alpha = np.expand_dims(alpha, axis=-1) # (512, 512, 1)
        
        # Fórmula: Output = Alpha * Foreground + (1 - Alpha) * Background
        blended_block = (alpha * fg + (1.0 - alpha) * bg).astype(np.uint8)
        
        # 3. Calcula as dimensões de recorte (caso seja borda da imagem)
        # O bloco pode ter padding (ser 512x512 mas a imagem acabar antes)
        bh, bw = blended_block.shape[:2]
        
        # Até onde vai na imagem final
        y_end = min(y + bh, full_h)
        x_end = min(x + bw, full_w)
        
        # Quanto devemos pegar do bloco (remove o padding se houver)
        h_take = y_end - y
        w_take = x_end - x
        
        # 4. Cola no canvas gigante
        final_canvas[y:y_end, x:x_end] = blended_block[:h_take, :w_take]
        
    return final_canvas