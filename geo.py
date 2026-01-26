import os
import sys

# --- 1. CRITICAL CONFIGURATION (MUST BE FIRST) ---
# Define environment variables BEFORE importing tensorflow or segmentation_models
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
os.environ['SM_FRAMEWORK'] = 'tf.keras'

# --- 2. IMPORTS ---
import numpy as np
import cv2
import rasterio
import tensorflow as tf
from tensorflow import keras

# --- 3. COMPATIBILITY PATCH (The Monkey Patch) ---
# This fixes the "AttributeError: module 'keras.utils' has no attribute 'generic_utils'"
# by redirecting the old call to the new location of the utils.
if not hasattr(keras.utils, 'generic_utils'):
    import keras.utils
    keras.utils.generic_utils = keras.utils

import segmentation_models as sm
import gc
import streamlit as st

# --- REST OF YOUR CODE ---
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
    # Check if GPUs are listed before setting config
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
    # Using sm.Unet with the correct backbone
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
