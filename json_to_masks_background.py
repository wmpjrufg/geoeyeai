import os
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

# ===============================
# PATHS (AJUSTE SE NECESSÁRIO)
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ANNOTATIONS_PATH = os.path.join(
    BASE_DIR,
    'annotations',
    '_annotations.coco.json'
)

IMG_DIR = os.path.join(BASE_DIR, 'images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# COLOR MAP (CLASSES DO SEU JSON)
# ===============================
# Nomes exatamente como no COCO (em minúsculo)
COLOR_MAP = {
    'agua': (61, 61, 245),
    'erosao': (221, 255, 51),
    'trinca': (252, 128, 7),
    'ruptura': (36, 179, 83)
}

# ===============================
# INICIALIZA COCO
# ===============================
coco = COCO(ANNOTATIONS_PATH)

categories = coco.loadCats(coco.getCatIds())
id_to_name = {cat['id']: cat['name'].lower() for cat in categories}

print("\nCategorias detectadas:")
for k, v in id_to_name.items():
    print(f"  {k}: {v}")

# ===============================
# GERA MÁSCARAS
# ===============================
for img_id in coco.imgs:
    img_info = coco.imgs[img_id]

    print(f"\nGerando máscara para: {img_info['file_name']}")

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    height, width = img_info['height'], img_info['width']
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for ann in anns:
        if ann.get('iscrowd', 0) == 1:
            continue

        if not ann.get('segmentation'):
            continue

        cat_name = id_to_name.get(ann['category_id'])

        if cat_name not in COLOR_MAP:
            continue

        ann_mask = coco.annToMask(ann)

        if ann_mask.sum() == 0:
            continue

        colored_mask[ann_mask == 1] = COLOR_MAP[cat_name]

    mask_name = os.path.splitext(img_info['file_name'])[0] + '.png'
    output_path = os.path.join(OUTPUT_DIR, mask_name)

    Image.fromarray(colored_mask).save(output_path)
    print(f"✔ máscara salva em: {output_path}")

print("\n🎉 Processamento concluído! Todas as máscaras foram geradas.")