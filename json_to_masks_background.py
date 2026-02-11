
import os
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
from collections import Counter

BASE_DIR = os.getcwd()
ANNOTATIONS_PATH = os.path.join(BASE_DIR, 'annotations', '_annotations.coco.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_MAP = {
                'agua': (61, 61, 245),
                'erosao': (221, 255, 51),
                'trinca': (252, 128, 7),
                'ruptura': (36, 179, 83)
            }

coco = COCO(ANNOTATIONS_PATH)
categories = coco.loadCats(coco.getCatIds())
id_to_name = {cat['id']: cat['name'].lower() for cat in categories}
print("\nDetected categories:")
for k, v in id_to_name.items():
    print(f"  {k}: {v}")
