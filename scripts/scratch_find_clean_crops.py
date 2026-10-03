import glob
import cv2
import numpy as np
from src.orchestration.stage2_pipeline import Stage2Pipeline

pipe = Stage2Pipeline(enable_debug_overlay=False)
all_crops = sorted(glob.glob("datasets/v4_crop/raw_crops/*.jpg"))
clean_crops = []

w, h = 1920, 1080
slot_w, slot_h = w // 5, h // 4
target_h = int(slot_h * 0.80)
target_w = int(slot_w * 0.50)

for p in all_crops:
    img = cv2.imread(p)
    if img is None:
        continue
    canvas = np.full((h, w, 3), 190, dtype=np.uint8)
    resized = cv2.resize(img, (target_w, target_h))
    canvas[100 : 100 + target_h, 100 : 100 + target_w] = resized
    dets = pipe.registry.detector.detect(canvas)
    persons = [d for d in dets if d.class_name == "person"]
    if len(persons) == 1 and persons[0].confidence >= 0.80:
        clean_crops.append(p.replace("\\", "/"))
        if len(clean_crops) >= 20:
            break

print("CLEAN_CROPS = [")
for p in clean_crops:
    print(f'    "{p}",')
print("]")
