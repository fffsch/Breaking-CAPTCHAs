import cv2
import numpy as np
import os
from pathlib import Path

# Detail how to build a YOLO text-based captcha recognition model from scratch. The dataset is available.

# --- 1. Setup Directories ---
IMG_DIR = Path("images/train")
LABEL_DIR = Path("train/labels")
DEBUG_DIR = Path("train/debug_visuals") # NEW: Folder for verification images
PROCESS_DIR = Path("train/processed_visuals") # NEW: Folder for verification images

for d in [LABEL_DIR, DEBUG_DIR, PROCESS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- 2. Define Class List ---
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z']
char_to_id = {char: idx for idx, char in enumerate(CLASSES)}

def to_yolo_format(img_w, img_h, x, y, w, h):
    return ((x + w / 2) / img_w), ((y + h / 2) / img_h), (w / img_w), (h / img_h)

# --- 3. Main Processing Loop ---
image_files = sorted(list(IMG_DIR.glob("*.png")))
print(f"Starting processing of {len(image_files)} images...")

skipped_count = 0
sizes = []
for img_path in image_files:
    filename = img_path.name
    label_text = img_path.stem.split("-")[0]
    target_len = len(label_text)

    # Load Image
    img = cv2.imread(str(img_path))
    sizes.append(img.shape[1])
    if img is None:
        print(f"[Error] Could not read image: {filename}")
        skipped_count += 1
        continue
        
    h_img, w_img, _ = img.shape

    # --- Image Processing & Contour Finding ---
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 9, 2)
    bit_not = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(bit_not, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter noise and get initial rects
    rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 30]

    # --- NEW: Skip if nothing detected ---
    if len(rects) < target_len:
        print(f"[Warning] No contours found in {filename}. Skipping.")
        skipped_count += 1
        continue

    # --- Split/Filter Logic ---
    while len(rects) < target_len:
        rects.sort(key=lambda x: x[2]) 
        x, y, w, h = rects.pop()       
        rects.append((x, y, w // 2, h))
        rects.append((x + w // 2, y, w // 2, h))

    if len(rects) > target_len:
         rects.sort(key=lambda x: x[2] * x[3], reverse=True)
         rects = rects[:target_len]

    rects.sort(key=lambda x: x[0])

    # --- Save Labels & Visuals ---
    txt_path = LABEL_DIR / (img_path.stem + ".txt")
    debug_img = img.copy()

    with open(txt_path, 'w') as f:
        for rect, char in zip(rects, label_text):
            if char in char_to_id:
                cls_id = char_to_id[char]
                x, y, w, h = rect
                xc, yc, nw, nh = to_yolo_format(w_img, h_img, x, y, w, h)
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n")

                cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
                cv2.putText(debug_img, char, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 0, 255), 2)

    cv2.imwrite(str(DEBUG_DIR / filename), debug_img)
    cv2.imwrite(str(PROCESS_DIR / filename), bit_not)

print(f"Done! Processed {len(image_files) - skipped_count} images. Skipped {skipped_count}.")