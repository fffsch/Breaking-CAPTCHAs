import cv2
import numpy as np
import os
from pathlib import Path
import shutil

# --- 1. Setup Directories ---
BASE_DIR = Path("yolo_dataset3_mask") # Centralize output
IMG_OUTPUT_DIR = BASE_DIR / "images/train"
LABEL_OUTPUT_DIR = BASE_DIR / "labels/train"
DEBUG_DIR = BASE_DIR / "debug_visuals"
PROCESSED_DIR = BASE_DIR / "processed_images"

# Clean start for output directories
for d in [IMG_OUTPUT_DIR, LABEL_OUTPUT_DIR, DEBUG_DIR, PROCESSED_DIR]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

# Source directory for your raw captcha images
RAW_IMG_DIR = Path("images/train") 

# --- 2. Define Class List ---
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z']
char_to_id = {char: idx for idx, char in enumerate(CLASSES)}

# --- Helper Functions ---
def to_yolo_format(img_w, img_h, x, y, w, h):
    # Clamp values to ensure they don't slightly exceed 1.0 due to rounding
    xc = max(0.0, min(1.0, (x + w / 2) / img_w))
    yc = max(0.0, min(1.0, (y + h / 2) / img_h))
    nw = max(0.0, min(1.0, w / img_w))
    nh = max(0.0, min(1.0, h / img_h))
    return xc, yc, nw, nh

def merge_vertical_contours(rects, threshold_x=5):
    """
    Merges contours that are vertically aligned and close to each other.
    Essential for 'i', 'j', ':', etc.
    """
    if not rects:
        return []
        
    rects.sort(key=lambda r: (r[0], r[1]))
    
    merged = []
    used = [False] * len(rects)
    
    for i in range(len(rects)):
        if used[i]:
            continue
            
        x1, y1, w1, h1 = rects[i]
        cx1 = x1 + w1 // 2
        
        merged_rect = (x1, y1, w1, h1)
        used[i] = True
        
        for j in range(i + 1, len(rects)):
            if used[j]:
                continue
                
            x2, y2, w2, h2 = rects[j]
            cx2 = x2 + w2 // 2
            
            if abs(cx1 - cx2) < threshold_x:
                nx = min(merged_rect[0], x2)
                ny = min(merged_rect[1], y2)
                nw = max(merged_rect[0] + merged_rect[2], x2 + w2) - nx
                nh = max(merged_rect[1] + merged_rect[3], y2 + h2) - ny
                
                merged_rect = (nx, ny, nw, nh)
                used[j] = True
        
        merged.append(merged_rect)
        
    return merged

def create_color_mask(image, lower_bgr, upper_bgr):
    # Define the lower and upper bounds
    lower_bound = np.array(lower_bgr, dtype=np.uint8)
    upper_bound = np.array(upper_bgr, dtype=np.uint8)

    mask = cv2.inRange(image, lower_bound, upper_bound)
    
    return mask

lower_black = (0, 0, 0)   # BGR for "000000"
upper_black = (0, 0, 0)   # BGR for "000000"

# --- 3. Main Processing Loop ---
image_files = sorted(list(RAW_IMG_DIR.glob("*.png")))
print(f"Starting strict processing of {len(image_files)} images...")

processed_count = 0
skipped_count = 0

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

for img_path in image_files:
    filename = img_path.name
    label_text = img_path.stem.split("-")[0].lower() 
    target_len = len(label_text)

    img = cv2.imread(str(img_path))
    if img is None:
        continue
        
    line_mask = create_color_mask(img, lower_black, upper_black)
        
    dilated_mask = cv2.dilate(line_mask, kernel, iterations=1)

    inpainted_image = cv2.inpaint(img, dilated_mask, 3, cv2.INPAINT_TELEA)
    
    h_img, w_img, _ = img.shape

    # --- Preprocessing ---
    gray = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.medianBlur(gray, 3)
    
    # Adaptive threshold to get a binary image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Continue with previous morphological operations on the cleaned threshold
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_thresh = cv2.dilate(thresh, kernel, iterations=1)


    # --- Contour Detection on cleaned_thresh ---
    contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # Filter: Must be tall enough (e.g., > 15% of image height) AND have reasonable area
        if h > (h_img * 0.15) and (w * h) > 30: 
             rects.append((x, y, w, h))

    # --- Smart Merging ---
    rects = merge_vertical_contours(rects, threshold_x=5)

    # --- STRICT VALIDATION ---
    if len(rects) != target_len:
        print(f"[Skip] {filename}: Found {len(rects)} contours, expected {target_len}")
        skipped_count += 1
        continue

    rects.sort(key=lambda x: x[0])

    # --- Save Validated Data ---
    processed_count += 1
    
    shutil.copy(str(img_path), str(IMG_OUTPUT_DIR / filename))

    txt_path = LABEL_OUTPUT_DIR / (img_path.stem + ".txt")
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
                            0.5, (0, 0, 255), 2)

    cv2.imwrite(str(DEBUG_DIR / ("debug_" + filename)), debug_img)
    cv2.imwrite(str(PROCESSED_DIR / ("debug_" + filename)), cleaned_thresh)

print("-" * 30)
print(f"Completed.")
print(f"Successfully processed: {processed_count}")
print(f"Skipped (low confidence): {skipped_count}")
print(f"Yield rate: {(processed_count / len(image_files)) * 100:.1f}%")

