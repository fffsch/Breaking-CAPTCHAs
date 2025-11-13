import numpy as np
import cv2
import shutil
from pathlib import Path

BASE_DIR = Path("../data/images")
IMG_DIR = BASE_DIR / "test"
CLEAN_DIR = BASE_DIR / "clean_test_processed"

for d in [CLEAN_DIR]:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

def create_color_mask(image, lower_bgr, upper_bgr):
    lower_bound = np.array(lower_bgr, dtype=np.uint8)
    upper_bound = np.array(upper_bgr, dtype=np.uint8)
    
    mask = cv2.inRange(image, lower_bound, upper_bound)
    
    return mask

lower_black = (0, 0, 0)   # BGR for "000000"
upper_black = (0, 0, 0) # BGR for "000000"

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

if __name__ == "__main__":
    image_files = sorted(list(IMG_DIR.glob("*.png")))

    for img_path in image_files:
        img = cv2.imread(str(img_path))
        filename = img_path.name

        line_mask = create_color_mask(img, lower_black, upper_black)

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # dilated_mask = cv2.dilate(line_mask, kernel, iterations=1)

        inpainted_image = cv2.inpaint(img, line_mask, 3, cv2.INPAINT_TELEA)

        gray = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
        
        cleaned_thresh = cv2.dilate(thresh, kernel, iterations=1)

        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        image = cv2.bitwise_not(cleaned_thresh)
        
        clean_path = CLEAN_DIR / img_path

        cv2.imwrite(str(CLEAN_DIR / filename), image)