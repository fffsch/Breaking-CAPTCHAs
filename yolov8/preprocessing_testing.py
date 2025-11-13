import numpy as np
import cv2

from pathlib import Path

img_path = Path("../data/images/test/0col7w-0.png")
img = cv2.imread(str(img_path))

label_text = img_path.stem.split("-")[0].lower() 

def create_color_mask(image, lower_bgr, upper_bgr):
    lower_bound = np.array(lower_bgr, dtype=np.uint8)
    upper_bound = np.array(upper_bgr, dtype=np.uint8)

    mask = cv2.inRange(image, lower_bound, upper_bound)
    
    return mask

lower_black = (0, 0, 0)   # BGR for "000000"
upper_black = (0, 0, 0) # BGR for "000000"

h_img, w_img, _ = img.shape

line_mask = create_color_mask(img, lower_black, upper_black)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated_mask = cv2.dilate(line_mask, kernel, iterations=1)

inpainted_image = cv2.inpaint(img, dilated_mask, 3, cv2.INPAINT_TELEA)

# --- Preprocessing ---
gray = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2GRAY)
# blur = cv2.medianBlur(gray, 3)

# Adaptive threshold to get a binary image
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)

# Continue with previous morphological operations on the cleaned threshold
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
cleaned_thresh = cv2.dilate(cleaned_thresh, kernel, iterations=1)


# --- Contour Detection on cleaned_thresh ---
contours, _ = cv2.findContours(cleaned_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# rects = []
# for c in contours:
#     x, y, w, h = cv2.boundingRect(c)
#     # Filter: Must be tall enough (e.g., > 15% of image height) AND have reasonable area
#     if h > (h_img * 0.15) and (w * h) > 30: 
#             rects.append((x, y, w, h))

# # --- Smart Merging ---
# # rects = merge_vertical_contours(rects, threshold_x=5)

# rects.sort(key=lambda x: x[0])

# for rect, char in zip(rects, label_text):
#         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
#         cv2.putText(img, char, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, (0, 0, 255), 2)

cv2.imshow("Image", thresh)
cv2.imshow("Image2", img)
cv2.waitKey(0)
cv2.destroyAllWindows()