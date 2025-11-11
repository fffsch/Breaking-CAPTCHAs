from ultralytics import YOLO
import cv2

from pathlib import Path
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import random

TEST_DIR = Path("images/clean_test")

# model = YOLO('runs_v1/detect/captcha_yolo_v87/weights/best.pt') #V1 - YOLOv8n
# model = YOLO('runs/detect/captcha_yolo_v82/weights/best.pt') #V2
model = YOLO('runs/detect/captcha_yolo_v8/weights/best.pt') #V3 - YOLO11s


# # # 2. Run prediction on a new image
# img_paths = sorted(list(map(str, list(TEST_DIR.glob("*.png")))))



# img_path = "images/test/07z0-0.png"
# results = model.predict(source=img_path, save=False, conf=0.5) # conf=0.5 means only accept 50%+ confident detections

# # 3. Process Results
# for result in results:
#     boxes = result.boxes.cpu().numpy()
    
#     detected_chars = []
#     for box in boxes:
#         cls = int(box.cls[0])           # Class ID (e.g., 5)
#         char = model.names[cls]         # Class Name (e.g., 'a')
#         x_coord = box.xyxy[0][0]        # X-coordinate (for sorting left-to-right)
#         conf = box.conf[0]              # Confidence score
#         detected_chars.append((x_coord, char, conf))

#     # CRITICAL: Captchas must be read left-to-right.
#     # YOLO doesn't guarantee order, so we MUST sort by the X-coordinate.
#     detected_chars.sort(key=lambda x: x[0])

#     # Join them into the final string
#     final_captcha = "".join([item[1] for item in detected_chars])
#     print(f"Solved Captcha: {final_captcha}")

#     # Optional: Show image with detections for debugging
#     res_plotted = result.plot()
#     plt.imshow(res_plotted, 'gray')
#     plt.axis('off')
#     plt.show()

CONF_THRESHOLD = 0.51

def load_model(path):
    return YOLO(path)

def get_ground_truth(img_path):
    return img_path.stem.split('-')[0].lower()

def predict_captcha(model, img_path, conf=0.3):
    # Run inference
    results = model.predict(
        source=str(img_path),
        save=False, 
        verbose=False, 
        conf=conf,
        device=0
        )
    
    result = results[0]
    
    detected = []
    boxes = result.boxes.cpu().numpy()
    for box in boxes:
        cls_id = int(box.cls[0])
        char = result.names[cls_id]
        x_coord = box.xyxy[0][0]
        conf_score = box.conf[0]
        detected.append((x_coord, char, conf_score))

    detected.sort(key=lambda x: x[0])
    pred_text = "".join([d[1] for d in detected])
    
    return pred_text, result 

# --- NEW: Plotting Function ---
def plot_ten_samples(samples):
    if not samples:
        return

    plt.figure(figsize=(15, 6))
    plt.suptitle("10 random samples", fontsize=14)

    for i, (img, gt, pred) in enumerate(samples[:10]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        
        # Set title color based on accuracy
        color = 'green' if gt == pred else 'red'
        plt.title(f"True: {gt}\nPred: {pred}", color=color, fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

def main():
    if not TEST_DIR.exists():
        print(f"[Error] Test directory not found: {TEST_DIR.absolute()}")
        return

    # model = load_model(MODEL_PATH)
    test_images = list(TEST_DIR.glob("*.png"))
    total_images = len(test_images)
    
    print(f"Starting evaluation on {total_images} images...")
    start_time = time.time()

    total_captchas = 0
    correct_captchas = 0
    correct_adjusted_captchas = 0

    total_chars = 0
    correct_chars = 0
    correct_adjusted_chars = 0
    
    plot_samples = [] # NEW: List to store images for plotting

    for i, img_path in enumerate(test_images):
        gt_text = get_ground_truth(img_path)
        # Get both prediction text AND result object
        pred_text, result = predict_captcha(model, img_path, CONF_THRESHOLD)

        # --- Accuracy Metrics ---
        if pred_text == gt_text:
            correct_captchas += 1
            correct_adjusted_captchas += 1
        # for i, j in zip(pred_text, gt_text):
        #     if (i == '0' and j == 'o') or (i == 'o' and j == '0'):
        #         correct_adjusted_captchas += 1

        total_chars += len(gt_text)
        matches = 0
        adjusted_matches = 0
        for j in range(min(len(gt_text), len(pred_text))):
            if gt_text[j] == pred_text[j]:
                matches += 1
                adjusted_matches += 1
            # elif (gt_text[j] == '0' and pred_text[j] == 'o') or (gt_text[j] == 'o' and pred_text[j] == '0'):
            #     adjusted_matches += 1
        correct_chars += matches
        correct_adjusted_chars += adjusted_matches
        total_captchas += 1
        

        # if (i + 1) % 100 == 0:
        #     print(f"Processed {i+1}...")

    for path in random.sample(test_images,10):
        gt_text = get_ground_truth(path)
        pred_text, result = predict_captcha(model, path, CONF_THRESHOLD)
        plotted_img_bgr = result.plot()
        plotted_img_rgb = cv2.cvtColor(plotted_img_bgr, cv2.COLOR_BGR2RGB)
        plot_samples.append((plotted_img_rgb, gt_text, pred_text))

    # --- Final Report ---
    elapsed = time.time() - start_time
    print("\n" + "="*40)
    print(f"CAPTCHA Accuracy: {(correct_captchas / total_captchas) * 100:.2f}%")
    # print(f"ADJUSTED CAPTCHA Accuracy: {(correct_adjusted_captchas / total_captchas) * 100:.2f}%")
    print(f"Character Accuracy: {(correct_chars / total_chars) * 100:.2f}%")
    # print(f"Adjusted Character Accuracy: {(correct_adjusted_chars / total_chars) * 100:.2f}%")
    print("="*40)

    plot_ten_samples(plot_samples)

if __name__ == '__main__':
    main()