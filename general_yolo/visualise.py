import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

# === CONFIGURATION ===
MODEL_PATH = 'runs/detect/captcha_yolo_v11/weights/best.pt'
# MODEL_PATH = 'runs/detect/masked_training/weights/best.pt'
TEST_DIR = Path('images/clean_test_processed')
CONF_THRESHOLD = 0.5
# =====================

def get_ground_truth(img_path):
    # Assumes filename format "label-uuid.png" or "label.png"
    return img_path.stem.split('-')[0].lower()

def predict_and_parse(model, img_path):
    """Runs prediction and returns parsed text and the plotted image."""
    # Run inference
    results = model.predict(source=str(img_path), conf=CONF_THRESHOLD, verbose=False)
    r = results[0]
    
    # Parse detections into text (sorted left-to-right)
    detections = []
    for box in r.boxes.cpu().numpy():
        x = box.xyxy[0][0]
        char = r.names[int(box.cls[0])]
        detections.append((x, char))
    
    detections.sort(key=lambda x: x[0])
    pred_text = "".join([d[1] for d in detections])
    
    # Generate visualization (YOLO built-in plotter)
    # It returns BGR, so we convert to RGB for matplotlib
    plotted_bgr = r.plot()
    plotted_rgb = cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
    
    return pred_text, plotted_rgb

def plot_grid(samples):
    """Displays a 2x5 grid of results."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f'Random 10 Test Samples (Conf > {CONF_THRESHOLD})', fontsize=16)
    axes = axes.flatten()

    for i, (img, gt, pred) in enumerate(samples):
        ax = axes[i]
        ax.imshow(img)
        
        # Color-code title based on accuracy
        color = 'green' if gt == pred else 'red'
        title = f"True: {gt}\nPred: {pred}"
        
        ax.set_title(title, color=color, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 1. Validation
    if not TEST_DIR.exists():
        print(f"Error: Test directory not found at {TEST_DIR.absolute()}")
        exit()
        
    # 2. Load Model & Images
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    all_images = list(TEST_DIR.glob("*.png"))
    if len(all_images) < 10:
        print(f"Warning: Not enough images for 10 random samples. Found {len(all_images)}.")
        selected_images = all_images
    else:
        selected_images = random.sample(all_images, 10)

    # 3. Process Random Samples
    print("Processing 10 random images...")
    samples_to_plot = []
    for img_path in selected_images:
        gt = get_ground_truth(img_path)
        pred, plotted_img = predict_and_parse(model, img_path)
        samples_to_plot.append((plotted_img, gt, pred))

    # 4. Display
    plot_grid(samples_to_plot)