from ultralytics import YOLO

# --- 1. Load the Model ---
# Option A (Recommended): Transfer Learning. Starts with weights pre-trained on COCO objects.
# It trains faster and often achieves better accuracy even on custom data.
# model = YOLO('yolov8n.pt')

# Option B (True "From Scratch"): standard architecture with RANDOM weights.
# significantly harder to train, requires more data, and takes longer.
# model = YOLO('yolov8n.yaml')

# --- 2. Start Training ---
# data: Absolute path to your data.yaml file
# epochs: 100 is a good starting point. The model uses 'early stopping' if it stops improving.
# imgsz: 640 is standard. You can lower to 320 if your captchas are very small to speed it up.
# device: 0 for GPU, 'cpu' for CPU (slow!)

# Wrap main execution in this guard block
if __name__ == '__main__':
    # 1. Load the Model
    model = YOLO('yolo11s.pt')

    # 2. Start Training
    # Ensure data path is correct and uses forward slashes or raw strings
    model.train(data='C:/Users/fvsch/OneDrive/Documents/Code/NUS/CV/CAPTCHA/final_yolo_dataset_v2/data.yaml',
                epochs=100,
                imgsz=640,
                batch=-1,
                patience = 10,
                name='captcha_yolo_v8',
                device=0,
                dropout = 0.3,
                workers=8) # Lower workers if you still have issues (default is usually 8)