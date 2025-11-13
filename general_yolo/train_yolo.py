from ultralytics import YOLO

# --- 1. Load the Model ---
# Option A: Transfer Learning. Starts with weights pre-trained on COCO objects.
# It trains faster and often achieves better accuracy even on custom data.
# model = YOLO('yolov8n.pt')

# Option B: standard architecture with RANDOM weights.
# significantly harder to train, requires more data, and takes longer.
# model = YOLO('yolov8n.yaml')

# --- 2. Start Training ---
# data: Absolute path to your data.yaml file
# epochs: 100 is a good starting point. The model uses 'early stopping' if it stops improving.
# imgsz: 640 is standard. You can lower to 320 if your captchas are very small to speed it up.
# device: 0 for GPU, 'cpu' for CPU

if __name__ == '__main__':
    model = YOLO('runs/detect/captcha_yolo_v11/weights/best.pt')
    model.train(data='final_yolo_dataset_v3/data.yaml',
                epochs=50,
                imgsz=640,
                batch=-1,
                patience = 10,
                name='masked_training_11s',
                device=0,
                dropout = 0.3,
                workers=8) # Lower workers if you still have issues (default is usually 8)