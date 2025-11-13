import torch

DATA_YAML = '../final_yolo_dataset_v2/data.yaml'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# YOLOv1 parameters
S = 7  # Grid size (7x7)
B = 2  # Number of bounding boxes per grid cell
# C will be loaded from data.yaml

# Training parameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True
EPOCHS = 100