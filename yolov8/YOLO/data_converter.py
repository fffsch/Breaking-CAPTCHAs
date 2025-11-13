import torch
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from pathlib import Path

class YoloV1Dataset(Dataset):
    """
    This class is the "translator" you need.
    It reads your modern YOLO .txt labels and converts them
    to the fixed 7x7x(C+B*5) tensor required by YOLOv1.
    """
    def __init__(self, img_dir, label_dir, S=7, C=20, transform=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.S = S
        self.C = C
        self.B = 2 # Hardcoded for YOLOv1
        self.transform = transform
        
        # Get list of images that have corresponding labels
        self.image_files = []
        for img_file in sorted(self.img_dir.glob("*.png")): # Assumes .png, add .jpg etc. if needed
            label_file = self.label_dir / (img_file.stem + ".txt")
            if label_file.exists():
                self.image_files.append(img_file)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        label_path = self.label_dir / (img_path.stem + ".txt")

        # --- Load and Transform Image ---
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            # Note: Albumentations is great for transforms that
            # also update bounding boxes, but for this simple
            # conversion, we'll just resize.
            # A real implementation needs to scale boxes with image.
            image = cv2.resize(image, (448, 448)) # YOLOv1 standard size
            
        # Convert to tensor (H, W, C) -> (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        # --- Create Target Tensor (The "Translator") ---
        # Target tensor: [S, S, C + 5]
        # (We only store one box, so C+5. The loss fn will handle B=2)
        label_matrix = torch.zeros((self.S, self.S, self.C + 5))
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.split()
                class_id = int(parts[0])
                # These are normalized to image (0-1)
                x_center, y_center, width, height = [float(p) for p in parts[1:5]]
                
                # --- Convert to grid-cell-relative coordinates ---
                
                # 1. Find which grid cell this object belongs to
                # (e.g., if x_center=0.55 and S=7, 0.55*7 = 3.85, so cell_i = 3)
                cell_i = int(self.S * y_center)
                cell_j = int(self.S * x_center)
                
                # 2. Calculate x, y *relative to the cell*
                # (e.g., x_center=0.55 -> (0.55 * 7) - 3 = 3.85 - 3 = 0.85)
                x_cell = (self.S * x_center) - cell_j
                y_cell = (self.S * y_center) - cell_i
                
                # 3. Calculate width/height *relative to the image*
                # (These are already in that format)
                w_img = width
                h_img = height
                
                # --- Populate the target tensor ---
                
                # If this grid cell is not already taken
                if label_matrix[cell_i, cell_j, self.C] == 0:
                    # Set confidence to 1 (object present)
                    label_matrix[cell_i, cell_j, self.C] = 1
                    
                    # Set box coordinates
                    box_coords = torch.tensor([x_cell, y_cell, w_img, h_img])
                    label_matrix[cell_i, cell_j, self.C+1:self.C+5] = box_coords
                    
                    # Set one-hot encoding for class
                    label_matrix[cell_i, cell_j, class_id] = 1.0
                    
        return image, label_matrix