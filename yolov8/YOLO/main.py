import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from yolov1 import YoloV1
from yololoss import YoloV1Loss
from data_converter import YoloV1Dataset
import config

def train_fn(train_loader, model, optimizer, loss_fn, device):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)

        # Forward pass
        out = model(x)
        
        # Calculate loss
        loss = loss_fn(out, y)
        
        mean_loss.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss for epoch: {sum(mean_loss)/len(mean_loss)}")

def main():
    # --- 1. Load data.yaml ---
    try:
        with open(config.DATA_YAML, 'r') as f:
            data_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {config.DATA_YAML}: {e}")
        return

    # Extract paths and class count
    train_img_dir = os.path.join(data_config['path'], data_config['train'])
    train_label_dir = os.path.join(data_config['path'], data_config['train'].replace('images', 'labels'))
    
    # Val paths (assuming similar structure)
    # val_img_dir = os.path.join(data_config['path'], data_config['val'])
    # val_label_dir = os.path.join(data_config['path'], data_config['val'].replace('images', 'labels'))
    
    NUM_CLASSES = data_config['nc']
    print(f"Found {NUM_CLASSES} classes.")

    # --- 2. Initialize Model, Loss, Optimizer ---
    model = YoloV1(S=config.S, B=config.B, C=NUM_CLASSES).to(config.DEVICE)
    loss_fn = YoloV1Loss(S=config.S, B=config.B, C=NUM_CLASSES)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --- 3. Setup DataLoaders ---
    train_dataset = YoloV1Dataset(
        img_dir=train_img_dir,
        label_dir=train_label_dir,
        S=config.S,
        C=NUM_CLASSES,
        transform=None # Add transforms here
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=True, # Important for batch-dependent loss
    )
    
    # (Setup val_loader similarly)

    # --- 4. Training Loop ---
    print("Starting training...")
    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch+1} / {config.EPOCHS} ---")
        train_fn(train_loader, model, optimizer, loss_fn, config.DEVICE)
        
        # (Add validation loop here)
        
        # (Add save checkpoint logic here)

    print("Training complete.")

if __name__ == "__main__":
    # You will need PyYAML to read the data.yaml
    # pip install pyyaml
    main()