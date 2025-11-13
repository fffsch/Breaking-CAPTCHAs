import os
import shutil
import random
import yaml
from pathlib import Path

SOURCE_IMG_DIR = Path("../data/yolo_dataset3_mask/images/train")
SOURCE_LABEL_DIR = Path("../data/yolo_dataset3_mask/labels/train")

# Destination path (where the final YOLO dataset will be created)
DEST_BASE_DIR = Path("../data/final_yolo_dataset_v4")

SPLIT_RATIO = 0.8

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
           'u', 'v', 'w', 'x', 'y', 'z']

DIRS = {
    'train_img': DEST_BASE_DIR / 'images/train',
    'val_img': DEST_BASE_DIR / 'images/val',
    'train_label': DEST_BASE_DIR / 'labels/train',
    'val_label': DEST_BASE_DIR / 'labels/val'
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


valid_pairs = []
all_images = list(SOURCE_IMG_DIR.glob("*.png"))

print(f"Found {len(all_images)} source images. Verifying pairs...")

for img_path in all_images:
    txt_path = SOURCE_LABEL_DIR / (img_path.stem + ".txt")
    if txt_path.exists():
        valid_pairs.append((img_path, txt_path))
    else:
        print(f"[Warning] Missing label for {img_path.name}, skipping.")

random.seed(42) # Ensures the same split every time you run it
random.shuffle(valid_pairs)

split_idx = int(len(valid_pairs) * SPLIT_RATIO)
train_set = valid_pairs[:split_idx]
val_set = valid_pairs[split_idx:]

print(f"Total valid pairs: {len(valid_pairs)}")
print(f"Training samples: {len(train_set)}")
print(f"Validation samples: {len(val_set)}")

def copy_set(dataset, img_dest, label_dest, set_name):
    print(f"Copying {len(dataset)} files to {set_name}...")
    for img_src, txt_src in dataset:
        shutil.copy2(img_src, img_dest / img_src.name)
        shutil.copy2(txt_src, label_dest / txt_src.name)

copy_set(train_set, DIRS['train_img'], DIRS['train_label'], "Train")
copy_set(val_set, DIRS['val_img'], DIRS['val_label'], "Validation")


yaml_content = {
    'path': str(DEST_BASE_DIR.absolute()), # Use absolute path to avoid confusion
    'train': 'images/train',
    'val': 'images/val',
    'nc': len(CLASSES),
    'names': CLASSES
}

yaml_path = DEST_BASE_DIR / 'data.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=None)

print("-" * 30)
print("SUCCESS!")
print(f"Final dataset is ready at: {DEST_BASE_DIR.absolute()}")
print(f"Use this path for your data.yaml file when training: {yaml_path.absolute()}")