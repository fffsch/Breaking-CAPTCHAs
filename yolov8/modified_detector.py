import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np

# --- 1. Configuration ---
MODEL_PATH = 'runs/detect/captcha_yolo_v8/weights/best.pt'
IMAGE_PATH = '../data/images/test/03yuav5-0.png' 

# Slicer settings:
# Adjust (128, 128) based on your captcha size. 
# It should be small enough to "zoom in" but large enough 
# to contain one or two characters.
SLICE_WH = (80,112)
OVERLAP_RATIO = (0, 0) # 20% overlap
# ---

# --- 2. Load Model & Image ---
try:
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model or image: {e}")
    exit()

# --- 3. Define Slicer Callback ---
def slicer_callback(slice: np.ndarray) -> sv.Detections:
    """
    This function is called by the slicer for each image slice.
    It runs inference using your local YOLO model.
    """
    # Run inference using Ultralytics model
    # verbose=False silences the console output for each slice
    result = model.predict(slice, verbose=False)[0] 
    
    # Convert Ultralytics results to supervision.Detections
    detections = sv.Detections.from_ultralytics(result)
    return detections

# --- 4. Set up Slicer ---
slicer = sv.InferenceSlicer(
    callback=slicer_callback,
    slice_wh=SLICE_WH,
    overlap_ratio_wh=OVERLAP_RATIO,
)

print(f"Slicing image ({image.shape}) into {SLICE_WH} chunks...")
detections = slicer(image)

# --- 5. Post-process (Non-Maximum Suppression) ---
# Slicing will create many duplicate boxes. NMS cleans them up.
# Adjust iou_threshold as needed.
print(f"Found {len(detections)} raw detections. Applying NMS...")
detections = detections.with_nms(threshold=0.5)
print(f"Found {len(detections)} unique detections after NMS.")

# --- 6. Annotate Results ---
# Create labels from the model's class names
labels = [
    f"{model.names[class_id]} ({confidence*100:.1f}%)"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

# Set up annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

# Annotate the original image
annotated_frame = box_annotator.annotate(
    scene=image.copy(),
    detections=detections)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame,
    detections=detections,
    labels=labels)

# --- 7. Display Result ---
sv.plot_image(annotated_frame, (12, 8))