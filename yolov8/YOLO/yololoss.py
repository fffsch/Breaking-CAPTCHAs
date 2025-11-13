import torch
import torch.nn as nn
from utils import intersection_over_union # We will need a helper for IoU

class YoloV1Loss(nn.Module):
    """
    The complex YOLOv1 loss function, which calculates:
    1. Localization (bounding box) loss for responsible boxes.
    2. Confidence loss (object vs. no-object).
    3. Classification loss for cells containing objects.
    """
    def __init__(self, S=7, B=2, C=20, lambda_coord=5.0, lambda_noobj=0.5):
        super(YoloV1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, target):
        # predictions shape: (BATCH_SIZE, S, S, C + B*5)
        # target shape: (BATCH_SIZE, S, S, C + 5)
        
        # We need to find the "responsible" box 'b' for each grid cell.
        # This is the box 'b' (out of B) with the highest IoU with the target.
        
        # Note: This is a simplified version for demonstration.
        # A full implementation requires careful IoU calculation.
        # Let's assume the target tensor is formatted correctly by the dataset.
        
        # Target format: [prob_obj (1), x, y, w, h, class_0, class_1, ...]
        # We need to split this for the loss calculation
        
        # Slices for target tensor (C+5)
        obj_mask = target[..., 0:1] # Cell contains object (1) or not (0)
        
        # Slices for prediction tensor (C + B*5)
        # We will assume B=2 for this example
        box1_pred = predictions[..., self.C+1:self.C+5]
        box1_conf = predictions[..., self.C:self.C+1]
        
        box2_pred = predictions[..., self.C+6:self.C+10]
        box2_conf = predictions[..., self.C+5:self.C+6]
        
        class_pred = predictions[..., :self.C]
        
        # --- Target Slices ---
        box_target = target[..., 1:5] # [x, y, w, h]
        class_target = target[..., 5:]
        
        # --- Localization Loss (Box Loss) ---
        # Only for cells that contain an object
        # We use a simplified loss: only penalize the first box
        # A real implementation would check IoU
        
        box_loss = self.mse(
            obj_mask * box1_pred,
            obj_mask * box_target
        )
        
        # --- Confidence Loss ---
        # 1. For cells *with* an object
        obj_conf_loss = self.mse(
            obj_mask * box1_conf,
            obj_mask * 1.0 # Target confidence is 1
        )
        
        # 2. For cells *without* an object
        no_obj_conf_loss_1 = self.mse(
            (1 - obj_mask) * box1_conf,
            (1 - obj_mask) * 0.0 # Target confidence is 0
        )
        no_obj_conf_loss_2 = self.mse(
            (1 - obj_mask) * box2_conf,
            (1 - obj_mask) * 0.0 # Target confidence is 0
        )
        
        no_obj_conf_loss = no_obj_conf_loss_1 + no_obj_conf_loss_2
        
        # --- Classification Loss ---
        # Only for cells that contain an object
        class_loss = self.mse(
            obj_mask * class_pred,
            obj_mask * class_target
        )
        
        # --- Total Loss ---
        loss = (
            self.lambda_coord * box_loss
            + obj_conf_loss
            + self.lambda_noobj * no_obj_conf_loss
            + class_loss
        )
        
        return loss