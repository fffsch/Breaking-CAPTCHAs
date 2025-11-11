import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFusionFilterNetwork(nn.Module):
    """
    Adaptive Fusion Filter Networks (AFFN) for noise reduction
    """
    def __init__(self, max_filter_units=2):
        super(AdaptiveFusionFilterNetwork, self).__init__()
        self.max_filter_units = max_filter_units
        
        # Create encoder-decoder pairs for each filter unit
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Adaptive fusion factor (trainable)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        for i in range(max_filter_units):
            # Encoder: Conv -> BN -> ReLU
            encoder = nn.Sequential(
                nn.Conv2d(1 if i == 0 else 4*(i), 4*(i+1), kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(4*(i+1)),
                nn.ReLU()
            )
            
            # Decoder: Conv -> BN -> ReLU
            decoder = nn.Sequential(
                nn.Conv2d(4*(i+1), 1 if i == 0 else 4*(i), kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(1 if i == 0 else 4*(i)),
                nn.ReLU()
            )
            
            self.encoders.append(encoder)
            self.decoders.append(decoder)
    
    def forward(self, x):
        # Apply nested encoder-decoder with adaptive fusion
        encoded_features = []
        current = x
        
        # Encoding path
        for encoder in self.encoders:
            current = encoder(current)
            encoded_features.append(current)
        
        # Decoding path with fusion
        for i in range(len(self.decoders) - 1, -1, -1):
            current = self.decoders[i](current)
            if i > 0:
                # Apply adaptive fusion factor
                current = self.alpha * current + (1 - self.alpha) * encoded_features[i-1]
        
        return current


class CRNNModule(nn.Module):
    """
    CRNN Module: Conv layers + LSTM for sequence modeling
    """
    def __init__(self, input_channels=64, hidden_size=512, num_classes=26, dropout_rate=0.3):
        super(CRNNModule, self).__init__()
        
        # Conv4: 64->128 channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv5: 128->256 channels
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # LSTM layer - single layer as per paper
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output fully connected layers for 4 character positions
        self.fc_pos = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _ in range(4)
        ])
        
    def forward(self, x):
        # Apply convolutional layers
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Flatten for LSTM: (batch, channels*height*width)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Split into 4 parts for 4 character positions
        seq_len = 4
        feature_size = x.size(1) // seq_len
        x = x.view(batch_size, seq_len, feature_size)
        
        x = self.dropout1(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        x = self.dropout2(lstm_out)
        
        # Apply FC layers for each position
        outputs = []
        for i in range(4):
            out = self.fc_pos[i](x[:, i, :])  # (batch, num_classes)
            outputs.append(out)
        
        return outputs


class AdaptiveCAPTCHA(nn.Module):
    """
    Complete Adaptive CAPTCHA model with AFFN, CNN, and CRNN
    Input: 64x192 grayscale images
    Output: 4 character predictions (26 uppercase letters each)
    """
    def __init__(self, num_classes=26, max_filter_units=2, use_affn=True, 
                 residual_connection='T1'):
        super(AdaptiveCAPTCHA, self).__init__()
        
        self.use_affn = use_affn
        self.residual_connection = residual_connection
        
        # AFFN module (optional)
        if self.use_affn:
            self.affn = AdaptiveFusionFilterNetwork(max_filter_units=max_filter_units)
        
        # Conv1: 1->32 channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv2: 32->48 channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv3: 48->64 channels
        self.conv3 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # CRNN Module
        self.crnn = CRNNModule(input_channels=64, hidden_size=768, 
                              num_classes=num_classes, dropout_rate=0.3)
        
        # Store intermediate features for residual connections
        self.features = {}
        
    def forward(self, x):
        """
        x: (batch_size, 1, 64, 192) - grayscale CAPTCHA images
        returns: list of 4 tensors, each (batch_size, num_classes)
        """
        # Apply AFFN filtering if enabled (Pos T)
        if self.use_affn:
            x = self.affn(x)
            self.features['pos_T'] = x
        
        # Conv1 (Pos 0)
        x = self.conv1(x)
        self.features['pos_0'] = x
        
        # Conv2 (Pos 1)
        x = self.conv2(x)
        self.features['pos_1'] = x
        
        # Apply residual connection T1 if specified
        if self.residual_connection == 'T1' and self.use_affn:
            # Adjust dimensions if needed
            res = F.adaptive_avg_pool2d(self.features['pos_T'], x.size()[2:])
            if res.size(1) != x.size(1):
                # Match channel dimensions
                res = F.interpolate(res, size=x.size()[2:], mode='bilinear', align_corners=False)
                # Use 1x1 conv to match channels
                if not hasattr(self, 'res_proj'):
                    self.res_proj = nn.Conv2d(1, 48, kernel_size=1).to(x.device)
                res = self.res_proj(res)
            x = x + res
        
        # Conv3 (Pos 2)
        x = self.conv3(x)
        self.features['pos_2'] = x
        
        # CRNN Module (Pos 3)
        outputs = self.crnn(x)
        
        return outputs
    
    def predict(self, x):
        """
        Returns predicted characters as indices
        """
        outputs = self.forward(x)
        predictions = []
        for out in outputs:
            pred = torch.argmax(out, dim=1)
            predictions.append(pred)
        return torch.stack(predictions, dim=1)  # (batch_size, 4)


# Training utilities
class CAPTCHALoss(nn.Module):
    """
    Binary Cross-Entropy loss for multi-position character classification
    """
    def __init__(self):
        super(CAPTCHALoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        """
        predictions: list of 4 tensors, each (batch_size, num_classes)
        targets: (batch_size, 4) - character indices for each position
        """
        total_loss = 0
        for i, pred in enumerate(predictions):
            total_loss += self.criterion(pred, targets[:, i])
        return total_loss / len(predictions)


def calculate_asr(predictions, targets):
    """
    Calculate Attack Success Rate (character-level accuracy)
    predictions: (batch_size, 4) - predicted character indices
    targets: (batch_size, 4) - ground truth character indices
    """
    correct = (predictions == targets).float()
    asr = correct.mean().item() * 100
    return asr


def calculate_aasr(predictions, targets):
    """
    Calculate Average Attack Success Rate (full CAPTCHA accuracy)
    Only counts as correct if all 4 characters match
    """
    correct_captchas = (predictions == targets).all(dim=1).float()
    aasr = correct_captchas.mean().item() * 100
    return aasr


# Example usage and training loop
def train_model(model, train_loader, val_loader, num_epochs=130, learning_rate=0.0001):
    """
    Training function for Adaptive CAPTCHA model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = CAPTCHALoss()
    
    best_aasr = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)  # (batch, 1, 64, 192)
            labels = labels.to(device)  # (batch, 4)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.stack([torch.argmax(out, dim=1) for out in outputs], dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.numel()
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                predictions = torch.stack([torch.argmax(out, dim=1) for out in outputs], dim=1)
                
                val_predictions.append(predictions.cpu())
                val_targets.append(labels.cpu())
        
        val_predictions = torch.cat(val_predictions, dim=0)
        val_targets = torch.cat(val_targets, dim=0)
        
        # Calculate metrics
        asr = calculate_asr(val_predictions, val_targets)
        aasr = calculate_aasr(val_predictions, val_targets)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {train_loss/len(train_loader):.4f}, '
              f'Train ASR: {100*train_correct/train_total:.2f}%, '
              f'Val ASR: {asr:.2f}%, '
              f'Val AASR: {aasr:.2f}%')
        
        # Save best model
        if aasr > best_aasr:
            best_aasr = aasr
            torch.save(model.state_dict(), 'adaptive_captcha_best.pth')
    
    return model


# Data preprocessing example
class CAPTCHADataset(torch.utils.data.Dataset):
    """
    Dataset class for CAPTCHA images
    Expects images of size 64x192 pixels
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        image_paths: list of image file paths
        labels: list of 4-character strings (e.g., 'ABCD')
        transform: optional transforms
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
        # Character to index mapping (A-Z -> 0-25)
        self.char_to_idx = {chr(65+i): i for i in range(26)}
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np
        
        # Load image
        img = Image.open(self.image_paths[idx])
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Convert to numpy array and normalize
        img = np.array(img, dtype=np.float32) / 255.0
        
        # Add channel dimension
        img = img[np.newaxis, :, :]  # (1, 64, 192)
        
        # Convert to tensor
        img = torch.from_numpy(img)
        
        if self.transform:
            img = self.transform(img)
        
        # Convert label string to indices
        label_str = self.labels[idx]
        label_indices = torch.tensor([self.char_to_idx[c] for c in label_str], 
                                    dtype=torch.long)
        
        return img, label_indices


# Model instantiation
def create_adaptive_captcha_model(num_classes=26, max_filter_units=2, 
                                 use_affn=True, residual='T1'):
    """
    Factory function to create Adaptive CAPTCHA model
    
    Args:
        num_classes: Number of character classes (26 for A-Z)
        max_filter_units: Number of filter units in AFFN
        use_affn: Whether to use AFFN filtering
        residual: Residual connection type ('T1', 'T0', None, etc.)
    """
    model = AdaptiveCAPTCHA(
        num_classes=num_classes,
        max_filter_units=max_filter_units,
        use_affn=use_affn,
        residual_connection=residual
    )
    
    return model


if __name__ == "__main__":
    # Example: Create model
    model = create_adaptive_captcha_model(
        num_classes=26,
        max_filter_units=2,
        use_affn=True,
        residual='T1'
    )
    
    # Print model architecture
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 64, 192)
    
    with torch.no_grad():
        outputs = model(test_input)
        predictions = model.predict(test_input)
    
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Number of output heads: {len(outputs)}")
    print(f"Each output shape: {outputs[0].shape}")
    print(f"Predictions shape: {predictions.shape}")
