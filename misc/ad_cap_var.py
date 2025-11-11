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


class VariableLengthCRNN(nn.Module):
    """
    CRNN Module with CTC for variable-length output
    Uses bidirectional LSTM for better context modeling
    """
    def __init__(self, input_channels=64, hidden_size=256, num_classes=37, 
                 num_lstm_layers=2, dropout_rate=0.3):
        super(VariableLengthCRNN, self).__init__()
        
        # Conv4: 64->128 channels
        self.conv4 = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv5: 128->256 channels
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Additional conv for better feature extraction
        self.conv_bridge = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Calculate feature size after convolutions
        # After conv4: H/2, W/2
        # After conv5: H/4, W/4
        # For 64x192 input: 8x24 after all pooling
        # Feature dimension per timestep
        self.feature_size = 256  # channels after conv5
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Output projection layer
        # +1 for CTC blank token
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)
        
    def forward(self, x):
        """
        x: (batch, channels, height, width)
        returns: (batch, seq_len, num_classes+1) - logits for CTC
        """
        # Apply convolutional layers
        x = self.conv4(x)
        x = self.conv_bridge(x)
        x = self.conv5(x)
        
        # Extract sequence features
        # x shape: (batch, channels, height, width)
        batch_size, channels, height, width = x.size()
        
        # Collapse height dimension by averaging (or max pooling)
        x = x.permute(0, 3, 1, 2)  # (batch, width, channels, height)
        x = x.contiguous().view(batch_size, width, channels * height)
        
        # Alternative: use adaptive pooling to get fixed sequence length
        # x = F.adaptive_avg_pool2d(x, (1, width))
        # x = x.squeeze(2).permute(0, 2, 1)  # (batch, width, channels)
        
        x = self.dropout1(x)
        
        # LSTM
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        x = self.dropout2(lstm_out)
        
        # Project to character classes + blank
        output = self.fc(x)  # (batch, seq_len, num_classes+1)
        
        # Apply log softmax for CTC loss
        output = F.log_softmax(output, dim=2)
        
        return output


class AdaptiveCAPTCHAVariableLength(nn.Module):
    """
    Adaptive CAPTCHA model for variable-length sequences
    Supports both uppercase letters (A-Z) and digits (0-9)
    Input: Variable height grayscale images (standardized to 64 height)
    Output: Variable-length character predictions (A-Z, 0-9)
    """
    def __init__(self, num_classes=36, max_filter_units=2, use_affn=True,
                 hidden_size=256, num_lstm_layers=2, residual_connection='T1'):
        """
        num_classes: 36 for A-Z (26) + 0-9 (10)
        """
        super(AdaptiveCAPTCHAVariableLength, self).__init__()
        
        self.use_affn = use_affn
        self.residual_connection = residual_connection
        self.num_classes = num_classes
        
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
        
        # Variable-length CRNN Module with CTC
        self.crnn = VariableLengthCRNN(
            input_channels=64,
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_lstm_layers=num_lstm_layers,
            dropout_rate=0.3
        )
        
        # Character mappings
        self.idx_to_char = self._create_char_mapping()
        self.char_to_idx = {v: k for k, v in self.idx_to_char.items()}
        
        # Store intermediate features for residual connections
        self.features = {}
        
    def _create_char_mapping(self):
        """
        Create mapping from indices to characters
        0-25: A-Z
        26-35: 0-9
        36: CTC blank token
        """
        chars = {}
        # Letters A-Z
        for i in range(26):
            chars[i] = chr(65 + i)  # A=65 in ASCII
        # Digits 0-9
        for i in range(10):
            chars[26 + i] = str(i)
        # Blank token for CTC
        chars[self.num_classes] = '<blank>'
        return chars
        
    def forward(self, x):
        """
        x: (batch_size, 1, height, width) - grayscale CAPTCHA images
        returns: (batch_size, seq_len, num_classes+1) - log probabilities for CTC
        """
        # Apply AFFN filtering if enabled
        if self.use_affn:
            x = self.affn(x)
            self.features['pos_T'] = x
        
        # Conv1
        x = self.conv1(x)
        self.features['pos_0'] = x
        
        # Conv2
        x = self.conv2(x)
        self.features['pos_1'] = x
        
        # Apply residual connection T1 if specified
        if self.residual_connection == 'T1' and self.use_affn:
            res = F.adaptive_avg_pool2d(self.features['pos_T'], x.size()[2:])
            if res.size(1) != x.size(1):
                if not hasattr(self, 'res_proj'):
                    self.res_proj = nn.Conv2d(1, 48, kernel_size=1).to(x.device)
                res = self.res_proj(res)
            x = x + res
        
        # Conv3
        x = self.conv3(x)
        self.features['pos_2'] = x
        
        # CRNN Module with CTC output
        log_probs = self.crnn(x)
        
        return log_probs
    
    def decode_predictions(self, log_probs, method='greedy'):
        """
        Decode CTC predictions to text
        
        Args:
            log_probs: (batch, seq_len, num_classes+1)
            method: 'greedy' or 'beam_search'
        
        Returns:
            list of decoded strings
        """
        if method == 'greedy':
            return self._greedy_decode(log_probs)
        elif method == 'beam_search':
            return self._beam_search_decode(log_probs)
        else:
            raise ValueError(f"Unknown decoding method: {method}")
    
    def _greedy_decode(self, log_probs):
        """
        Simple greedy decoding for CTC
        """
        batch_size = log_probs.size(0)
        predictions = []
        
        # Get most likely class at each timestep
        _, indices = torch.max(log_probs, dim=2)  # (batch, seq_len)
        
        for batch_idx in range(batch_size):
            sequence = indices[batch_idx].cpu().numpy()
            decoded = []
            prev_char = None
            
            for idx in sequence:
                # Skip blanks and repeated characters
                if idx != self.num_classes and idx != prev_char:
                    decoded.append(self.idx_to_char[int(idx)])
                prev_char = idx
            
            predictions.append(''.join(decoded))
        
        return predictions
    
    def _beam_search_decode(self, log_probs, beam_width=5):
        """
        Beam search decoding for better accuracy
        """
        try:
            from torch.nn.functional import log_softmax
            # For production use, consider using libraries like:
            # - ctcdecode (https://github.com/parlance/ctcdecode)
            # - pytorch's built-in beam search (torch.nn.functional.ctc_beam_search_decoder)
            
            # Simplified beam search implementation
            batch_size = log_probs.size(0)
            predictions = []
            
            for batch_idx in range(batch_size):
                probs = log_probs[batch_idx]  # (seq_len, num_classes+1)
                
                # Initialize beam with empty sequence
                beams = [('', 0.0)]  # (sequence, log_prob)
                
                for timestep in range(probs.size(0)):
                    new_beams = []
                    
                    for sequence, seq_prob in beams:
                        for char_idx in range(probs.size(1)):
                            char_prob = probs[timestep, char_idx].item()
                            
                            if char_idx == self.num_classes:
                                # Blank token
                                new_seq = sequence
                            else:
                                char = self.idx_to_char[char_idx]
                                # Only add if different from last character
                                if len(sequence) == 0 or sequence[-1] != char:
                                    new_seq = sequence + char
                                else:
                                    new_seq = sequence
                            
                            new_prob = seq_prob + char_prob
                            new_beams.append((new_seq, new_prob))
                    
                    # Keep top beam_width beams
                    new_beams.sort(key=lambda x: x[1], reverse=True)
                    beams = new_beams[:beam_width]
                
                # Return most likely sequence
                best_sequence = beams[0][0]
                predictions.append(best_sequence)
            
            return predictions
            
        except Exception as e:
            print(f"Beam search failed, falling back to greedy: {e}")
            return self._greedy_decode(log_probs)


# Training utilities
def train_variable_length_model(model, train_loader, val_loader, num_epochs=130, 
                                learning_rate=0.0001, device=None):
    """
    Training function with CTC loss for variable-length sequences
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # CTC Loss
    ctc_loss = nn.CTCLoss(blank=model.num_classes, reduction='mean', zero_infinity=True)
    
    best_acc = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (images, labels, label_lengths) in enumerate(train_loader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            log_probs = model(images)  # (batch, seq_len, num_classes+1)
            
            # CTC requires: (seq_len, batch, num_classes)
            log_probs = log_probs.permute(1, 0, 2)
            
            # Get input lengths (sequence length for each batch item)
            input_lengths = torch.full(
                size=(log_probs.size(1),), 
                fill_value=log_probs.size(0), 
                dtype=torch.long
            )
            
            # Calculate CTC loss
            loss = ctc_loss(log_probs, labels, input_lengths, label_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at batch {batch_idx}, skipping...")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels_list, _ in val_loader:
                images = images.to(device)
                
                log_probs = model(images)
                predictions = model.decode_predictions(log_probs, method='greedy')
                
                # Compare with ground truth
                for pred, label in zip(predictions, labels_list):
                    if pred == label:
                        correct += 1
                    total += 1
        
        val_acc = 100 * correct / total if total > 0 else 0
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Accuracy: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'adaptive_captcha_variable_best.pth')
    
    return model


# Dataset class for variable-length CAPTCHAs
class VariableLengthCAPTCHADataset(torch.utils.data.Dataset):
    """
    Dataset for variable-length CAPTCHA images with alphanumeric characters
    """
    def __init__(self, image_paths, labels, transform=None, target_height=64):
        """
        image_paths: list of image file paths
        labels: list of variable-length strings (e.g., 'AB12', 'XYZ789', etc.)
        transform: optional transforms
        target_height: standardized height for images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_height = target_height
        
        # Character to index mapping
        # A-Z: 0-25, 0-9: 26-35
        self.char_to_idx = {}
        for i in range(26):
            self.char_to_idx[chr(65 + i)] = i  # A-Z
        for i in range(10):
            self.char_to_idx[str(i)] = 26 + i  # 0-9
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np
        
        # Load image
        img = Image.open(self.image_paths[idx])
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize to target height while maintaining aspect ratio
        width, height = img.size
        new_width = int(width * self.target_height / height)
        img = img.resize((new_width, self.target_height), Image.LANCZOS)
        
        # Convert to numpy and normalize
        img = np.array(img, dtype=np.float32) / 255.0
        img = img[np.newaxis, :, :]  # Add channel dimension
        
        # Convert to tensor
        img = torch.from_numpy(img)
        
        if self.transform:
            img = self.transform(img)
        
        # Convert label string to indices
        label_str = self.labels[idx].upper()  # Ensure uppercase
        label_indices = [self.char_to_idx[c] for c in label_str]
        label_length = len(label_indices)
        
        return img, label_indices, label_length


def collate_fn_variable_length(batch):
    """
    Custom collate function for variable-length sequences
    Handles batching of variable-width images and variable-length labels
    """
    images, label_lists, label_lengths = zip(*batch)
    
    # Find max width in batch
    max_width = max(img.size(2) for img in images)
    max_height = images[0].size(1)
    
    # Pad images to same width
    padded_images = []
    for img in images:
        pad_width = max_width - img.size(2)
        if pad_width > 0:
            img = F.pad(img, (0, pad_width, 0, 0), value=0)
        padded_images.append(img)
    
    # Stack images
    images_tensor = torch.stack(padded_images)
    
    # Concatenate labels
    labels_tensor = torch.tensor([idx for label in label_lists for idx in label], 
                                dtype=torch.long)
    label_lengths_tensor = torch.tensor(label_lengths, dtype=torch.long)
    
    # Also return original label strings for validation
    label_strings = [''.join([chr(65+idx) if idx < 26 else str(idx-26) 
                             for idx in label]) for label in label_lists]
    
    return images_tensor, labels_tensor, label_lengths_tensor, label_strings


# Model instantiation
def create_variable_length_model(num_classes=36, max_filter_units=2, 
                                use_affn=True, hidden_size=256, 
                                num_lstm_layers=2):
    """
    Create Adaptive CAPTCHA model for variable-length sequences
    
    Args:
        num_classes: 36 for A-Z (26) + 0-9 (10)
        max_filter_units: Number of AFFN filter units
        use_affn: Whether to use AFFN
        hidden_size: LSTM hidden size
        num_lstm_layers: Number of LSTM layers
    """
    model = AdaptiveCAPTCHAVariableLength(
        num_classes=num_classes,
        max_filter_units=max_filter_units,
        use_affn=use_affn,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        residual_connection='T1'
    )
    
    return model


if __name__ == "__main__":
    # Example: Create variable-length model
    model = create_variable_length_model(
        num_classes=36,  # A-Z (26) + 0-9 (10)
        max_filter_units=2,
        use_affn=True,
        hidden_size=256,
        num_lstm_layers=2
    )
    
    print("Variable-Length Adaptive CAPTCHA Model")
    print("=" * 50)
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test with variable-width inputs
    print("\nTesting variable-length inputs:")
    test_inputs = [
        torch.randn(2, 1, 64, 120),  # Shorter CAPTCHA
        torch.randn(2, 1, 64, 192),  # Medium CAPTCHA
        torch.randn(2, 1, 64, 256),  # Longer CAPTCHA
    ]
    
    model.eval()
    with torch.no_grad():
        for i, test_input in enumerate(test_inputs):
            log_probs = model(test_input)
            predictions = model.decode_predictions(log_probs)
            
            print(f"\nTest {i+1}:")
            print(f"  Input shape: {test_input.shape}")
            print(f"  Output shape: {log_probs.shape}")
            print(f"  Decoded predictions: {predictions}")
    
    print("\nCharacter mapping:")
    print("  A-Z: indices 0-25")
    print("  0-9: indices 26-35")
    print("  <blank>: index 36 (CTC blank token)")
