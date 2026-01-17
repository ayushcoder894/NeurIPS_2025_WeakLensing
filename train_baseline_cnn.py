"""
NeurIPS 2025 Weak Lensing Challenge - Baseline CNN Model
Train a CNN to predict cosmological parameters (Omega_m, S_8) with uncertainties
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)

# Device configuration - GPU ONLY (Training on CPU is too slow)
print("\n" + "="*70)
print("DEVICE CONFIGURATION")
print("="*70)
print(f"PyTorch version: {torch.__version__}")

if not torch.cuda.is_available():
    print("\n❌ ERROR: No GPU detected!")
    print("\n" + "="*70)
    print("This training script requires a CUDA-capable GPU.")
    print("Training 25,000+ images on CPU would take days/weeks.")
    print("="*70)
    print("\nTo set up GPU support:")
    print("  1. Run the setup script: .\\setup_env.ps1")
    print("  2. Or manually install PyTorch with CUDA:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("\nIf you don't have an NVIDIA GPU:")
    print("  - Consider using Google Colab (free GPU)")
    print("  - Use a cloud GPU service (AWS, Azure, GCP)")
    print("  - Reduce training data size for CPU training (not recommended)")
    print("="*70)
    exit(1)

# GPU is available
device = torch.device('cuda')
torch.cuda.manual_seed(42)

# Print GPU information
print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
print(f"✓ CUDA version: {torch.version.cuda}")
gpu_props = torch.cuda.get_device_properties(0)
print(f"✓ GPU memory: {gpu_props.total_memory / 1024**3:.2f} GB")
print(f"✓ Compute capability: {gpu_props.major}.{gpu_props.minor}")
print(f"✓ Using device: {device}")

# Clear GPU cache
torch.cuda.empty_cache()

# ============================================================================
# Configuration
# ============================================================================
DATA_DIR = r'c:\ML\Challenges\NeurIPS_2025'
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
PATIENCE = 7

# ============================================================================
# Dataset Class
# ============================================================================
class ConvergenceMapDataset(Dataset):
    """Memory-efficient dataset that reconstructs images on-the-fly"""
    
    def __init__(self, flat_data, targets, mask):
        """
        Args:
            flat_data: (N, n_pixels) flattened convergence maps
            targets: (N, 2) cosmological parameters
            mask: (H, W) boolean mask for reconstruction
        """
        self.flat_data = flat_data
        self.targets = torch.FloatTensor(targets)
        self.mask = mask
        self.H, self.W = mask.shape
        
    def __len__(self):
        return len(self.flat_data)
    
    def __getitem__(self, idx):
        # Reconstruct 2D image on-the-fly
        image_2d = np.zeros((self.H, self.W), dtype=np.float32)
        image_2d[self.mask > 0] = self.flat_data[idx]
        
        # Convert to tensor with channel dimension
        image = torch.FloatTensor(image_2d).unsqueeze(0)  # (1, H, W)
        target = self.targets[idx]
        
        return image, target

# ============================================================================
# Model Architecture
# ============================================================================
class WeakLensingCNN(nn.Module):
    """CNN for predicting cosmological parameters from weak lensing maps"""
    
    def __init__(self, input_channels=1):
        super(WeakLensingCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)  # [Omega_m, S_8, log_var_Omega_m, log_var_S_8]
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ============================================================================
# Loss Function
# ============================================================================
def kl_divergence_loss(predictions, targets):
    """KL Divergence loss for uncertainty estimation"""
    omega_m_pred = predictions[:, 0]
    s_8_pred = predictions[:, 1]
    log_var_omega_m = predictions[:, 2]
    log_var_s_8 = predictions[:, 3]
    
    omega_m_true = targets[:, 0]
    s_8_true = targets[:, 1]
    
    var_omega_m = torch.exp(log_var_omega_m)
    var_s_8 = torch.exp(log_var_s_8)
    
    loss_omega_m = ((omega_m_pred - omega_m_true) ** 2) / var_omega_m + log_var_omega_m
    loss_s_8 = ((s_8_pred - s_8_true) ** 2) / var_s_8 + log_var_s_8
    
    return torch.mean(loss_omega_m + loss_s_8)

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
def load_and_prepare_data():
    """Load and prepare training data (memory-efficient version)"""
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load data files
    labels = np.load(os.path.join(DATA_DIR, 'label.npy'))
    kappa_train = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_kappa.npy'))
    mask = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_mask.npy'))
    
    print(f"Labels shape: {labels.shape}")
    print(f"Training kappa shape: {kappa_train.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Extract cosmological parameters
    omega_m = labels[:, :, 0]
    s_8 = labels[:, :, 1]
    
    # Flatten data (keep as flattened to save memory)
    n_cosmologies, n_samples_per_cosmo = omega_m.shape
    total_samples = n_cosmologies * n_samples_per_cosmo
    
    X_train_flat = kappa_train.reshape(total_samples, -1)
    y_omega_m = omega_m.flatten()
    y_s_8 = s_8.flatten()
    y_train = np.column_stack([y_omega_m, y_s_8])
    
    print(f"\nTotal training samples: {total_samples}")
    print(f"Flattened data shape: {X_train_flat.shape}")
    print(f"Memory usage: {X_train_flat.nbytes / (1024**3):.2f} GB")
    print(f"Target statistics:")
    print(f"  Omega_m: mean={y_omega_m.mean():.4f}, std={y_omega_m.std():.4f}")
    print(f"  S_8: mean={y_s_8.mean():.4f}, std={y_s_8.std():.4f}")
    
    print("\n✓ Data loaded (images will be reconstructed on-the-fly during training)")
    
    return X_train_flat, y_train, mask

# ============================================================================
# Training Functions
# ============================================================================
def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = kl_divergence_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = kl_divergence_loss(output, target)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

# ============================================================================
# Main Training Loop
# ============================================================================
def main():
    print("="*70)
    print("NeurIPS 2025 Weak Lensing Challenge - Baseline CNN Training")
    print("="*70)
    
    # Load data
    X_train_flat, y_train, mask = load_and_prepare_data()
    
    # Split into train/validation
    print("\n" + "="*70)
    print("CREATING TRAIN/VAL SPLIT")
    print("="*70)
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_flat, y_train, test_size=0.15, random_state=42
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    # Create datasets and loaders (now using on-the-fly reconstruction)
    train_dataset = ConvergenceMapDataset(X_train, y_train_split, mask)
    val_dataset = ConvergenceMapDataset(X_val, y_val, mask)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    print("\n" + "="*70)
    print("CREATING MODEL")
    print("="*70)
    model = WeakLensingCNN(input_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    print("\n" + "="*70)
    print(f"TRAINING FOR {NUM_EPOCHS} EPOCHS")
    print("="*70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, os.path.join(DATA_DIR, 'best_model.pth'))
            print(f"  → Best model saved! (Val Loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Evaluate on validation set
    print("\n" + "="*70)
    print("EVALUATING ON VALIDATION SET")
    print("="*70)
    
    checkpoint = torch.load(os.path.join(DATA_DIR, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)
            output = model(data)
            all_predictions.append(output[:, :2].cpu().numpy())
            all_targets.append(target.numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    omega_m_rmse = np.sqrt(mean_squared_error(all_targets[:, 0], all_predictions[:, 0]))
    s_8_rmse = np.sqrt(mean_squared_error(all_targets[:, 1], all_predictions[:, 1]))
    omega_m_r2 = r2_score(all_targets[:, 0], all_predictions[:, 0])
    s_8_r2 = r2_score(all_targets[:, 1], all_predictions[:, 1])
    
    print(f"\nValidation Performance:")
    print(f"  Ω_m RMSE: {omega_m_rmse:.6f}")
    print(f"  S_8 RMSE: {s_8_rmse:.6f}")
    print(f"  Ω_m R²: {omega_m_r2:.4f}")
    print(f"  S_8 R²: {s_8_r2:.4f}")
    
    # Generate test predictions
    print("\n" + "="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70)
    
    kappa_test = np.load(os.path.join(DATA_DIR, 'WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'))
    print(f"Test data shape: {kappa_test.shape}")
    
    # Prepare test data (keep flattened)
    if len(kappa_test.shape) == 3:
        n_test_cosmo, n_test_samples, n_pixels = kappa_test.shape
        total_test_samples = n_test_cosmo * n_test_samples
        X_test_flat = kappa_test.reshape(total_test_samples, -1)
    else:
        X_test_flat = kappa_test
        total_test_samples = X_test_flat.shape[0]
    
    print(f"Total test samples: {total_test_samples}")
    
    # Create test dataset (will reconstruct on-the-fly)
    test_dataset = ConvergenceMapDataset(X_test_flat, np.zeros((total_test_samples, 2)), mask)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Generate predictions
    print("Generating predictions...")
    test_predictions = []
    test_uncertainties = []
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            
            predictions = output[:, :2].cpu().numpy()
            log_vars = output[:, 2:].cpu().numpy()
            uncertainties = np.sqrt(np.exp(log_vars))
            
            test_predictions.append(predictions)
            test_uncertainties.append(uncertainties)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE}/{total_test_samples}")
    
    test_predictions = np.vstack(test_predictions)
    test_uncertainties = np.vstack(test_uncertainties)
    
    # Save submission
    submission = np.column_stack([
        test_predictions[:, 0],  # Omega_m
        test_predictions[:, 1],  # S_8
        test_uncertainties[:, 0],  # sigma_Omega_m
        test_uncertainties[:, 1]   # sigma_S_8
    ])
    
    submission_file = os.path.join(DATA_DIR, 'submission_baseline_cnn.npy')
    np.save(submission_file, submission)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Submission saved to: {submission_file}")
    print(f"Submission shape: {submission.shape}")
    print(f"  Columns: [Omega_m, S_8, sigma_Omega_m, sigma_S_8]")
    print("\nTest predictions summary:")
    print(f"  Ω_m: {test_predictions[:, 0].mean():.4f} ± {test_predictions[:, 0].std():.4f}")
    print(f"  S_8: {test_predictions[:, 1].mean():.4f} ± {test_predictions[:, 1].std():.4f}")
    print("="*70)

if __name__ == "__main__":
    main()
