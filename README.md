# NeurIPS 2025 Weak Lensing Challenge

A CNN-based solution for the FAIR Universe Weak Lensing ML Uncertainty Challenge to predict cosmological parameters (Ω_m, S_8) from weak lensing convergence maps.

## 🎯 Challenge Overview

Predict cosmological parameters from weak gravitational lensing convergence maps:
- **Ω_m**: Fraction of matter energy density in the universe
- **S_8**: Amplitude of matter fluctuations

The model outputs point estimates and uncertainties: `[Ω_m, S_8, σ_Ω_m, σ_S_8]`

## 📁 Project Structure

```
NeurIPS_2025/
├── train_baseline_cnn.py          # Baseline CNN training (RTX 4060/consumer GPU)
├── train_baseline_cnn_a100.py     # Optimized for NVIDIA A100 40GB
├── baseline_cnn_model.ipynb       # Interactive notebook version
├── dataset_analysis.ipynb         # Data exploration notebook
├── setup_env.ps1                  # Environment setup script (Windows)
├── challenge_info.txt             # Challenge description
└── README.md                      # This file
```

## Requirements

- **NVIDIA GPU with CUDA support** (Required - training won't run on CPU)
- Python 3.8+
- NVIDIA drivers installed
- 8GB+ GPU memory recommended (40GB for A100 version)

## 🚀 Quick Start

### 1. Setup Virtual Environment with GPU Support

```powershell
# Windows (PowerShell) - Automated setup
.\setup_env.ps1

# Or manually:
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib seaborn scikit-learn pandas
```

### 2. Download Data

Download from [Codabench Competition](https://www.codabench.org/) and place in project directory:
- `WIDE12H_bin2_2arcmin_kappa.npy` - Training convergence maps
- `WIDE12H_bin2_2arcmin_kappa_noisy_test.npy` - Test data  
- `WIDE12H_bin2_2arcmin_mask.npy` - Mask file
- `label.npy` - Training labels

### 3. Train the Model

```powershell
# For RTX 4060 or similar consumer GPU (8GB VRAM)
.\venv\Scripts\python.exe train_baseline_cnn.py

# For A100 40GB or datacenter GPU
.\venv\Scripts\python.exe train_baseline_cnn_a100.py
```

## 🧠 Model Architecture

### Baseline CNN (Consumer GPU)
```
Conv2D(1→32) → BatchNorm → ReLU → MaxPool
Conv2D(32→64) → BatchNorm → ReLU → MaxPool  
Conv2D(64→128) → BatchNorm → ReLU → MaxPool
Conv2D(128→256) → BatchNorm → ReLU → AdaptiveAvgPool
Linear(4096→512) → ReLU → Dropout(0.3)
Linear(512→256) → ReLU → Dropout(0.2)
Linear(256→4)  # [Ω_m, S_8, log_var_Ω_m, log_var_S_8]
```
- ~2.6M parameters | Batch size: 32 | Training: 30-60 min

### A100 Optimized CNN
- 4 conv blocks: 64→128→256→512 channels (double conv per block)
- ~11M parameters | Batch size: 128 | Training: 15-25 min
- Mixed precision (FP16) | Data augmentation

## 📊 Training Details

- **Loss**: KL Divergence for uncertainty estimation
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine Annealing / ReduceLROnPlateau
- **Early Stopping**: Patience 7-12 epochs
- **Data**: 25,856 samples (101 cosmologies × 256 each)

## � Results

### Model Performance (A100 Optimized)

| Metric | Ω_m | S_8 |
|--------|-----|-----|
| **Mean Prediction** | 0.2498 | 0.6572 |
| **Std Deviation** | 0.0129 | 0.0394 |
| **Avg Uncertainty (σ)** | 0.1829 | 0.2029 |

### Test Set Predictions Summary
- **Total samples**: 4,000
- **Output format**: `[Ω_m, S_8, σ_Ω_m, σ_S_8]`

### Training Configuration
| Setting | Baseline (RTX 4060) | A100 Optimized |
|---------|---------------------|----------------|
| Batch Size | 32 | 128 |
| Parameters | ~2.6M | ~11M |
| Training Time | 30-60 min | 15-25 min |
| Mixed Precision | No | Yes (FP16) |
| Data Augmentation | No | Yes (flips) |

## �📝 Generate Submission

```python
import numpy as np, json, zipfile, os

data = np.load('submission_baseline_cnn_a100.npy')
payload = {
    "means": data[:, :2].tolist(),      # [Ω_m, S_8] for 4000 samples
    "errorbars": data[:, 2:].tolist()   # [σ_Ω_m, σ_S_8] for 4000 samples
}

os.makedirs('submissions', exist_ok=True)
with open('result.json', 'w') as f:
    json.dump(payload, f)
with zipfile.ZipFile('submission.zip', 'w') as zf:
    zf.write('result.json', arcname='result.json')
```

## 📚 References

- [FAIR Universe Cosmology Challenge](https://github.com/FAIR-Universe/Cosmology_Challenge)
- [Codabench Competition](https://www.codabench.org/)

## Troubleshooting

### "No GPU detected" error

1. Check if you have an NVIDIA GPU:
   ```powershell
   nvidia-smi
   ```

2. Verify PyTorch sees the GPU:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. If False, reinstall PyTorch with CUDA support:
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Out of Memory Error

If you get CUDA out of memory errors:
- Reduce `BATCH_SIZE` in `train_baseline_cnn.py` (try 16 or 8)
- Close other GPU-intensive applications

### Slow Training

Training should take 30-60 minutes on a modern GPU. If it's much slower:
- Verify you're using GPU (check console output)
- Check GPU utilization with `nvidia-smi`
- Ensure no other processes are using the GPU

## Files

- `train_baseline_cnn.py` - Main training script
- `setup_env.ps1` - Environment setup script
- `dataset_analysis.ipynb` - Data exploration notebook
- `baseline_cnn_model.ipynb` - Interactive model notebook
- `challenge_info.txt` - Challenge description

## Next Steps for Improvement

1. Data augmentation (rotations, flips)
2. Deeper architecture (ResNet, EfficientNet)
3. Ensemble multiple models
4. Add attention mechanisms
5. Incorporate power spectrum features
6. Test-time augmentation
7. Hyperparameter tuning
