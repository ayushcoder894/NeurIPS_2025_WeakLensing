# Setup script for NeurIPS 2025 Weak Lensing Challenge
# Creates venv and installs PyTorch with CUDA support

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*69) -ForegroundColor Cyan
Write-Host "NeurIPS 2025 Weak Lensing Challenge - Environment Setup" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("="*69) -ForegroundColor Cyan

# Check if NVIDIA GPU is available
Write-Host "`nChecking for NVIDIA GPU..." -ForegroundColor Yellow
try {
    $gpu = nvidia-smi --query-gpu=name --format=csv,noheader 2>$null
    if ($gpu) {
        Write-Host "✓ NVIDIA GPU detected: $gpu" -ForegroundColor Green
    } else {
        Write-Host "✗ No NVIDIA GPU detected!" -ForegroundColor Red
        Write-Host "This project requires an NVIDIA GPU with CUDA support." -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "✗ nvidia-smi not found. NVIDIA drivers may not be installed." -ForegroundColor Red
    exit 1
}

# Create virtual environment
$venvPath = ".\venv"
if (Test-Path $venvPath) {
    Write-Host "`n⚠️  Virtual environment already exists at $venvPath" -ForegroundColor Yellow
    $response = Read-Host "Delete and recreate? (y/n)"
    if ($response -eq 'y') {
        Write-Host "Removing existing venv..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $venvPath
    } else {
        Write-Host "Using existing venv..." -ForegroundColor Yellow
        & "$venvPath\Scripts\Activate.ps1"
        exit 0
    }
}

Write-Host "`nCreating virtual environment..." -ForegroundColor Yellow
python -m venv venv

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

Write-Host "`nUpgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

Write-Host "`nInstalling PyTorch with CUDA 11.8 support..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Write-Host "`nInstalling other dependencies..." -ForegroundColor Yellow
pip install numpy matplotlib seaborn scikit-learn pandas

Write-Host "`nVerifying PyTorch CUDA installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host "`n" + ("="*70) -ForegroundColor Cyan
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ("="*70) -ForegroundColor Cyan
Write-Host "`nTo activate the environment in the future, run:" -ForegroundColor Yellow
Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "`nTo train the model, run:" -ForegroundColor Yellow
Write-Host "  python train_baseline_cnn.py" -ForegroundColor Cyan
Write-Host ""
