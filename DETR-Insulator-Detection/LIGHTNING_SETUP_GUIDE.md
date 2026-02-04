# Lightning.ai Setup Guide for HITL-proof/DETR-Insulator-Detection

## 🎯 Quick Start Commands for Lightning.ai Studio

### Step 1: Install Dependencies

```bash
# Install PyTorch with CUDA 12.1 (for L4 GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers pillow numpy pycocotools omegaconf tqdm matplotlib scikit-learn roboflow
```

### Step 2: Download Insulators Dataset using Roboflow CLI

```bash
# Set your API key
export ROBOFLOW_API_KEY='your_api_key_here'

# Install roboflow if not already installed
pip install roboflow

# Download using Python API
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='$ROBOFLOW_API_KEY')
project = rf.workspace('sofia-valdivieso-von-teuber').project('insulators-wo6lb')
dataset = project.version(3).download('coco', location='data/')
"
```

**Alternative: One-liner Download**

```bash
pip install roboflow && python -c "from roboflow import Roboflow; rf = Roboflow(api_key='YOUR_KEY'); rf.workspace('sofia-valdivieso-von-teuber').project('insulators-wo6lb').version(3).download('coco', location='data/insulators')"
```

### Step 3: Verify Download

```bash
ls -lh data/insulators/
# Should show: train/, valid/, test/ folders

# Count images
echo "Train images: $(ls data/insulators/train/*.jpg 2>/dev/null | wc -l)"
echo "Valid images: $(ls data/insulators/valid/*.jpg 2>/dev/null | wc -l)"
echo "Test images: $(ls data/insulators/test/*.jpg 2>/dev/null | wc -l)"
```

---

## 📋 Code Review & Fixes

### Issues Found & Fixed

#### 1. **requirements.txt** - PyTorch Version Issue

**Problem**: `torch==2.8.0+cu128` doesn't exist yet (too new)

**Fix**: Use stable versions
```txt
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
Pillow>=10.0.0
numpy>=1.24.0
pycocotools>=2.0.6
omegaconf>=2.3.0
tqdm>=4.65.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
roboflow>=1.1.0
```

#### 2. **dataset.py** - Missing Transform Implementation

**Problem**: `make_detr_transforms()` uses `T.RandomResize()` which doesn't exist in standard torchvision

**Fix**: Implement proper transforms

```python
import torchvision.transforms.functional as F

class RandomResize:
    """Custom random resize for DETR."""
    def __init__(self, sizes, max_size=None):
        self.sizes = sizes
        self.max_size = max_size
    
    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        img = F.resize(img, size, max_size=self.max_size)
        
        if target is not None:
            # Update target size
            target['size'] = torch.tensor([img.height, img.width])
        
        return img, target
```

#### 3. **Config** - Data Path Issue

**Problem**: `data_dir: "data/insulators"` hardcoded

**Fix**: Use absolute paths or make configurable

---

## 🛠️ Fixed Files

I'll create corrected versions of the key files:

### Fixed requirements.txt

```txt
# PyTorch (CUDA 12.1 for L4 GPU)
torch>=2.0.0
torchvision>=0.15.0

# Transformers and DETR dependencies
transformers>=4.30.0

# Image processing
Pillow>=10.0.0

# Core ML libs
numpy>=1.24.0
scikit-learn>=1.3.0

# COCO tools
pycocotools>=2.0.6

# Configuration
omegaconf>=2.3.0

# Utilities
tqdm>=4.65.0
matplotlib>=3.7.0

# Roboflow for dataset download
roboflow>=1.1.0

# Ultralytics for YOLO (if needed)
ultralytics>=8.0.0
```

---

## 🚀 Complete Lightning.ai Setup Script

Create `setup_lightning.sh`:

```bash
#!/bin/bash

echo "=================================================="
echo "Lightning.ai Setup for HITL DETR Experiment"
echo "=================================================="

# 1. Install dependencies
echo ""
echo "Step 1: Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow numpy pycocotools omegaconf tqdm matplotlib scikit-learn roboflow ultralytics

# 2. Verify GPU
echo ""
echo "Step 2: Checking GPU..."
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 3. Download dataset
echo ""
echo "Step 3: Downloading dataset..."
echo "Please set ROBOFLOW_API_KEY first:"
echo "  export ROBOFLOW_API_KEY='your_key'"

if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo ""
    echo "WARNING: ROBOFLOW_API_KEY not set!"
    echo "Set it and run this download command:"
    echo ""
    echo "python -c \"from roboflow import Roboflow; rf = Roboflow(api_key='$ROBOFLOW_API_KEY'); rf.workspace('sofia-valdivieso-von-teuber').project('insulators-wo6lb').version(3).download('coco', location='data/insulators')\""
else
    python -c "from roboflow import Roboflow; rf = Roboflow(api_key='$ROBOFLOW_API_KEY'); rf.workspace('sofia-valdivieso-von-teuber').project('insulators-wo6lb').version(3).download('coco', location='data/insulators')"
fi

echo ""
echo "=================================================="
echo "Setup complete!"
echo "=================================================="
```

---

## 📥 Roboflow Download Commands

### Option 1: Using Python API (Recommended)

```bash
# Set API key
export ROBOFLOW_API_KEY='your_roboflow_api_key_here'

# Install roboflow
pip install roboflow

# Download insulators dataset (COCO format)
python << 'EOF'
from roboflow import Roboflow
import os

api_key = os.environ.get('ROBOFLOW_API_KEY')
if not api_key:
    print("ERROR: ROBOFLOW_API_KEY not set!")
    exit(1)

print("Downloading insulators dataset...")
rf = Roboflow(api_key=api_key)
project = rf.workspace("sofia-valdivieso-von-teuber").project("insulators-wo6lb")
dataset = project.version(3).download("coco", location="data/insulators")
print(f"✓ Downloaded to: {dataset.location}")
EOF
```

### Option 2: Direct Python Script

Create `download_data.py`:

```python
#!/usr/bin/env python3
from roboflow import Roboflow
import os

# Get API key from environment
api_key = os.environ.get('ROBOFLOW_API_KEY')
if not api_key:
    print("ERROR: Set ROBOFLOW_API_KEY first!")
    print("  export ROBOFLOW_API_KEY='your_key'")
    exit(1)

print("Downloading insulators dataset from Roboflow...")
rf = Roboflow(api_key=api_key)
project = rf.workspace("sofia-valdivieso-von-teuber").project("insulators-wo6lb")
dataset = project.version(3).download("coco", location="data/insulators")
print(f"✓ Dataset downloaded to: {dataset.location}")
```

Then run:
```bash
export ROBOFLOW_API_KEY='your_key'
python download_data.py
```

### Option 3: Using Roboflow CLI (if available)

```bash
# Install roboflow CLI
pip install roboflow

# Download
roboflow download -w sofia-valdivieso-von-teuber -p insulators-wo6lb -v 3 -f coco -l data/insulators
```

---

## 🔧 Configuration Fixes

### Update `configs/detr_insulator.yaml`

Replace hardcoded paths:

```yaml
data:
  dataset_name: "insulators"
  data_dir: "data/insulators"  # ← Will be downloaded here
  train_annotations: "train/_annotations.coco.json"
  val_annotations: "valid/_annotations.coco.json"
  test_annotations: "test/_annotations.coco.json"
  image_size: 640  # ← Increase from 512 for better accuracy
  batch_size: 16   # ← Increase for L4 GPU
  num_workers: 4
```

---

## 📝 Complete Setup Sequence for Lightning.ai

### Copy-Paste This Entire Block:

```bash
#!/bin/bash
# Complete Lightning.ai setup for HITL DETR experiment

echo "Setting up HITL DETR Experiment..."

# 1. Navigate to project
cd ~/DETR-Insulator-Detection  # or wherever your code is

# 2. Install dependencies
echo "Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow numpy pycocotools omegaconf tqdm matplotlib scikit-learn roboflow ultralytics

# 3. Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 4. Set API key (REPLACE WITH YOUR KEY!)
export ROBOFLOW_API_KEY='YOUR_ROBOFLOW_API_KEY_HERE'

# 5. Download dataset
echo "Downloading dataset..."
python -c "
from roboflow import Roboflow
import os

rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
project = rf.workspace('sofia-valdivieso-von-teuber').project('insulators-wo6lb')
dataset = project.version(3).download('coco', location='data/insulators')
print(f'Downloaded to: {dataset.location}')
"

# 6. Verify download
echo "Verifying download..."
echo "Train: $(ls data/insulators/train/*.jpg 2>/dev/null | wc -l) images"
echo "Valid: $(ls data/insulators/valid/*.jpg 2>/dev/null | wc -l) images"
echo "Test: $(ls data/insulators/test/*.jpg 2>/dev/null | wc -l) images"

echo ""
echo "✅ Setup complete! Ready to train."
```

---

## 🎯 After Setup - Run HITL Experiment

Once dataset is downloaded, you can either:

### Option A: Use DETR (Your Current Code)

```bash
python train.py --config configs/detr_insulator.yaml
```

### Option B: Use YOLO for HITL Proof (Recommended)

```bash
# Copy the HITL-proof scripts into your Lightning.ai workspace
# Then run:
bash ../RUN_ALL.sh
```

---

## 📊 Expected Directory Structure After Download

```
DETR-Insulator-Detection/
├── configs/
│   └── detr_insulator.yaml
├── src/
│   └── dataset.py
├── data/
│   └── insulators/              ← Downloaded here
│       ├── train/
│       │   ├── _annotations.coco.json
│       │   └── *.jpg (hundreds of images)
│       ├── valid/
│       │   ├── _annotations.coco.json
│       │   └── *.jpg
│       └── test/
│           ├── _annotations.coco.json
│           └── *.jpg
├── requirements.txt
└── train.py (you'll need to create this)
```

---

## 🚨 Common Issues & Solutions

### Issue 1: Roboflow API Key Invalid

**Error**: `Authentication failed`

**Solution**: Get fresh API key from https://app.roboflow.com/settings/api

### Issue 2: Download to Wrong Location

**Error**: Dataset downloaded to unexpected folder

**Solution**: Roboflow creates subdirectory. Check:
```bash
find data/ -name "_annotations.coco.json"
```

### Issue 3: CUDA Version Mismatch

**Error**: `CUDA driver version is insufficient`

**Solution**: Use CPU or update PyTorch version

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## ✅ Verification Checklist

After setup, verify:

- [ ] PyTorch CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Dataset downloaded: `ls data/insulators/train/*.jpg | wc -l` shows ~400+ images
- [ ] Annotations exist: `cat data/insulators/train/_annotations.coco.json | head`
- [ ] GPU detected: `nvidia-smi` shows L4

---

## 📖 Complete Example Session

```bash
# Terminal session in Lightning.ai Studio

# 1. Navigate to project
cd ~/DETR-Insulator-Detection

# 2. Set API key
export ROBOFLOW_API_KEY='pk_xxxxxxxxxxxxxxxxxxx'

# 3. Quick setup
pip install roboflow torch torchvision transformers

# 4. Download data (one command!)
python -c "from roboflow import Roboflow; Roboflow(api_key='$ROBOFLOW_API_KEY').workspace('sofia-valdivieso-von-teuber').project('insulators-wo6lb').version(3).download('coco', location='data/insulators')"

# 5. Verify
ls -lh data/insulators/train/ | head

# 6. You're ready to train!
```

---

## 🎓 Pro Tips

1. **Use tmux** for long training runs:
   ```bash
   tmux new -s detr
   # Run training
   # Ctrl+B, D to detach
   # Later: tmux attach -t detr
   ```

2. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Save API key** in a file (Lightning.ai persists):
   ```bash
   echo "export ROBOFLOW_API_KEY='your_key'" >> ~/.bashrc
   source ~/.bashrc
   ```

---

## 🔗 Useful Links

- **Roboflow Insulators Dataset**: https://universe.roboflow.com/sofia-valdivieso-von-teuber/insulators-wo6lb/dataset/3
- **Lightning.ai Docs**: https://lightning.ai/docs
- **Roboflow API Docs**: https://docs.roboflow.com/api-reference/workspace

---

**Ready to download?** Copy-paste this complete command:

```bash
export ROBOFLOW_API_KEY='YOUR_KEY_HERE' && pip install roboflow && python -c "from roboflow import Roboflow; Roboflow(api_key='$ROBOFLOW_API_KEY').workspace('sofia-valdivieso-von-teuber').project('insulators-wo6lb').version(3).download('coco', location='data/insulators')"
```

🚀 **Replace `YOUR_KEY_HERE` with your actual API key and run!**
