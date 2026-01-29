# ⚡ Lightning.ai Quick Start - HITL DETR Experiment

## 🚀 Copy-Paste This ENTIRE Block into Lightning.ai Terminal

```bash
#!/bin/bash
# ============================================================================
# Complete Lightning.ai Setup for HITL DETR Insulator Detection
# ============================================================================

echo "🚀 Starting Lightning.ai setup..."

# 1. Install PyTorch with CUDA 12.1
echo ""
echo "Step 1: Installing PyTorch with CUDA 12.1..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install dependencies
echo ""
echo "Step 2: Installing dependencies..."
pip install -r requirements.txt

# 3. Verify GPU
echo ""
echo "Step 3: Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print(f'✓ PyTorch version: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')"

# 4. Download insulator dataset from Roboflow
echo ""
echo "Step 4: Downloading insulators dataset from Roboflow..."
pip install roboflow

python << 'EOPYTHON'
from roboflow import Roboflow
rf = Roboflow(api_key="lbXALpBLK1UO9TLPqobo")
project = rf.workspace("sofia-valdivieso-von-teuber").project("insulators-wo6lb")
version = project.version(3)
dataset = version.download("coco")
print(f"✓ Dataset downloaded to: {dataset.location}")
EOPYTHON

# 5. Verify download
echo ""
echo "Step 5: Verifying dataset download..."
echo "Checking for dataset files..."

# Find where Roboflow downloaded the data
DATASET_DIR=$(find . -name "_annotations.coco.json" -type f | head -1 | xargs dirname)
if [ -z "$DATASET_DIR" ]; then
    echo "❌ Dataset not found! Check download step."
else
    echo "✓ Dataset found at: $DATASET_DIR"
    
    # Count images
    TRAIN_COUNT=$(find $DATASET_DIR/../train -name "*.jpg" 2>/dev/null | wc -l)
    VALID_COUNT=$(find $DATASET_DIR/../valid -name "*.jpg" 2>/dev/null | wc -l)
    TEST_COUNT=$(find $DATASET_DIR/../test -name "*.jpg" 2>/dev/null | wc -l)
    
    echo "  Train images: $TRAIN_COUNT"
    echo "  Valid images: $VALID_COUNT"
    echo "  Test images: $TEST_COUNT"
    
    # Create symlink to expected location
    mkdir -p data
    if [ ! -L "data/insulators" ]; then
        ln -s "$(cd $DATASET_DIR/.. && pwd)" data/insulators
        echo "✓ Created symlink: data/insulators -> dataset"
    fi
fi

echo ""
echo "========================================================================"
echo "✅ Setup Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Verify data: ls data/insulators/train/ | head"
echo "  2. Run HITL experiment (choose one):"
echo ""
echo "     Option A - YOLO HITL Proof (Recommended):"
echo "       cd .."
echo "       bash RUN_ALL.sh"
echo ""
echo "     Option B - DETR Training:"
echo "       python train.py --config configs/detr_insulator.yaml"
echo ""
```

---

## 📋 What This Does

1. ✅ Installs PyTorch with CUDA 12.1
2. ✅ Installs all dependencies
3. ✅ Verifies GPU is available
4. ✅ Downloads 602 insulator images from Roboflow using your API key
5. ✅ Creates proper data directory structure
6. ✅ Verifies download completed successfully

**Time**: 5-10 minutes

---

## 🔑 Roboflow Download Command (Standalone)

If you just need to download the dataset:

```bash
pip install roboflow

python << 'EOF'
from roboflow import Roboflow
rf = Roboflow(api_key="lbXALpBLK1UO9TLPqobo")
project = rf.workspace("sofia-valdivieso-von-teuber").project("insulators-wo6lb")
version = project.version(3)
dataset = version.download("coco")
print(f"✓ Downloaded to: {dataset.location}")
EOF
```

---

## 📊 Expected Output

```
Downloading insulators dataset from Roboflow...
loading Roboflow workspace...
loading Roboflow project...
Downloading Dataset Version Zip in coco to /root: 100%|████| 123M/123M [00:05<00:00, 23.1MB/s]
Extracting Dataset Version Zip to /root in coco:: 100%|████| 602/602 [00:02<00:00, 245file/s]

✓ Dataset downloaded to: /root/insulators-wo6lb-3
  Train images: 421
  Valid images: 120  
  Test images: 61
```

---

## 🔧 If Roboflow Download Fails

### Alternative: Manual Download

1. **Visit**: https://universe.roboflow.com/sofia-valdivieso-von-teuber/insulators-wo6lb/dataset/3
2. **Click**: "Download Dataset"
3. **Select**: "COCO JSON" format
4. **Upload** the ZIP to Lightning.ai
5. **Extract**:
   ```bash
   unzip insulators-wo6lb-3.zip -d data/insulators
   ```

---

## 📁 Expected Directory Structure

After download completes:

```
DETR-Insulator-Detection/
├── data/
│   └── insulators/ (or insulators-wo6lb-3/)
│       ├── train/
│       │   ├── _annotations.coco.json
│       │   └── *.jpg (421 images)
│       ├── valid/
│       │   ├── _annotations.coco.json
│       │   └── *.jpg (120 images)
│       └── test/
│           ├── _annotations.coco.json
│           └── *.jpg (61 images)
├── configs/
│   └── detr_insulator.yaml
├── src/
│   └── dataset.py
└── requirements.txt
```

---

## ⚠️ SECURITY NOTE

**API Key Exposed**: Your Roboflow API key `lbXALpBLK1UO9TLPqobo` is now public in the code you shared.

### Recommendation:
1. **Rotate your API key** at https://app.roboflow.com/settings/api
2. **Use environment variables** instead of hardcoding:
   ```bash
   export ROBOFLOW_API_KEY='your_new_key'
   # Then in Python:
   import os
   rf = Roboflow(api_key=os.environ['ROBOFLOW_API_KEY'])
   ```

---

## ✅ Verification Commands

After running the setup block, verify everything works:

```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check dataset
ls -lh data/insulators/train/ | head
cat data/insulators/train/_annotations.coco.json | head -20

# Count images
echo "Train: $(ls data/insulators/train/*.jpg | wc -l)"
echo "Valid: $(ls data/insulators/valid/*.jpg | wc -l)"
echo "Test: $(ls data/insulators/test/*.jpg | wc -l)"
```

Expected output:
```
Train: 421
Valid: 120
Test: 61
```

---

## 🎯 Next Steps After Setup

### Option 1: Run Complete HITL Proof (Recommended)

```bash
# Go to parent directory
cd ..

# Run the complete HITL experiment
bash RUN_ALL.sh --device 0 --batch 16
```

This will train 5 models incrementally and prove HITL works!

### Option 2: Quick Test Training

```bash
# Quick training test (10 epochs)
python << 'EOF'
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
results = model.train(
    data='data/insulators/data.yaml',  # You may need to create this
    epochs=10,
    imgsz=640,
    batch=16,
    device=0
)
print(f"✓ Test training complete!")
EOF
```

---

## 💡 Pro Tips for Lightning.ai

1. **Use tmux** for long runs:
   ```bash
   tmux new -s hitl
   bash RUN_ALL.sh
   # Ctrl+B, then D to detach
   ```

2. **Monitor GPU**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Download results**:
   ```bash
   # After experiment completes
   zip -r results.zip results/
   # Download via Files panel in Lightning.ai
   ```

---

## 🚨 Troubleshooting

### Error: "API key invalid"
Solution: Use new API key from https://app.roboflow.com/settings/api

### Error: "CUDA out of memory"
Solution: Reduce batch size:
```bash
bash RUN_ALL.sh --device 0 --batch 8
```

### Error: "Dataset not found"
Solution: Check download location:
```bash
find . -name "_annotations.coco.json"
```

---

**Ready to Go?** Just copy-paste the entire setup block at the top into your Lightning.ai terminal! 🚀

Then run `bash ../RUN_ALL.sh` to execute the complete HITL proof experiment.
