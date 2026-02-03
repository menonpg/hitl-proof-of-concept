#!/usr/bin/env python3
"""
HITL Proof-of-Concept: Faster R-CNN via Detectron2
===================================================
Tests HITL on two-stage detector architecture using Facebook's Detectron2

This represents the "classical" two-stage detection paradigm for comparison
with YOLO (one-stage) and DETR (transformer).

Usage:
    python hitl_detectron2_script.py --trials 1

Requirements:
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.0/index.html
"""

print("Installing dependencies...")
import subprocess
subprocess.run(['pip', 'install', '-q', 'roboflow', 'matplotlib', 'pycocotools'])

# Intelligent device detection
import torch

def get_device():
    if torch.cuda.is_available():
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("🍎 Using Apple Silicon (MPS)")
        print("⚠️  NOTE: Detectron2 doesn't support MPS, falling back to CPU")
        return "cpu"
    else:
        print("⚠️  Using CPU (will be slow!)")
        return "cpu"

device = get_device()

print("\nInstalling Detectron2...")
if device == "cuda":
    subprocess.run([
        'pip', 'install', '-q',
        'detectron2', '-f',
        'https://dl.fbaipublicfiles.com/detectron2/wheels/cu121/torch2.0/index.html'
    ])
else:
    print("⚠️  Detectron2 installation complex on CPU/MPS")
    print("   Recommend using Google Colab with GPU for Detectron2")
    print("\nFor CPU/MPS, recommend using:")
    print("   - YOLO (works on all platforms)")
    print("   - PyTorch Detection (torchvision.models)")
    exit(1)

# Data preparation (reuse from YOLO script)
from roboflow import Roboflow
from pathlib import Path
import json, random, shutil

random.seed(42)

# Download data
DATASET_ROOT = "/content/insulators-3"
if not Path(DATASET_ROOT).exists():
    rf = Roboflow(api_key="lbXALpBLK1UO9TLPqobo")
    project = rf.workspace("sofia-valdivieso-von-teuber").project("insulators-wo6lb")
    dataset = project.version(3).download("coco")
    DATASET_ROOT = dataset.location

dataset_path = Path(DATASET_ROOT)

# Load and filter annotations
with open(dataset_path / 'train' / '_annotations.coco.json') as f:
    data = json.load(f)

# Filter to insulators only
insulator_anns = [a for a in data['annotations'] if a['category_id'] == 1]
ann_map = {}
for ann in insulator_anns:
    if ann['image_id'] not in ann_map:
        ann_map[ann['image_id']] = []
    ann_map[ann['image_id']].append(ann)

print(f"\nFiltered to {len(insulator_anns)} insulator annotations from {len(ann_map)} images")

# Register dataset with Detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

def create_detectron2_dataset(img_ids, ann_map, img_map, img_folder):
    """Convert COCO format to Detectron2 format"""
    dataset_dicts = []
    
    for img_id in img_ids:
        if img_id not in img_map or img_id not in ann_map:
            continue
        
        img_info = img_map[img_id]
        record = {}
        
        record["file_name"] = str(img_folder / img_info['file_name'])
        record["image_id"] = img_id
        record["height"] = img_info['height']
        record["width"] = img_info['width']
        
        objs = []
        for ann in ann_map[img_id]:
            x, y, w, h = ann['bbox']
            obj = {
                "bbox": [x, y, x+w, y+h],  # Convert to xyxy
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": 0,  # All insulators = class 0
            }
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

# Create splits
img_map = {i['id']: i for i in data['images']}
all_ids = list(ann_map.keys())
random.shuffle(all_ids)

splits = {
    'baseline': 50,
    'iter1': 100,
    'iter2': 200,
    'iter3': 300,
    'full': min(800, len(all_ids))
}

print("\nRegistering datasets with Detectron2...")
for split_name, split_size in splits.items():
    split_ids = all_ids[:split_size]
    
    # Register dataset
    DatasetCatalog.register(
        f"insulators_{split_name}",
        lambda ids=split_ids: create_detectron2_dataset(
            ids, ann_map, img_map, dataset_path / 'train'
        )
    )
    
    MetadataCatalog.get(f"insulators_{split_name}").set(
        thing_classes=["insulator"]
    )
    
    print(f"  Registered: insulators_{split_name} ({split_size} images)")

# Train with Detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

results = []

for split_name in ['baseline', 'iter1', 'iter2', 'iter3', 'full']:
    print(f"\n{'='*70}")
    print(f"Training Faster R-CNN: {split_name}")
    print(f"{'='*70}")
    
    # Configure
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (f"insulators_{split_name}",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2 if device == "cpu" else 4
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 1000 if split_name == 'baseline' else 500  # Fewer iters for transfer learning
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only insulators
    cfg.OUTPUT_DIR = f"./output_{split_name}"
    
    Path(cfg.OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Train
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # Evaluate (simplified - full evaluation needs separate code)
    print(f"✅ {split_name} training complete")
    print(f"   Models saved to: {cfg.OUTPUT_DIR}")
    
    results.append({
        'model': 'Faster R-CNN',
        'iteration': split_name,
        'note': 'Full COCO evaluation needed for mAP'
    })

print("\n✅ Faster R-CNN experiments complete")
print("\n💡 NOTE: Detectron2 requires additional evaluation code for mAP")
print("   Recommend using MMDetection instead for easier evaluation")
