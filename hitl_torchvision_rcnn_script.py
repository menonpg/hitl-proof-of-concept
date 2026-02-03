#!/usr/bin/env python3
"""
HITL Proof-of-Concept: Faster R-CNN via TorchVision
====================================================
Cross-platform two-stage detector (works on CUDA/MPS/CPU)

Simpler alternative to Detectron2 - uses PyTorch's built-in models.
Best for manuscript comparison as it's easy to run and evaluate.

Usage:
    python hitl_torchvision_rcnn_script.py

Works on: Colab GPU, Mac M3 (MPS), CPU
"""

print("Installing dependencies...")
import subprocess
subprocess.run(['pip', 'install', '-q', 'torch', 'torchvision', 'roboflow', 'matplotlib', 'pycocotools'])

from roboflow import Roboflow
from pathlib import Path
import json, random, shutil, torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Device detection
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        batch_size = 4
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        batch_size = 4
        print("🍎 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        batch_size = 2
        print("⚠️  Using CPU (will be slow!)")
    
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    return device, batch_size

DEVICE, BATCH_SIZE = get_device()

# Download and prepare data
DATASET_ROOT = "/content/insulators-3"
if not Path(DATASET_ROOT).exists():
    print("\nDownloading dataset...")
    rf = Roboflow(api_key="lbXALpBLK1UO9TLPqobo")
    project = rf.workspace("sofia-valdivieso-von-teuber").project("insulators-wo6lb")
    dataset = project.version(3).download("coco")
    DATASET_ROOT = dataset.location

dataset_path = Path(DATASET_ROOT)

# Load annotations
with open(dataset_path / 'train' / '_annotations.coco.json') as f:
    data = json.load(f)

# Filter to insulators
random.seed(42)
insulator_anns = [a for a in data['annotations'] if a['category_id'] == 1]
ann_map = {}
for ann in insulator_anns:
    if ann['image_id'] not in ann_map:
        ann_map[ann['image_id']] = []
    ann_map[ann['image_id']].append(ann)

img_map = {i['id']: i for i in data['images']}
all_ids = list(ann_map.keys())
random.shuffle(all_ids)

print(f"\nFiltered to {len(insulator_anns)} insulator annotations from {len(ann_map)} images")

# Dataset class
class CocoDataset(Dataset):
    def __init__(self, img_ids, ann_map, img_map, img_folder, transforms=None):
        self.img_ids = img_ids
        self.ann_map = ann_map
        self.img_map = img_map
        self.img_folder = img_folder
        self.transforms = transforms
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        from PIL import Image
        import torchvision.transforms.functional as F
        
        img_id = self.img_ids[idx]
        img_info = self.img_map[img_id]
        img_path = self.img_folder / img_info['file_name']
        img = Image.open(img_path).convert('RGB')
        
        # Get bboxes
        boxes = []
        labels = []
        for ann in self.ann_map.get(img_id, []):
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])  # xyxy format
            labels.append(1)  # Class 1 for insulators (0 is background)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        img = F.to_tensor(img)
        
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([img_id])}
        
        return img, target

# Train function
def train_faster_rcnn(model, data_loader, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
        
        print(f"  Epoch {epoch+1}/{epochs}: Loss={epoch_loss/len(data_loader):.4f}")
    
    return model

# Simple evaluation (count predictions)
@torch.no_grad()
def evaluate_model(model, data_loader, device):
    model.eval()
    total_preds = 0
    total_boxes = 0
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        predictions = model(images)
        
        for pred in predictions:
            total_preds += len(pred['boxes'])
        for target in targets:
            total_boxes += len(target['boxes'])
    
    return total_preds, total_boxes

# Run experiments
print("\nTraining Faster R-CNN on incremental datasets...")
results = []

for split_name, split_size in splits.items():
    split_ids = all_ids[:split_size]
    
    print(f"\n{'='*70}")
    print(f"Training: {split_name} ({split_size} images)")
    print(f"{'='*70}")
    
    # Create dataset
    dataset = CocoDataset(split_ids, ann_map, img_map, dataset_path / 'train')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                       collate_fn=lambda x: tuple(zip(*x)), num_workers=0)
    
    # Create model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)  # 1 class + background
    model = model.to(DEVICE)
    
    # Train
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    epochs = 10 if split_name in ['baseline', 'full'] else 5
    
    model = train_faster_rcnn(model, loader, optimizer, DEVICE, epochs)
    
    # Simplified evaluation
    total_preds, total_gt = evaluate_model(model, loader, DEVICE)
    
    results.append({
        'iteration': split_name,
        'total_predictions': total_preds,
        'total_ground_truth': total_gt,
        'note': 'Simplified evaluation - proper mAP calculation needed'
    })
    
    print(f"✅ {split_name}: {total_preds} predictions, {total_gt} ground truth boxes")

print("\n✅ Faster R-CNN experiments complete")
print("\n💡 For manuscript-quality evaluation:")
print("   Use COCO evaluation API for precise mAP@0.5 calculation")
print("   Or use Detectron2's built-in evaluator")
