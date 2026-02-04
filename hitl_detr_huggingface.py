#!/usr/bin/env python3
"""
HITL Proof-of-Concept: DETR via Hugging Face
==============================================
Transformer-based detection using Facebook's DETR

Simpler and more reliable than building DETR from scratch.
Uses pre-trained weights from COCO.

Usage:
    python hitl_detr_huggingface.py

Works on: Colab GPU recommended (DETR is slow on CPU/MPS)
"""

print("Installing dependencies...")
import subprocess
subprocess.run(['pip', 'install', '-q', 'transformers', 'datasets', 'roboflow', 'matplotlib', 'pycocotools', 'pillow'])

from roboflow import Roboflow
from pathlib import Path
import json, random, shutil, torch
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Device detection
def get_device():
    if torch.cuda.is_available():
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device('cuda'), 2
    elif torch.backends.mps.is_available():
        print("🍎 Using MPS (will be slow for DETR)")
        return torch.device('mps'), 2
    else:
        print("⚠️  Using CPU (VERY slow for DETR!)")
        return torch.device('cpu'), 1

DEVICE, BATCH_SIZE = get_device()

# Download data
DATASET_ROOT = "/content/insulators-3"
if not Path(DATASET_ROOT).exists():
    print("\nDownloading dataset...")
    rf = Roboflow(api_key="lbXALpBLK1UO9TLPqobo")
    project = rf.workspace("sofia-valdivieso-von-teuber").project("insulators-wo6lb")
    dataset = project.version(3).download("coco")
    DATASET_ROOT = dataset.location

dataset_path = Path(DATASET_ROOT)

# Prepare data
random.seed(42)
with open(dataset_path / 'train' / '_annotations.coco.json') as f:
    data = json.load(f)

# Filter to insulators
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

# Dataset class for DETR
from torch.utils.data import Dataset
from PIL import Image
from transformers import DetrImageProcessor

class CocoDetrDataset(Dataset):
    def __init__(self, img_ids, ann_map, img_map, img_folder, processor):
        self.img_ids = img_ids
        self.ann_map = ann_map
        self.img_map = img_map
        self.img_folder = img_folder
        self.processor = processor
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.img_map[img_id]
        img_path = self.img_folder / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        
        # Get annotations
        boxes = []
        labels = []
        for ann in self.ann_map.get(img_id, []):
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])  # xyxy
            labels.append(0)  # Class 0 for insulators
        
        # Prepare targets
        target = {
            'boxes': boxes,
            'class_labels': labels,
            'image_id': img_id
        }
        
        # Process with DETR processor
        encoding = self.processor(images=image, annotations=[target], return_tensors="pt")
        
        # Remove batch dimension
        pixel_values = encoding["pixel_values"].squeeze(0)
        target = encoding["labels"][0]
        
        return pixel_values, target

# Initialize DETR
from transformers import DetrForObjectDetection, DetrImageProcessor

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

print("\n🤖 Initializing DETR model...")

# Define splits
splits = {
    'baseline': 50,
    'iter1': 100,
    'iter2': 200,
    'iter3': 300,
    'full': min(800, len(all_ids))
}

results = []
prev_model_path = None

# Train each iteration
for split_name, split_size in splits.items():
    split_ids = all_ids[:split_size]
    
    print(f"\n{'='*70}")
    print(f"Training DETR: {split_name} ({split_size} images)")
    print(f"{'='*70}")
    
    # Create dataset
    train_dataset = CocoDetrDataset(split_ids, ann_map, img_map, dataset_path / 'train', processor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize model
    if prev_model_path:
        print(f"  Loading from: {prev_model_path}")
        model = DetrForObjectDetection.from_pretrained(prev_model_path)
    else:
        print("  Loading from: facebook/detr-resnet-50 (COCO pretrained)")
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=1,  # Only insulators
            ignore_mismatched_sizes=True
        )
    
    model = model.to(DEVICE)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 10 if split_name in ['baseline', 'full'] else 5  # Reduced for proof-of-concept
    
    # Train
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        batches = 0
        
        for pixel_values_list, targets_list in train_loader:
            # Stack pixel values
            pixel_values = torch.stack(pixel_values_list).to(DEVICE)
            
            # Prepare targets
            targets = []
            for target in targets_list:
                targets.append({k: v.to(DEVICE) for k, v in target.items()})
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=targets)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
        
        avg_loss = epoch_loss / batches
        print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}")
    
    # Save model
    save_path = f"detr_{split_name}"
    model.save_pretrained(save_path)
    prev_model_path = save_path
    
    print(f"✅ {split_name} complete - model saved to {save_path}")
    
    # Simplified evaluation (just count if model runs)
    model.eval()
    with torch.no_grad():
        sample_imgs = next(iter(train_loader))[0][:1]
        sample_input = torch.stack(sample_imgs).to(DEVICE)
        preds = model(pixel_values=sample_input)
        num_preds = (preds.logits.softmax(-1)[..., :-1].max(-1).values > 0.5).sum().item()
    
    results.append({
        'iteration': split_name,
        'num_detections': num_preds,
        'note': 'Full mAP evaluation requires COCO API'
    })

print("\n✅ DETR experiments complete")
print("\n📊 Results (simplified):")
for r in results:
    print(f"  {r['iteration']:10s}: {r['num_detections']} detections")

print("\n💡 For precise mAP@0.5:")
print("   Use COCO evaluation API or adapt utility-inventory-detr-main evaluation code")
