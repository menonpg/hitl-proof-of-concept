#!/usr/bin/env python3
"""
Step 2: Split Dataset into Incremental Batches
===============================================
Creates incremental training splits to simulate HITL iterations.

Splits:
- Baseline: 50 random training images
- Iter1: +50 more (100 total)
- Iter2: +100 more (200 total)
- Iter3: +100 more (300 total)
- Full: All training images

Test set remains fixed throughout all iterations.

Usage:
    python 02_split_dataset.py
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict


def load_coco_json(json_path):
    """Load COCO format annotations."""
    with open(json_path, 'r') as f:
        return json.load(f)


def save_coco_json(data, json_path):
    """Save COCO format annotations."""
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def filter_annotations(annotations, image_ids):
    """Filter annotations to only include specified image IDs."""
    return [ann for ann in annotations if ann['image_id'] in image_ids]


def main():
    """Split dataset into incremental batches."""
    print("\n" + "="*70)
    print("HITL Proof-of-Concept: Step 2 - Split Dataset")
    print("="*70 + "\n")
    
    # Setup paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    raw_dir = data_dir / "raw"
    splits_dir = data_dir / "splits"
    
    # Find the downloaded dataset
    # Roboflow downloads to a subfolder with project name
    train_dir = None
    for subdir in raw_dir.iterdir():
        if subdir.is_dir():
            potential_train = subdir / "train"
            if potential_train.exists():
                train_dir = potential_train
                valid_dir = subdir / "valid"
                test_dir = subdir / "test"
                break
    
    if not train_dir or not train_dir.exists():
        print(f"❌ Training data not found in {raw_dir}")
        print("\n📋 Expected structure:")
        print("  data/raw/")
        print("    └── insulators-wo6lb-3/  (or similar)")
        print("        ├── train/")
        print("        │   ├── _annotations.coco.json")
        print("        │   └── *.jpg")
        print("        ├── valid/")
        print("        └── test/")
        print("\n⚠️  Run step 1 first: python 01_download_dataset.py")
        return 1
    
    print(f"✓ Found training data: {train_dir}")
    print(f"✓ Found validation data: {valid_dir}")
    print(f"✓ Found test data: {test_dir}\n")
    
    # Load annotations
    train_json_path = train_dir / "_annotations.coco.json"
    if not train_json_path.exists():
        print(f"❌ Annotations not found: {train_json_path}")
        return 1
    
    train_data = load_coco_json(train_json_path)
    print(f"✓ Loaded training annotations")
    print(f"  Images: {len(train_data['images'])}")
    print(f"  Annotations: {len(train_data['annotations'])}")
    print(f"  Categories: {len(train_data['categories'])}\n")
    
    # Get all training image IDs
    all_image_ids = [img['id'] for img in train_data['images']]
    total_images = len(all_image_ids)
    
    print(f"Total training images available: {total_images}\n")
    
    # Set random seed for reproducibility
    random.seed(42)
    random.shuffle(all_image_ids)
    
    # Define splits
    splits = {
        'baseline': 50,
        'iter1': 100,
        'iter2': 200,
        'iter3': 300,
        'full': total_images
    }
    
    print("Creating incremental splits:")
    for split_name, size in splits.items():
        actual_size = min(size, total_images)
        print(f"  {split_name:10s}: {actual_size:3d} images")
    print()
    
    # Create splits directory
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image ID mapping
    image_id_to_filename = {img['id']: img['file_name'] for img in train_data['images']}
    
    # Create each split
    for split_name, split_size in splits.items():
        print(f"\n📦 Creating split: {split_name} ({min(split_size, total_images)} images)")
        
        # Select image IDs for this split
        split_image_ids = set(all_image_ids[:min(split_size, total_images)])
        
        # Create split directory
        split_dir = splits_dir / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter images and annotations
        split_images = [img for img in train_data['images'] if img['id'] in split_image_ids]
        split_annotations = filter_annotations(train_data['annotations'], split_image_ids)
        
        # Copy images
        copied = 0
        for img in split_images:
            src = train_dir / img['file_name']
            dst = images_dir / img['file_name']
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1
        
        print(f"  ✓ Copied {copied} images")
        
        # Save COCO annotations
        split_coco = {
            'info': train_data.get('info', {}),
            'licenses': train_data.get('licenses', []),
            'images': split_images,
            'annotations': split_annotations,
            'categories': train_data['categories']
        }
        
        coco_json_path = split_dir / "_annotations.coco.json"
        save_coco_json(split_coco, coco_json_path)
        print(f"  ✓ Saved COCO annotations: {coco_json_path.name}")
        print(f"    Images: {len(split_images)}, Annotations: {len(split_annotations)}")
    
    # Copy validation and test sets (unchanged for all iterations)
    print(f"\n📦 Copying validation set...")
    val_split_dir = splits_dir / "val"
    val_images_dir = val_split_dir / "images"
    val_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy validation images
    val_copied = 0
    for img_file in valid_dir.glob("*.jpg"):
        shutil.copy2(img_file, val_images_dir / img_file.name)
        val_copied += 1
    for img_file in valid_dir.glob("*.png"):
        shutil.copy2(img_file, val_images_dir / img_file.name)
        val_copied += 1
    
    # Copy validation annotations
    val_json = valid_dir / "_annotations.coco.json"
    if val_json.exists():
        shutil.copy2(val_json, val_split_dir / "_annotations.coco.json")
    
    print(f"  ✓ Copied {val_copied} validation images")
    
    # Copy test set
    print(f"\n📦 Copying test set...")
    test_split_dir = splits_dir / "test"
    test_images_dir = test_split_dir / "images"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy test images
    test_copied = 0
    for img_file in test_dir.glob("*.jpg"):
        shutil.copy2(img_file, test_images_dir / img_file.name)
        test_copied += 1
    for img_file in test_dir.glob("*.png"):
        shutil.copy2(img_file, test_images_dir / img_file.name)
        test_copied += 1
    
    # Copy test annotations
    test_json = test_dir / "_annotations.coco.json"
    if test_json.exists():
        shutil.copy2(test_json, test_split_dir / "_annotations.coco.json")
    
    print(f"  ✓ Copied {test_copied} test images")
    
    # Summary
    print("\n" + "="*70)
    print("✅ Step 2 Complete: Dataset Split")
    print("="*70)
    print(f"\n📁 Splits saved to: {splits_dir}")
    print("\n📊 Split Summary:")
    for split_name in ['baseline', 'iter1', 'iter2', 'iter3', 'full']:
        split_dir = splits_dir / split_name
        if split_dir.exists():
            img_count = len(list((split_dir / "images").glob("*")))
            print(f"  {split_name:10s}: {img_count:3d} images")
    print(f"  {'val':10s}: {val_copied:3d} images")
    print(f"  {'test':10s}: {test_copied:3d} images")
    
    print("\n📋 Next step: Run python 03_convert_to_yolo.py")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
