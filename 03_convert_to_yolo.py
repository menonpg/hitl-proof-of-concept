#!/usr/bin/env python3
"""
Step 3: Convert COCO to YOLO Format
====================================
Converts COCO JSON annotations to YOLO format for training.

YOLO format: one .txt file per image with lines:
    class_id x_center y_center width height
    
All values normalized to [0, 1].

Usage:
    python 03_convert_to_yolo.py
"""

import json
from pathlib import Path


def coco_to_yolo(coco_json_path, images_dir, labels_dir, class_mapping):
    """
    Convert COCO annotations to YOLO format.
    
    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Directory containing images
        labels_dir: Directory to save YOLO labels
        class_mapping: Dict mapping COCO category IDs to YOLO class IDs
    """
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image ID to filename mapping
    images = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Convert each image's annotations
    converted = 0
    for img_id, img_info in images.items():
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Create label file path
        label_file = labels_dir / (Path(img_info['file_name']).stem + '.txt')
        
        # Convert annotations for this image
        yolo_lines = []
        if img_id in annotations_by_image:
            for ann in annotations_by_image[img_id]:
                # COCO bbox: [x, y, width, height] (top-left corner)
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                # Map COCO category ID to YOLO class ID
                coco_cat_id = ann['category_id']
                yolo_class_id = class_mapping.get(coco_cat_id, 0)
                
                # Format: class x_center y_center width height
                yolo_lines.append(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        # Write label file
        with open(label_file, 'w') as f:
            f.writelines(yolo_lines)
        
        converted += 1
    
    return converted


def main():
    """Convert all splits to YOLO format."""
    print("\n" + "="*70)
    print("HITL Proof-of-Concept: Step 3 - Convert to YOLO Format")
    print("="*70 + "\n")
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    splits_dir = data_dir / "splits"
    
    if not splits_dir.exists():
        print(f"❌ Splits directory not found: {splits_dir}")
        print("\n⚠️  Run step 2 first: python 02_split_dataset.py")
        return 1
    
    # Define class mapping (COCO category ID -> YOLO class ID)
    # Insulators dataset has category_id=1 for insulators
    # YOLO classes are 0-indexed, so we map 1->0
    class_mapping = {
        1: 0  # insulators -> class 0
    }
    
    splits_to_convert = ['baseline', 'iter1', 'iter2', 'iter3', 'full', 'val', 'test']
    
    print("Converting splits to YOLO format:\n")
    
    for split_name in splits_to_convert:
        split_dir = splits_dir / split_name
        if not split_dir.exists():
            continue
        
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        coco_json = split_dir / "_annotations.coco.json"
        
        if not coco_json.exists():
            print(f"⚠️  Skipping {split_name}: no annotations found")
            continue
        
        labels_dir.mkdir(exist_ok=True)
        
        print(f"📝 Converting {split_name}...")
        converted = coco_to_yolo(coco_json, images_dir, labels_dir, class_mapping)
        print(f"   ✓ Converted {converted} label files\n")
    
    # Create data.yaml files for each split
    print("📄 Creating YOLO data.yaml files...\n")
    
    for split_name in ['baseline', 'iter1', 'iter2', 'iter3', 'full']:
        split_dir = splits_dir / split_name
        if not split_dir.exists():
            continue
        
        yaml_content = f"""# YOLO dataset config for {split_name}
path: {splits_dir.absolute()}
train: {split_name}/images
val: val/images
test: test/images

# Classes
names:
  0: insulators

# Number of classes
nc: 1
"""
        
        yaml_path = split_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"   ✓ Created {split_name}/data.yaml")
    
    print("\n" + "="*70)
    print("✅ Step 3 Complete: YOLO Format Ready")
    print("="*70)
    print(f"\n📁 YOLO labels saved to: {splits_dir}/*/labels/")
    print("\n📋 Next step: Run python 04_train_all_iterations.py")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
