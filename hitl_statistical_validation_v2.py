#!/usr/bin/env python3
"""
HITL Statistical Validation v2: Proper Out-of-Sample Testing
=============================================================
Version 2 with proper train/val splits for each iteration.

Key Changes from v1:
- Splits each iteration into 80% train / 20% validation
- True out-of-sample mAP measurement
- More realistic performance estimates

Usage:
    python hitl_statistical_validation_v2.py --trials 3

Output:
    - v2_results_trial_*.json (with train/val split mAP)
    - v2_aggregated_results.json
    - v2_publication_plot.png (true out-of-sample results)
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch

# Device detection
def get_best_device():
    if torch.cuda.is_available():
        return 0, 16, torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        return 'mps', 16, "Apple Silicon (MPS)"
    else:
        return 'cpu', 8, "CPU (slow)"

DEVICE, BATCH_SIZE, device_name = get_best_device()
print(f"🚀 Using: {device_name}, batch={BATCH_SIZE}")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=3)
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
parser.add_argument('--val_split', type=float, default=0.2, help='Validation split (default 20%)')
args = parser.parse_args()

print(f"\n{'='*70}")
print(f"HITL Statistical Validation v2: Out-of-Sample Testing")
print(f"Trials: {args.trials}, Val split: {args.val_split*100:.0f}%")
print(f"{'='*70}\n")

# Load dataset
from roboflow import Roboflow

DATASET_ROOT = "/content/insulators-3"
if not Path(DATASET_ROOT).exists():
    rf = Roboflow(api_key="lbXALpBLK1UO9TLPqobo")
    dataset = rf.workspace("sofia-valdivieso-von-teuber").project("insulators-wo6lb").version(3).download("coco")
    DATASET_ROOT = dataset.location

dataset_path = Path(DATASET_ROOT)

with open(dataset_path / 'train' / '_annotations.coco.json') as f:
    data = json.load(f)

# Filter to insulators
insulator_anns = [a for a in data['annotations'] if a['category_id'] == 1]
ann_map_base = {}
for ann in insulator_anns:
    if ann['image_id'] not in ann_map_base:
        ann_map_base[ann['image_id']] = []
    ann_map_base[ann['image_id']].append(ann)

img_map = {i['id']: i for i in data['images']}
all_image_ids = list(ann_map_base.keys())

print(f"Dataset: {len(all_image_ids)} images with insulators\n")

# Run trials
all_trials = []

for trial_idx in range(args.trials):
    seed = args.seeds[trial_idx] if trial_idx < len(args.seeds) else 100 + trial_idx
    
    print(f"\n{'='*70}")
    print(f"Trial {trial_idx + 1}/{args.trials} (seed={seed})")
    print(f"{'='*70}\n")
    
    random.seed(seed)
    np.random.seed(seed)
    
    trial_image_ids = all_image_ids.copy()
    random.shuffle(trial_image_ids)
    
    splits = {
        'baseline': 50,
        'iter1': 100,
        'iter2': 200,
        'iter3': 300,
        'full': min(800, len(trial_image_ids))
    }
    
    trial_results = {'seed': seed, 'iterations': []}
    
    # For each iteration, create proper train/val split
    for split_name, total_size in splits.items():
        split_ids = trial_image_ids[:total_size]
        
        # 80/20 train/val split
        split_point = int(total_size * (1 - args.val_split))
        train_ids = split_ids[:split_point]
        val_ids = split_ids[split_point:]
        
        print(f"\n{split_name}: {len(train_ids)} train, {len(val_ids)} val")
        
        # Create directories
        base_dir = Path(f'/content/v2_splits_trial_{seed}/{split_name}')
        for subset in ['train', 'val']:
            (base_dir / subset / 'images').mkdir(parents=True, exist_ok=True)
            (base_dir / subset / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Prepare train and val sets
        for subset, ids in [('train', train_ids), ('val', val_ids)]:
            for img_id in ids:
                if img_id not in img_map:
                    continue
                img_info = img_map[img_id]
                src = dataset_path / 'train' / img_info['file_name']
                
                if src.exists():
                    shutil.copy2(src, base_dir / subset / 'images' / img_info['file_name'])
                    
                    with open(base_dir / subset / 'labels' / (Path(img_info['file_name']).stem + '.txt'), 'w') as f:
                        if img_id in ann_map_base:
                            for ann in ann_map_base[img_id]:
                                x, y, w, h = ann['bbox']
                                xc, yc = (x + w/2) / img_info['width'], (y + h/2) / img_info['height']
                                wn, hn = w / img_info['width'], h / img_info['height']
                                f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
        
        # Create data.yaml with proper train/val split
        with open(base_dir / 'data.yaml', 'w') as f:
            f.write(f"""path: /content/v2_splits_trial_{seed}
train: {split_name}/train
val: {split_name}/val

names:
  0: insulators

nc: 1
""")
        
        # Train
        from ultralytics import YOLO
        
        weights = trial_results['prev_weights'] if split_name != 'baseline' and 'prev_weights' in trial_results else 'yolo11n.pt'
        epochs = 50 if split_name in ['baseline', 'full'] else 30
        
        model = YOLO(weights)
        results = model.train(
            data=str(base_dir / 'data.yaml'),
            epochs=epochs,
            imgsz=640,
            batch=BATCH_SIZE,
            device=DEVICE,
            name=f'v2_{split_name}_trial{seed}',
            patience=10,
            verbose=False
        )
        
        # Evaluate on VALIDATION set (out-of-sample!)
        metrics = model.val()
        
        iter_results = {
            'iteration': split_name,
            'train_size': len(train_ids),
            'val_size': len(val_ids),
            'val_map50': float(metrics.box.map50),  # Out-of-sample!
            'val_precision': float(metrics.box.mp),
            'val_recall': float(metrics.box.mr)
        }
        
        trial_results['iterations'].append(iter_results)
        trial_results['prev_weights'] = model.trainer.best
        
        print(f"  ✅ {split_name}: Val mAP@0.5 = {iter_results['val_map50']:.4f} (out-of-sample!)")
    
    with open(f'v2_results_trial_{seed}.json', 'w') as f:
        json.dump(trial_results, f, indent=2)
    
    all_trials.append(trial_results)

# Aggregate
print(f"\n{'='*70}")
print("Aggregated Out-of-Sample Results")
print(f"{'='*70}\n")

iterations = ['baseline', 'iter1', 'iter2', 'iter3', 'full']
aggregated = {}

for iter_name in iterations:
    map50_values = []
    for trial in all_trials:
        for result in trial['iterations']:
            if result['iteration'] == iter_name:
                map50_values.append(result['val_map50'])
                break
    
    aggregated[iter_name] = {
        'mean': np.mean(map50_values),
        'std': np.std(map50_values),
        'values': map50_values
    }
    
    print(f"{iter_name:10s}: mAP@0.5 = {aggregated[iter_name]['mean']:.4f} ± {aggregated[iter_name]['std']:.4f} (OUT-OF-SAMPLE)")

# Save and plot
with open('v2_aggregated_results.json', 'w') as f:
    json.dump({'trials': all_trials, 'aggregated': aggregated, 'metadata': {'validation': 'out-of-sample', 'val_split': args.val_split}}, f, indent=2)

# Plot
fig, ax = plt.subplots(figsize=(12, 8))
train_sizes = [50, 100, 200, 300, 800]
means = [aggregated[it]['mean'] for it in iterations]
stds = [aggregated[it]['std'] for it in iterations]

ax.errorbar(train_sizes, means, yerr=stds, marker='o', linewidth=3, markersize=12, 
            capsize=10, capthick=2, color='#2E86AB', elinewidth=2, alpha=0.9,
            label=f'YOLO11 (n={args.trials}, out-of-sample)')
ax.fill_between(train_sizes, [m-s for m,s in zip(means,stds)], [m+s for m,s in zip(means,stds)], alpha=0.2, color='#2E86AB')

for x, m, s in zip(train_sizes, means, stds):
    ax.annotate(f'{m:.3f}±{s:.3f}', (x, m), textcoords="offset points", xytext=(0,15), ha='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Number of Training Images', fontsize=14, fontweight='bold')
ax.set_ylabel('mAP@0.5 (Out-of-Sample Validation)', fontsize=14, fontweight='bold')
ax.set_title(f'HITL: Out-of-Sample Performance vs Training Size\n({args.trials} trials, {args.val_split*100:.0f}% held-out validation)', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=12)

plt.tight_layout()
plt.savefig('v2_publication_plot_out_of_sample.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ v2 Complete: True out-of-sample validation!")
print("   Files: v2_results_trial_*.json, v2_aggregated_results.json, v2_publication_plot_out_of_sample.png")
