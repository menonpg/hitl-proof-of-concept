#!/usr/bin/env python3
"""
HITL Statistical Validation: Multiple Trials with Different Folds
==================================================================
Runs YOLO11 HITL experiment 3+ times with different random seeds
to get mean ± std error bars for publication-quality results.

Usage:
    python hitl_statistical_validation.py --trials 3

Output:
    - results_trial_*.json for each trial
    - aggregated_results.json with mean/std
    - publication_plot_with_errorbars.png
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Intelligent device detection (CUDA → MPS → CPU)
import torch

def get_best_device():
    """Detect best available device: CUDA → MPS → CPU"""
    if torch.cuda.is_available():
        device = 0  # CUDA GPU
        device_name = torch.cuda.get_device_name(0)
        batch_size = 16
        print(f"🚀 Using CUDA GPU: {device_name}")
    elif torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon GPU
        device_name = "Apple Silicon (MPS)"
        batch_size = 16
        print(f"🍎 Using Apple Silicon GPU (MPS)")
    else:
        device = 'cpu'
        device_name = "CPU"
        batch_size = 8  # Reduce batch size for CPU
        print(f"⚠️  Using CPU (will be slow!)")
    
    print(f"   Device: {device_name}")
    print(f"   Batch size: {batch_size}")
    return device, batch_size

# Detect device
DEVICE, BATCH_SIZE = get_best_device()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=3, help='Number of trials to run')
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456], help='Random seeds')
parser.add_argument('--device', type=str, default=None, help='Override device (cuda, mps, cpu)')
parser.add_argument('--batch', type=int, default=None, help='Override batch size')
args = parser.parse_args()

# Allow command-line override
if args.device:
    DEVICE = args.device
    print(f"   Overridden to: {DEVICE}")
if args.batch:
    BATCH_SIZE = args.batch
    print(f"   Batch size overridden to: {BATCH_SIZE}")

print(f"\n{'='*70}")
print(f"HITL Statistical Validation: {args.trials} Trials")
print(f"{'='*70}\n")

# Get dataset (assume already downloaded)
from roboflow import Roboflow

DATASET_ROOT = "/content/insulators-3"
if not Path(DATASET_ROOT).exists():
    print("Downloading dataset...")
    rf = Roboflow(api_key="lbXALpBLK1UO9TLPqobo")
    project = rf.workspace("sofia-valdivieso-von-teuber").project("insulators-wo6lb")
    dataset = project.version(3).download("coco")
    DATASET_ROOT = dataset.location

dataset_path = Path(DATASET_ROOT)

# Load annotations once
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

# Storage for all trials
all_trials = []

# Run multiple trials
for trial_idx in range(args.trials):
    seed = args.seeds[trial_idx] if trial_idx < len(args.seeds) else 100 + trial_idx
    
    print(f"\n{'='*70}")
    print(f"Trial {trial_idx + 1}/{args.trials} (seed={seed})")
    print(f"{'='*70}\n")
    
    # Set random seed for this trial
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle image IDs with this seed
    trial_image_ids = all_image_ids.copy()
    random.shuffle(trial_image_ids)
    
    # Create splits for this trial
    splits = {
        'baseline': 50,
        'iter1': 100,
        'iter2': 200,
        'iter3': 300,
        'full': min(800, len(trial_image_ids))
    }
    
    splits_dir = Path(f'/content/splits_trial_{seed}')
    splits_dir.mkdir(exist_ok=True)
    
    # Prepare data for this trial
    print(f"Preparing data with seed {seed}...")
    for split_name, split_size in splits.items():
        sp = splits_dir / split_name
        (sp / 'images').mkdir(parents=True, exist_ok=True)
        (sp / 'labels').mkdir(parents=True, exist_ok=True)
        
        labels_created = 0
        for img_id in trial_image_ids[:split_size]:
            if img_id not in img_map:
                continue
            
            img_info = img_map[img_id]
            src = dataset_path / 'train' / img_info['file_name']
            
            if src.exists():
                shutil.copy2(src, sp / 'images' / img_info['file_name'])
                
                # Write YOLO labels
                with open(sp / 'labels' / (Path(img_info['file_name']).stem + '.txt'), 'w') as f:
                    if img_id in ann_map_base:
                        for ann in ann_map_base[img_id]:
                            x, y, w, h = ann['bbox']
                            xc = (x + w/2) / img_info['width']
                            yc = (y + h/2) / img_info['height']
                            wn = w / img_info['width']
                            hn = h / img_info['height']
                            f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                            labels_created += 1
        
        # Create data.yaml
        with open(sp / 'data.yaml', 'w') as f:
            f.write(f"""path: /content
train: splits_trial_{seed}/{split_name}
val: splits_trial_{seed}/{split_name}

names:
  0: insulators

nc: 1
""")
        
        print(f"  {split_name:10s}: {split_size:4d} images, {labels_created:4d} labels")
    
    # Train all iterations for this trial
    from ultralytics import YOLO
    
    trial_results = {'seed': seed, 'iterations': []}
    prev_weights = None
    
    for iter_name in ['baseline', 'iter1', 'iter2', 'iter3', 'full']:
        w = prev_weights if prev_weights else 'yolo11n.pt'
        e = 50 if iter_name in ['baseline', 'full'] else 30
        
        print(f"\n[Trial {trial_idx+1}] Training {iter_name}...")
        
        model = YOLO(w)
        model.train(
            data=str(splits_dir / iter_name / 'data.yaml'),
            epochs=e,
            imgsz=640,
            batch=BATCH_SIZE,  # Use detected batch size
            device=DEVICE,  # Use detected device (cuda/mps/cpu)
            name=f'{iter_name}_trial{seed}',
            patience=10,
            verbose=False
        )
        
        metrics = model.val()
        
        iter_results = {
            'iteration': iter_name,
            'map50': float(metrics.box.map50),
            'map50_95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr)
        }
        
        trial_results['iterations'].append(iter_results)
        print(f"  ✅ {iter_name}: mAP@0.5 = {iter_results['map50']:.4f}")
        
        prev_weights = model.trainer.best
    
    # Save trial results
    with open(f'results_trial_{seed}.json', 'w') as f:
        json.dump(trial_results, f, indent=2)
    
    all_trials.append(trial_results)
    print(f"\n✅ Trial {trial_idx + 1} complete!")

# Aggregate results
print(f"\n{'='*70}")
print("Aggregating Results Across Trials")
print(f"{'='*70}\n")

iterations = ['baseline', 'iter1', 'iter2', 'iter3', 'full']
aggregated = {}

for iter_name in iterations:
    # Collect metrics across all trials
    map50_values = [trial['iterations'][i]['map50'] for i, trial in enumerate(all_trials) for j, it in enumerate(trial['iterations']) if it['iteration'] == iter_name]
    
    # Calculate mean and std
    aggregated[iter_name] = {
        'map50_mean': np.mean(map50_values),
        'map50_std': np.std(map50_values),
        'map50_values': map50_values
    }
    
    print(f"{iter_name:10s}: mAP@0.5 = {aggregated[iter_name]['map50_mean']:.4f} ± {aggregated[iter_name]['map50_std']:.4f}")

# Save aggregated results
with open('aggregated_results.json', 'w') as f:
    json.dump({
        'trials': all_trials,
        'aggregated': {k: {'mean': v['map50_mean'], 'std': v['map50_std']} for k, v in aggregated.items()},
        'metadata': {
            'num_trials': args.trials,
            'seeds': args.seeds[:args.trials],
            'timestamp': datetime.now().isoformat()
        }
    }, f, indent=2)

# Create publication-quality plot with error bars
fig, ax = plt.subplots(figsize=(12, 8))

x = range(len(iterations))
means = [aggregated[it]['map50_mean'] for it in iterations]
stds = [aggregated[it]['map50_std'] for it in iterations]

# Plot line with error bars
ax.errorbar(x, means, yerr=stds, marker='o', linewidth=3, markersize=12, 
            capsize=10, capthick=2, color='#2E86AB', label='YOLO11', 
            elinewidth=2, alpha=0.9)

# Add value labels
for i, (mean, std) in enumerate(zip(means, stds)):
    ax.annotate(f'{mean:.3f}±{std:.3f}', 
               (i, mean), 
               textcoords="offset points",
               xytext=(0,15), 
               ha='center',
               fontsize=10,
               fontweight='bold')

# Styling
ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
ax.set_ylabel('mAP@0.5', fontsize=14, fontweight='bold')
ax.set_title(f'HITL Training: mAP@0.5 with Error Bars ({args.trials} Trials)', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(iterations)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=12)

# Add shaded region for ±1 std
ax.fill_between(x, 
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.2, color='#2E86AB')

plt.tight_layout()
plt.savefig('publication_plot_with_errorbars.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✅ Statistical validation complete!")
print(f"   Results saved to:")
print(f"   - aggregated_results.json")
print(f"   - publication_plot_with_errorbars.png")
print(f"   - results_trial_*.json (individual trials)")

# Statistical significance testing
print(f"\n{'='*70}")
print("Statistical Significance Testing")
print(f"{'='*70}\n")

from scipy import stats

# Compare baseline vs final
baseline_values = aggregated['baseline']['map50_values']
final_values = aggregated['full']['map50_values']

t_stat, p_value = stats.ttest_rel(final_values, baseline_values)

print(f"Paired t-test: Baseline vs. Full")
print(f"  Baseline: {np.mean(baseline_values):.4f} ± {np.std(baseline_values):.4f}")
print(f"  Full:     {np.mean(final_values):.4f} ± {np.std(final_values):.4f}")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")

if p_value < 0.001:
    print(f"  ✅ HIGHLY SIGNIFICANT (p < 0.001)")
elif p_value < 0.01:
    print(f"  ✅ VERY SIGNIFICANT (p < 0.01)")
elif p_value < 0.05:
    print(f"  ✅ SIGNIFICANT (p < 0.05)")
else:
    print(f"  ⚠️  Not significant (p = {p_value:.4f})")

print(f"\n🎉 Ready for manuscript submission with statistical validation!")
