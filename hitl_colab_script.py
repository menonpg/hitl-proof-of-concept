#!/usr/bin/env python3
"""
HITL Proof-of-Concept Script for Colab
Run this in Colab to prove incremental learning works
"""

# Cell 1: Check GPU
print("Checking GPU...")
import subprocess
subprocess.run(['nvidia-smi'])

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

# Cell 2: Install
print("\nInstalling packages...")
subprocess.run(['pip', 'install', '-q', 'ultralytics', 'roboflow', 'matplotlib'])

# Cell 3: Download data
from roboflow import Roboflow

print("\nDownloading dataset...")
rf = Roboflow(api_key="lbXALpBLK1UO9TLPqobo")
project = rf.workspace("sofia-valdivieso-von-teuber").project("insulators-wo6lb")
dataset = project.version(3).download("coco")
DATASET_ROOT = dataset.location
print(f"Downloaded to: {DATASET_ROOT}")

# Cell 4: Prepare data
import json
import random
import shutil
from pathlib import Path

random.seed(42)

dataset_path = Path(DATASET_ROOT)
with open(dataset_path / 'train' / '_annotations.coco.json') as f:
    data = json.load(f)

# CRITICAL FIX: Filter to insulators ONLY (category_id == 1)
insulator_anns = [a for a in data['annotations'] if a['category_id'] == 1]
ann_map = {}
for ann in insulator_anns:
    if ann['image_id'] not in ann_map:
        ann_map[ann['image_id']] = []
    ann_map[ann['image_id']].append(ann)

print(f"\nFiltered to {len(insulator_anns)} insulator annotations")
print(f"From {len(ann_map)} images with insulators")

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

splits_dir = Path('/content/splits')
splits_dir.mkdir(exist_ok=True)

print("\nCreating splits...")
for name, size in splits.items():
    sp = splits_dir / name
    (sp / 'images').mkdir(parents=True, exist_ok=True)
    (sp / 'labels').mkdir(parents=True, exist_ok=True)
    
    labels = 0
    for img_id in all_ids[:size]:
        img = img_map[img_id]
        src = dataset_path / 'train' / img['file_name']
        if src.exists():
            shutil.copy2(src, sp / 'images' / img['file_name'])
            
            # Write YOLO labels (class 0 for all insulators)
            with open(sp / 'labels' / (Path(img['file_name']).stem + '.txt'), 'w') as f:
                for ann in ann_map[img_id]:
                    x, y, w, h = ann['bbox']
                    xc = (x + w/2) / img['width']
                    yc = (y + h/2) / img['height']
                    wn = w / img['width']
                    hn = h / img['height']
                    f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                    labels += 1
    
    # Create data.yaml
    with open(sp / 'data.yaml', 'w') as f:
        f.write(f"""path: /content
train: splits/{name}
val: splits/{name}

names:
  0: insulators

nc: 1
""")
    
    print(f"  {name:10s}: {size:4d} images, {labels:4d} labels")

print("\n✅ Data ready!")

# Cell 5: Train all iterations
from ultralytics import YOLO
from datetime import datetime

results = []
prev_w = None
start_time = datetime.now()

for name in ['baseline', 'iter1', 'iter2', 'iter3', 'full']:
    w = prev_w if prev_w else 'yolo11n.pt'
    e = 50 if name in ['baseline', 'full'] else 30
    
    print(f"\n{'='*70}")
    print(f"Training: {name} ({e} epochs)")
    print(f"{'='*70}")
    
    m = YOLO(w)
    m.train(
        data=str(splits_dir / name / 'data.yaml'),
        epochs=e,
        imgsz=640,
        batch=16,
        device=0,
        name=name,
        patience=10
    )
    
    met = m.val()
    results.append({
        'iter': name,
        'map50': float(met.box.map50),
        'prec': float(met.box.mp),
        'rec': float(met.box.mr)
    })
    
    print(f"\n✅ {name}: mAP@0.5 = {met.box.map50:.4f}")
    prev_w = m.trainer.best

duration = (datetime.now() - start_time).total_seconds()
print(f"\n⏱️  Total time: {duration/3600:.2f} hours")

# Cell 6: Visualize
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 7))
scores = [r['map50'] for r in results]
ax.plot(scores, marker='o', linewidth=3, markersize=10, color='#2E86AB')

for i, s in enumerate(scores):
    ax.annotate(f'{s:.3f}', (i, s), textcoords='offset points',
               xytext=(0,10), ha='center', fontweight='bold')

ax.set_xlabel('Iteration', fontweight='bold')
ax.set_ylabel('mAP@0.5', fontweight='bold')
ax.set_title('HITL Proof: mAP@0.5 Improvement', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(results)))
ax.set_xticklabels([r['iter'] for r in results])
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hitl_results.png', dpi=300)
plt.show()

print("\n📊 Final Results:")
print("-" * 50)
for r in results:
    print(f"{r['iter']:10s}: mAP@0.5 = {r['map50']:.4f}")

print(f"\n✅ Improvement: {scores[0]:.4f} → {scores[-1]:.4f} (+{((scores[-1]-scores[0])/scores[0]*100):.1f}%)")

# Download results
from google.colab import files
files.download('hitl_results.png')

print("\n🎉 HITL Proof Complete!")
