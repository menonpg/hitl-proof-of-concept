#!/usr/bin/env python3
"""
Step 4: Train All HITL Iterations
==================================
Trains YOLOv11 models for each iteration with transfer learning.

Iterations:
- Baseline: Train from scratch on 50 images (50 epochs)
- Iter1: Transfer learn on 100 images (30 epochs)
- Iter2: Transfer learn on 200 images (30 epochs)
- Iter3: Transfer learn on 300 images (30 epochs)
- Full: Transfer learn on all images (50 epochs)

Usage:
    python 04_train_all_iterations.py [--skip-baseline] [--skip-iter1] etc.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime


def train_iteration(iteration_name, data_yaml, epochs, weights='yolo11n.pt', device=0, batch=16):
    """
    Train one iteration using ultralytics YOLO.
    
    Args:
        iteration_name: Name of iteration (baseline, iter1, etc.)
        data_yaml: Path to data.yaml file
        epochs: Number of training epochs
        weights: Model weights to start from (for transfer learning)
        device: GPU device (0, 1, ...) or 'cpu'
        batch: Batch size
    
    Returns:
        Path to best model weights
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed!")
        print("Installing...")
        import os
        os.system("pip install -q ultralytics")
        from ultralytics import YOLO
    
    print(f"\n{'='*70}")
    print(f"Training: {iteration_name}")
    print(f"{'='*70}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Weights: {weights}")
    print(f"Device: {device}")
    print(f"Batch: {batch}")
    print()
    
    # Initialize model
    model = YOLO(weights)
    
    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=device,
        name=iteration_name,
        patience=10,
        save=True,
        plots=True,
        verbose=True,
        project='runs/detect'
    )
    
    # Get path to best weights
    best_weights = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
    
    print(f"\n✅ Training complete: {iteration_name}")
    print(f"   Best weights: {best_weights}")
    
    return best_weights


def evaluate_iteration(iteration_name, data_yaml, weights, device=0):
    """
    Evaluate model on test set.
    
    Args:
        iteration_name: Name of iteration
        data_yaml: Path to data.yaml file  
        weights: Path to model weights
        device: GPU device
    
    Returns:
        Dict with evaluation metrics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    
    print(f"\n📊 Evaluating: {iteration_name}")
    
    model = YOLO(weights)
    results = model.val(data=str(data_yaml), device=device)
    
    # Extract metrics
    metrics = {
        'iteration': iteration_name,
        'map50': float(results.box.map50),
        'map50_95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
        'weights_path': str(weights)
    }
    
    print(f"   mAP@0.5: {metrics['map50']:.4f}")
    print(f"   mAP@0.5:0.95: {metrics['map50_95']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    
    return metrics


def main():
    """Train all iterations."""
    parser = argparse.ArgumentParser(description='Train all HITL iterations')
    parser.add_argument('--skip-baseline', action='store_true', help='Skip baseline training')
    parser.add_argument('--skip-iter1', action='store_true', help='Skip iteration 1')
    parser.add_argument('--skip-iter2', action='store_true', help='Skip iteration 2')
    parser.add_argument('--skip-iter3', action='store_true', help='Skip iteration 3')
    parser.add_argument('--skip-full', action='store_true', help='Skip full training')
    parser.add_argument('--device', type=str, default='0', help='GPU device (0, 1, ...) or cpu')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("HITL Proof-of-Concept: Step 4 - Train All Iterations")
    print("="*70 + "\n")
    
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    splits_dir = data_dir / "splits"
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Check if splits exist
    if not splits_dir.exists():
        print(f"❌ Splits directory not found: {splits_dir}")
        print("\n⚠️  Run previous steps first:")
        print("  python 01_download_dataset.py")
        print("  python 02_split_dataset.py")
        print("  python 03_convert_to_yolo.py")
        return 1
    
    # Define training iterations
    iterations = [
        {
            'name': 'baseline',
            'data_yaml': splits_dir / 'baseline' / 'data.yaml',
            'epochs': 50,
            'weights': 'yolo11n.pt',  # From scratch
            'skip': args.skip_baseline
        },
        {
            'name': 'iter1',
            'data_yaml': splits_dir / 'iter1' / 'data.yaml',
            'epochs': 30,
            'weights': None,  # Will be set to previous iteration's weights
            'skip': args.skip_iter1
        },
        {
            'name': 'iter2',
            'data_yaml': splits_dir / 'iter2' / 'data.yaml',
            'epochs': 30,
            'weights': None,
            'skip': args.skip_iter2
        },
        {
            'name': 'iter3',
            'data_yaml': splits_dir / 'iter3' / 'data.yaml',
            'epochs': 30,
            'weights': None,
            'skip': args.skip_iter3
        },
        {
            'name': 'full',
            'data_yaml': splits_dir / 'full' / 'data.yaml',
            'epochs': 50,
            'weights': None,
            'skip': args.skip_full
        }
    ]
    
    all_metrics = []
    prev_weights = None
    
    print("🚀 Starting HITL training pipeline...")
    print(f"   Device: {args.device}")
    print(f"   Batch size: {args.batch}\n")
    
    start_time = datetime.now()
    
    # Train each iteration
    for iter_config in iterations:
        if iter_config['skip']:
            print(f"\n⏭️  Skipping {iter_config['name']}")
            continue
        
        # Use previous iteration's weights for transfer learning
        if prev_weights is not None and iter_config['weights'] is None:
            iter_config['weights'] = prev_weights
        
        # Check if data.yaml exists
        if not iter_config['data_yaml'].exists():
            print(f"\n⚠️  Skipping {iter_config['name']}: data.yaml not found")
            continue
        
        # Train
        try:
            best_weights = train_iteration(
                iteration_name=iter_config['name'],
                data_yaml=iter_config['data_yaml'],
                epochs=iter_config['epochs'],
                weights=iter_config['weights'],
                device=args.device,
                batch=args.batch
            )
            
            # Evaluate
            metrics = evaluate_iteration(
                iteration_name=iter_config['name'],
                data_yaml=iter_config['data_yaml'],
                weights=best_weights,
                device=args.device
            )
            
            if metrics:
                all_metrics.append(metrics)
            
            # Save weights path for next iteration
            prev_weights = best_weights
            
        except Exception as e:
            print(f"\n❌ Error training {iter_config['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Save results
    results_file = results_dir / 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration,
            'iterations': all_metrics,
            'summary': {
                'total_iterations': len(all_metrics),
                'baseline_map50': all_metrics[0]['map50'] if all_metrics else 0,
                'final_map50': all_metrics[-1]['map50'] if all_metrics else 0,
                'improvement': all_metrics[-1]['map50'] - all_metrics[0]['map50'] if len(all_metrics) >= 2 else 0
            }
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("✅ Step 4 Complete: All Iterations Trained")
    print("="*70)
    print(f"\n⏱️  Total training time: {duration/3600:.2f} hours")
    print(f"📊 Results saved to: {results_file}")
    
    if all_metrics:
        print("\n📈 Performance Summary:")
        print(f"{'Iteration':12s} {'mAP@0.5':>10s} {'Improvement':>12s}")
        print("-" * 36)
        baseline_map = all_metrics[0]['map50']
        for m in all_metrics:
            improvement = m['map50'] - baseline_map
            improvement_pct = (improvement / baseline_map * 100) if baseline_map > 0 else 0
            print(f"{m['iteration']:12s} {m['map50']:10.4f} {improvement:+11.4f} ({improvement_pct:+.1f}%)")
    
    print("\n📋 Next step: Run python 05_evaluate_and_plot.py")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
