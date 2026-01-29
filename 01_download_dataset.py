#!/usr/bin/env python3
"""
Step 1: Download Dataset from Roboflow
=======================================
Downloads the insulator dataset for HITL proof-of-concept experiment.

Usage:
    python 01_download_dataset.py
    
Environment Variable:
    ROBOFLOW_API_KEY: Your Roboflow API key (get from https://app.roboflow.com/settings/api)
"""

import os
import sys
from pathlib import Path


def main():
    """Download dataset from Roboflow."""
    print("\n" + "="*70)
    print("HITL Proof-of-Concept: Step 1 - Download Dataset")
    print("="*70 + "\n")
    
    # Check for API key
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    
    if not api_key:
        print("❌ ROBOFLOW_API_KEY not set!")
        print("\n📋 To set it:")
        print("  export ROBOFLOW_API_KEY='your_api_key_here'")
        print("\n🔑 Get your API key from: https://app.roboflow.com/settings/api")
        print("\n⚠️  Alternatively, download manually:")
        print("  1. Visit: https://universe.roboflow.com/sofia-valdivieso-von-teuber/insulators-wo6lb/dataset/3")
        print("  2. Click 'Download Dataset'")
        print("  3. Select 'COCO JSON' format")
        print("  4. Extract to: HITL-proof/data/raw/")
        return 1
    
    # Try to import roboflow
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ Roboflow package not installed!")
        print("Installing...")
        os.system("pip install -q roboflow")
        from roboflow import Roboflow
    
    # Create directories
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Raw data directory: {raw_dir}\n")
    
    # Download dataset
    print("📥 Downloading insulators dataset from Roboflow...")
    print(f"   Using API key: {api_key[:8]}...")
    
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("sofia-valdivieso-von-teuber").project("insulators-wo6lb")
        dataset = project.version(3).download("coco", location=str(raw_dir))
        
        print(f"\n✅ Dataset downloaded successfully!")
        print(f"   Location: {raw_dir}")
        
        # Check what was downloaded
        downloaded_files = list(raw_dir.rglob("*"))
        print(f"   Files downloaded: {len(downloaded_files)}")
        
        # Look for train/valid/test splits
        train_dir = raw_dir / "train"
        valid_dir = raw_dir / "valid"
        test_dir = raw_dir / "test"
        
        if train_dir.exists():
            train_images = len(list(train_dir.glob("*.jpg"))) + len(list(train_dir.glob("*.png")))
            print(f"   Train images: {train_images}")
        
        if valid_dir.exists():
            valid_images = len(list(valid_dir.glob("*.jpg"))) + len(list(valid_dir.glob("*.png")))
            print(f"   Valid images: {valid_images}")
        
        if test_dir.exists():
            test_images = len(list(test_dir.glob("*.jpg"))) + len(list(test_dir.glob("*.png")))
            print(f"   Test images: {test_images}")
        
        print("\n" + "="*70)
        print("✅ Step 1 Complete: Dataset Downloaded")
        print("="*70)
        print("\n📋 Next step: Run python 02_split_dataset.py")
        return 0
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\n⚠️  Try manual download:")
        print("  1. Visit: https://universe.roboflow.com/sofia-valdivieso-von-teuber/insulators-wo6lb/dataset/3")
        print("  2. Click 'Download Dataset'")
        print("  3. Select 'COCO JSON' format")
        print("  4. Extract to: HITL-proof/data/raw/")
        return 1


if __name__ == "__main__":
    sys.exit(main())
