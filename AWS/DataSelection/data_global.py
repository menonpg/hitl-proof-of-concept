import os
import json
import random
import copy
import shutil
import filecmp
import hashlib
from enum import Enum
from pathlib import Path
from datetime import date
from collections import defaultdict

#compression
import zipfile
import gzip
import zlib
import lzma
#from PIL import Image
#import cv2
#import brotli
#import zstandard as zstd
#import blosc
#from coco_compress import compress_coco, decompress_coco
#import pyarrow as pa
#import pyarrow.parquet as pq

#DIRECTORIES

#DIRECTORIES-CREATE : 
#DIRECTORIES-READ : 
#DIRECTORIES-UPDATE : 
#DIRECTORIES-DELETE : 
#S3-CREATE : 
#S3-READ : 
#S3-UPDATE : 
#S3-DELETE : 

#DATASETS
class DATASETS:

    class MODALITY(Enum):
        IMAGE = 0

    class TYPE(Enum):
        json_COCO = 0
        json_COCO_MMDetection = 1
        json_CreateML = 2
        json_PaliGemma = 3
        json_Florence2_Object_Detection = 4
        json_OpenAI = 5
        xml_PascalVOC = 6
        txt_YOLO_Darknet = 7
        txt_YOLO_v3_Keras = 8
        txt_YOLO_v4_PyTorch = 9
        txt_Scaled_YOLOv4 = 10
        txt_YOLOv5_Oriented_Bounding_Boxes = 11
        txt_meituan_YOLOv6 = 12
        txt_YOLOv5_PyTorch = 13
        txt_YOLOv7_PyTorch = 14
        txt_YOLOv8 = 15
        txt_YOLOv8_Oriented_Bounding_Boxes = 16
        txt_YOLOv9 = 17
        txt_YOLOv11 = 18
        txt_YOLOv12 = 19
        csv_Tensorflow_Object_Detection = 20
        csv_RetinaNet_Keras = 21
        csv_Multi_Label_Classification = 22
        other_OpenAI_CLIP_Classification = 23
        other_Tensorflow_TFRecord = 24

    class STATE(Enum):
        RAW = 0
        SPLIT = 1
        CONSOLIDATED = 2
        ALL = 3
        ARCHIVE = 4

    class CREATE:
        #create an unpopulated dataset or make a dataset from another either to version an existing dataset or to make a dataset from a template
        def json_COCO_split(input_dir="",
                                    output_dir="",
                                    dataset_name=None,
                                    dataset_description=None,
                                    dataset_version=None,
                                    dataset_year=None,
                                    dataset_contributer=None,
                                    dataset_url=None):
            """
            Create an empty COCO dataset structure or copy and edit that copy

            Args:
                input_dir: Directory containing input dataset to set default initial values
                output_dir: Directory to create the dataset in
                dataset_name: Name of the dataset
                dataset_description: Description of the dataset
                dataset_version: Version of the dataset
                dataset_year: Year of the dataset
                dataset_contributer: Contributor of the dataset
                dataset_url: URL for the dataset
            """
            # Validate required parameters
            if not output_dir:
                raise ValueError("Output directory must be specified")
            if not dataset_name:
                raise ValueError("Dataset name must be specified")
            output_dir = Path(output_dir) / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            # Create empty directories for splits
            for split in ["train", "valid", "test"]:
                (output_dir / split).mkdir(exist_ok=True)
            # If no input directory provided, create empty dataset
            if not input_dir:
                empty_coco = {
                    "info": {
                        "description": dataset_description or "Empty COCO dataset",
                        "version": str(dataset_version) or "0.0",
                        "year": dataset_year or date.year,
                        "contributor": dataset_contributer or "",
                        "url": dataset_url or ""
                    },
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": []
                }
                for split in ["train", "valid", "test"]:
                    json_path = output_dir / split / "_annotations.coco.json"
                    with open(json_path, 'w') as f:
                        json.dump(empty_coco, f, indent=2)
                print(f"Created empty COCO dataset at: {output_dir}")
                return
            # If input directory provided, copy and modify
            input_dir = Path(input_dir)
    
            # If input is a file, check if it's the COCO JSON file
            if input_dir.is_file():
                is_coco_file = input_dir.name == "_annotations.coco.json"

            # If input is a directory, check for the file
            if input_dir.is_dir():
                coco_file = input_dir / "_annotations.coco.json"
                contains_coco_file = coco_file.exists()
    
            # Load each split with proper filename
            try:
                with open(input_dir / "train" / "_annotations.coco.json", 'r') as f:
                    coco_data_train = json.load(f)
                with open(input_dir / "valid" / "_annotations.coco.json", 'r') as f:
                    coco_data_val = json.load(f)
                with open(input_dir / "test" / "_annotations.coco.json", 'r') as f:
                    coco_data_test = json.load(f)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"COCO JSON file not found: {e}")
            # Create deep copies
            modified_data_train = copy.deepcopy(coco_data_train)
            modified_data_val = copy.deepcopy(coco_data_val)
            modified_data_test = copy.deepcopy(coco_data_test)
            # Apply modifications to all splits
            if dataset_description is not None:
                for data in [modified_data_train, modified_data_val, modified_data_test]:
                    data['info']['description'] = dataset_description
            if dataset_version is not None:
                for data in [modified_data_train, modified_data_val, modified_data_test]:
                    data['info']['version'] = str(dataset_version)
            if dataset_year is not None:
                for data in [modified_data_train, modified_data_val, modified_data_test]:
                    data['info']['year'] = dataset_year
            if dataset_contributer is not None:
                for data in [modified_data_train, modified_data_val, modified_data_test]:
                    data['info']['contributor'] = dataset_contributer
            if dataset_url is not None:
                for data in [modified_data_train, modified_data_val, modified_data_test]:
                    data['info']['url'] = dataset_url
            print("Copying image files from input directory...")
            for split in ["train", "valid", "test"]:
                # Find all image files in the split directory
                image_files = list((input_dir / split).glob("*.jpg")) + \
                            list((input_dir / split).glob("*.png")) + \
                            list((input_dir / split).glob("*.jpeg"))
                # Copy each image file
                for img_file in image_files:
                    dest_img = output_dir / split / img_file.name
                    shutil.copy2(img_file, dest_img)
                    print(f"Copied: {img_file} -> {dest_img}")
            # Save the modified datasets
            for split, data in [("train", modified_data_train),
                                ("valid", modified_data_val),
                                ("test", modified_data_test)]:
                json_path = output_dir / split / "_annotations.coco.json"
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
            print(f"COCO dataset saved to {output_dir}")

        def json_COCO_split_v0(input_dir="",
                                              output_dir="",
                                              dataset_name=None,
                                              dataset_description=None,
                                              dataset_version=None,
                                              dataset_year=None,
                                              dataset_contributer=None,
                                              dataset_url=None):
            """
            Create an empty COCO dataset structure or copy and edit that copy.
            Handles missing directories gracefully.

            Args:
                input_dir: Directory containing input dataset to set default initial values
                output_dir: Directory to create the dataset in
                dataset_name: Name of the dataset
                dataset_description: Description of the dataset
                dataset_version: Version of the dataset
                dataset_year: Year of the dataset
                dataset_contributer: Contributor of the dataset
                dataset_url: URL for the dataset
            """
            # Validate required parameters
            if not output_dir:
                raise ValueError("Output directory must be specified")
            if not dataset_name:
                raise ValueError("Dataset name must be specified")

            # Create output directory structure
            output_dir = Path(output_dir) / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create empty directories for splits
            for split in ["train", "valid", "test"]:
                (output_dir / split).mkdir(exist_ok=True)

            # If no input directory provided, create empty dataset
            if not input_dir:
                empty_coco = {
                    "info": {
                        "description": dataset_description or "Empty COCO dataset",
                        "version": str(dataset_version) or "0.0",
                        "year": dataset_year or date.year,
                        "contributor": dataset_contributer or "",
                        "url": dataset_url or ""
                    },
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": []
                }

                for split in ["train", "valid", "test"]:
                    json_path = output_dir / split / "_annotations.coco.json"
                    with open(json_path, 'w') as f:
                        json.dump(empty_coco, f, indent=2)

                print(f"Created empty COCO dataset at: {output_dir}")
                return

            # If input directory provided, copy and modify
            input_dir = Path(input_dir)

            # Initialize data structures with defaults
            data_structures = {
                "train": None,
                "valid": None,
                "test": None
            }

            # Try to load each split
            for split in ["train", "valid", "test"]:
                try:
                    json_path = input_dir / split / "_annotations.coco.json"
                    if json_path.exists():
                        with open(json_path, 'r') as f:
                            data_structures[split] = json.load(f)
                        print(f"Loaded {split} split from {json_path}")
                    else:
                        print(f"Warning: {json_path} not found. Creating empty {split} split.")
                        data_structures[split] = {
                            "info": {
                                "description": dataset_description or "Empty COCO dataset",
                                "version": str(dataset_version) or "0.0",
                                "year": dataset_year or date.year,
                                "contributor": dataset_contributer or "",
                                "url": dataset_url or ""
                            },
                            "licenses": [],
                            "images": [],
                            "annotations": [],
                            "categories": []
                        }
                except Exception as e:
                    print(f"Error loading {split} split: {e}. Using empty structure.")
                    data_structures[split] = {
                        "info": {
                            "description": dataset_description or "Empty COCO dataset",
                            "version": str(dataset_version) or "0.0",
                            "year": dataset_year or date.year,
                            "contributor": dataset_contributer or "",
                            "url": dataset_url or ""
                        },
                        "licenses": [],
                        "images": [],
                        "annotations": [],
                        "categories": []
                    }

            # Apply modifications to all splits
            for split, data in data_structures.items():
                if dataset_description is not None:
                    data['info']['description'] = dataset_description
                if dataset_version is not None:
                    data['info']['version'] = str(dataset_version)
                if dataset_year is not None:
                    data['info']['year'] = dataset_year
                if dataset_contributer is not None:
                    data['info']['contributor'] = dataset_contributer
                if dataset_url is not None:
                    data['info']['url'] = dataset_url

            # Copy image files if directories exist
            for split in ["train", "valid", "test"]:
                input_split_dir = input_dir / split
                output_split_dir = output_dir / split

                if input_split_dir.exists() and input_split_dir.is_dir():
                    print(f"Copying image files from {split} directory...")
                    # Find all image files in the split directory
                    image_files = list(input_split_dir.glob("*.jpg")) + \
                                list(input_split_dir.glob("*.png")) + \
                                list(input_split_dir.glob("*.jpeg"))

                    # Copy each image file
                    for img_file in image_files:
                        dest_img = output_split_dir / img_file.name
                        shutil.copy2(img_file, dest_img)
                        print(f"Copied: {img_file} -> {dest_img}")
                else:
                    print(f"Warning: {input_split_dir} not found. Skipping image copy.")

            # Save the modified datasets to the correct output directories
            for split, data in data_structures.items():
                json_path = output_dir / split / "_annotations.coco.json"
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Saved {split} split to {json_path}")

            print(f"COCO dataset successfully created at: {output_dir}")

        """
        json_COCO_split(
            input_dir="",
            output_dir="",
            dataset_name="",
            dataset_description="",
            dataset_version=1.0,
            dataset_year=2026,
            dataset_contributer="",
            dataset_url=""
        )
        """

        """
        # Example usage with fault tolerance
        json_COCO_split_v0(
            input_dir="",
            output_dir="",
            dataset_name="",
            dataset_description="",
            dataset_version=1.0,
            dataset_year=2026,
            dataset_contributer="",
            dataset_url=""
        )
        """

        def json_COCO(input_dir="",
                     output_dir="",
                     dataset_name=None,
                     dataset_description=None,
                     dataset_version=None,
                     dataset_year=None,
                     dataset_contributer=None,
                     dataset_url=None,
                     include_consolidated=False):
            """
            Create an empty COCO dataset structure or copy and edit that copy.
            Handles missing directories gracefully.

            Args:
                input_dir: Directory containing input dataset to set default initial values
                output_dir: Directory to create the dataset in
                dataset_name: Name of the dataset
                dataset_description: Description of the dataset
                dataset_version: Version of the dataset
                dataset_year: Year of the dataset (int or None)
                dataset_contributer: Contributor of the dataset
                dataset_url: URL for the dataset
                include_consolidated: Whether to create a consolidated version (default: False)
            """
            # Validate required parameters
            if not output_dir:
                raise ValueError("Output directory must be specified")
            if not dataset_name:
                raise ValueError("Dataset name must be specified")

            # Create output directory structure
            output_dir = Path(output_dir) / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create directories for splits and consolidated (if needed)
            split_directories = ["train", "valid", "test"]
            for split in split_directories:
                (output_dir / split).mkdir(exist_ok=True)

            if include_consolidated:
                (output_dir / "consolidated").mkdir(exist_ok=True)

            # If no input directory provided, create empty dataset
            if not input_dir:
                # Get current year as integer
                current_year = date.today().year if dataset_year is None else dataset_year

                empty_coco = {
                    "info": {
                        "description": dataset_description or "Empty COCO dataset",
                        "version": str(dataset_version) or "0.0",
                        "year": current_year,  # Now an integer, not a date object
                        "contributor": dataset_contributer or "",
                        "url": dataset_url or ""
                    },
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": []
                }

                # Create empty splits
                for split in split_directories:
                    json_path = output_dir / split / "_annotations.coco.json"
                    with open(json_path, 'w') as f:
                        json.dump(empty_coco, f, indent=2)

                # Create consolidated version if requested
                if include_consolidated:
                    json_path = output_dir / "consolidated" / "_annotations.coco.json"
                    with open(json_path, 'w') as f:
                        json.dump(empty_coco, f, indent=2)

                print(f"Created empty COCO dataset at: {output_dir}")
                return

            # If input directory provided, copy and modify
            input_dir = Path(input_dir)

            # Initialize data structures with defaults
            data_structures = {
                "train": None,
                "valid": None,
                "test": None
            }

            # Try to load each split
            for split in split_directories:
                try:
                    json_path = input_dir / split / "_annotations.coco.json"
                    if json_path.exists():
                        with open(json_path, 'r') as f:
                            data_structures[split] = json.load(f)
                        print(f"Loaded {split} split from {json_path}")
                    else:
                        print(f"Warning: {json_path} not found. Creating empty {split} split.")
                        data_structures[split] = {
                            "info": {
                                "description": dataset_description or "Empty COCO dataset",
                                "version": str(dataset_version) or "0.0",
                                "year": date.today().year,  # Ensure year is an integer
                                "contributor": dataset_contributer or "",
                                "url": dataset_url or ""
                            },
                            "licenses": [],
                            "images": [],
                            "annotations": [],
                            "categories": []
                        }
                except Exception as e:
                    print(f"Error loading {split} split: {e}. Using empty structure.")
                    data_structures[split] = {
                        "info": {
                            "description": dataset_description or "Empty COCO dataset",
                            "version": str(dataset_version) or "0.0",
                            "year": date.today().year,  # Ensure year is an integer
                            "contributor": dataset_contributer or "",
                            "url": dataset_url or ""
                        },
                        "licenses": [],
                        "images": [],
                        "annotations": [],
                        "categories": []
                    }

            # Apply modifications to all splits
            for split, data in data_structures.items():
                if dataset_description is not None:
                    data['info']['description'] = dataset_description
                if dataset_version is not None:
                    data['info']['version'] = str(dataset_version)
                if dataset_year is not None:
                    data['info']['year'] = dataset_year  # Now using the provided year as integer
                if dataset_contributer is not None:
                    data['info']['contributor'] = dataset_contributer
                if dataset_url is not None:
                    data['info']['url'] = dataset_url

            # Copy image files if directories exist
            for split in split_directories:
                input_split_dir = input_dir / split
                output_split_dir = output_dir / split

                if input_split_dir.exists() and input_split_dir.is_dir():
                    print(f"Copying image files from {split} directory...")
                    # Find all image files in the split directory
                    image_files = list(input_split_dir.glob("*.jpg")) + \
                                list(input_split_dir.glob("*.png")) + \
                                list(input_split_dir.glob("*.jpeg"))

                    # Copy each image file
                    for img_file in image_files:
                        dest_img = output_split_dir / img_file.name
                        shutil.copy2(img_file, dest_img)
                        print(f"Copied: {img_file} -> {dest_img}")
                else:
                    print(f"Warning: {input_split_dir} not found. Skipping image copy.")

            # Save the modified datasets to the correct output directories
            for split in split_directories:
                json_path = output_dir / split / "_annotations.coco.json"
                with open(json_path, 'w') as f:
                    json.dump(data_structures[split], f, indent=2)
                print(f"Saved {split} split to {json_path}")

            # Create consolidated version if requested
            if include_consolidated:
                # Get current year as integer
                current_year = date.today().year if dataset_year is None else dataset_year

                consolidated_data = {
                    "info": {
                        "description": dataset_description or "Consolidated COCO dataset",
                        "version": str(dataset_version) or "0.0",
                        "year": current_year,  # Ensure year is an integer
                        "contributor": dataset_contributer or "",
                        "url": dataset_url or ""
                    },
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": []
                }

                # Merge all categories (avoid duplicates)
                seen_categories = set()
                for split in split_directories:
                    if data_structures[split] and "categories" in data_structures[split]:
                        for cat in data_structures[split]["categories"]:
                            cat_id = cat["id"]
                            if cat_id not in seen_categories:
                                seen_categories.add(cat_id)
                                consolidated_data["categories"].append(cat)

                # Merge all images and annotations
                image_id_offset = 0
                for split in split_directories:
                    if data_structures[split] and "images" in data_structures[split]:
                        for img in data_structures[split]["images"]:
                            # Create new image ID to avoid conflicts
                            new_img = img.copy()
                            new_img["id"] = img["id"] + image_id_offset
                            consolidated_data["images"].append(new_img)

                        # Merge annotations
                        for ann in data_structures[split]["annotations"]:
                            new_ann = ann.copy()
                            new_ann["id"] = ann["id"] + image_id_offset
                            new_ann["image_id"] = ann["image_id"] + image_id_offset
                            consolidated_data["annotations"].append(new_ann)

                        image_id_offset += len(data_structures[split]["images"])

                # Copy images to consolidated directory
                consolidated_images_dir = output_dir / "consolidated"
                for split in split_directories:
                    input_split_dir = input_dir / split
                    if input_split_dir.exists() and input_split_dir.is_dir():
                        for img_file in input_split_dir.glob("*.jpg"):
                            shutil.copy2(img_file, consolidated_images_dir / img_file.name)
                        for img_file in input_split_dir.glob("*.png"):
                            shutil.copy2(img_file, consolidated_images_dir / img_file.name)
                        for img_file in input_split_dir.glob("*.jpeg"):
                            shutil.copy2(img_file, consolidated_images_dir / img_file.name)

                # Save consolidated dataset
                json_path = output_dir / "consolidated" / "_annotations.coco.json"
                with open(json_path, 'w') as f:
                    json.dump(consolidated_data, f, indent=2)
                print(f"Saved consolidated dataset to {json_path}")

            print(f"COCO dataset successfully created at: {output_dir}")

        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #class READ: 
        #read which datasets within a directory have the same data as a reference dataset. to create symbolic links, or find redundancies.

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #class UPDATE:
        #update an existing dataset records

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #class DELETE:
        #delete dataset, and remove from catalog, processes, etc..

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    class MERGE: 
        #merge multiple datasets within a folder and create a new datasetx
        def json_COCO(input_dir="",
                                    output_dir="",
                                    dataset_name=None,
                                    dataset_description=None,
                                    dataset_version=None,
                                    dataset_year=None,
                                    dataset_contributer=None,
                                    dataset_url=None):
            """
            Merge multiple COCO datasets while preserving train/valid/test splits.

            Args:
                input_dir: Directory containing multiple dataset folders to merge
                output_dir: Directory to create the merged dataset in
                dataset_name: Name of the merged dataset
                dataset_description: Description for the merged dataset
                dataset_version: Version for the merged dataset
                dataset_year: Year for the merged dataset
                dataset_contributer: Contributor for the merged dataset
                dataset_url: URL for the merged dataset
            """
            # Validate required parameters
            if not output_dir:
                raise ValueError("Output directory must be specified")
            if not dataset_name:
                raise ValueError("Dataset name must be specified")

            input_dir = Path(input_dir)
            output_dir = Path(output_dir) / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create output split directories
            split_directories = ["train", "valid", "test"]
            for split in split_directories:
                (output_dir / split).mkdir(exist_ok=True)

            # Get all dataset folders in input directory
            dataset_folders = [d for d in input_dir.iterdir() if d.is_dir() and (d/"train"/"_annotations.coco.json").exists()]

            if not dataset_folders:
                raise ValueError(f"No valid COCO datasets found in {input_dir}")

            print(f"Found {len(dataset_folders)} datasets to merge: {[d.name for d in dataset_folders]}")

            # Track categories and ID offsets
            all_categories = []
            category_ids = set()
            split_offsets = {"train": {"images": 0, "annotations": 0},
                            "valid": {"images": 0, "annotations": 0},
                            "test": {"images": 0, "annotations": 0}}

            # First pass: collect all categories and create empty merged datasets
            for dataset_folder in dataset_folders:
                for split in split_directories:
                    json_path = dataset_folder / split / "_annotations.coco.json"
                    if not json_path.exists():
                        continue

                    with open(json_path, 'r') as f:
                        split_data = json.load(f)

                    # Collect categories
                    for cat in split_data.get("categories", []):
                        if cat["id"] not in category_ids:
                            all_categories.append(cat)
                            category_ids.add(cat["id"])

            # Initialize merged datasets with metadata
            merged_datasets = {}
            for split in split_directories:
                merged_datasets[split] = {
                    "info": {
                        "description": dataset_description or "Merged COCO dataset",
                        "version": str(dataset_version) or "1.0",
                        "year": dataset_year or date.today().year,
                        "contributor": dataset_contributer or "Merge script",
                        "url": dataset_url or ""
                    },
                    "licenses": [],
                    "images": [],
                    "annotations": [],
                    "categories": all_categories.copy()
                }

            # Second pass: merge datasets
            for dataset_folder in dataset_folders:
                dataset_name = dataset_folder.name

                for split in split_directories:
                    json_path = dataset_folder / split / "_annotations.coco.json"
                    if not json_path.exists():
                        continue

                    with open(json_path, 'r') as f:
                        split_data = json.load(f)

                    # Process images and annotations
                    for img in split_data.get("images", []):
                        # Create new image ID to avoid conflicts
                        new_img_id = img["id"] + split_offsets[split]["images"]
                        img["id"] = new_img_id
                        img["file_name"] = f"{dataset_name}_{split}_{img['file_name']}"  # Prefix with dataset and split names
                        merged_datasets[split]["images"].append(img)

                    # Process annotations
                    for ann in split_data.get("annotations", []):
                        # Create new annotation ID to avoid conflicts
                        new_ann_id = ann["id"] + split_offsets[split]["annotations"]
                        ann["id"] = new_ann_id
                        ann["image_id"] += split_offsets[split]["images"]  # Update image_id reference
                        merged_datasets[split]["annotations"].append(ann)

                    # Update offsets
                    split_offsets[split]["images"] += len(split_data.get("images", []))
                    split_offsets[split]["annotations"] += len(split_data.get("annotations", []))

                    # Copy images
                    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
                    for ext in image_extensions:
                        for img_file in (dataset_folder / split).glob(ext):
                            try:
                                new_filename = f"{dataset_name}_{split}_{img_file.name}"
                                dest_path = output_dir / split / new_filename
                                shutil.copy2(img_file, dest_path)
                            except Exception as e:
                                print(f"Error copying {img_file}: {e}")

                    print(f"Merged {split} split from {dataset_name}: {len(split_data.get('images', []))} images, {len(split_data.get('annotations', []))} annotations")

            # Save merged datasets
            for split in split_directories:
                if merged_datasets[split]["images"]:
                    json_path = output_dir / split / "_annotations.coco.json"
                    with open(json_path, 'w') as f:
                        json.dump(merged_datasets[split], f, indent=2)
                    print(f"Saved merged {split} split to {json_path}")
                else:
                    print(f"No data found for {split} split - skipping")

            return {
                "success": True,
                "output_dir": str(output_dir),
                "total_images": {split: len(ds["images"]) for split, ds in merged_datasets.items()},
                "total_annotations": {split: len(ds["annotations"]) for split, ds in merged_datasets.items()},
                "total_categories": len(all_categories),
                "datasets_merged": [d.name for d in dataset_folders]
            }
        """
        # Example usage
        json_COCO(
            input_dir="",
            output_dir="",
            dataset_name="",
            dataset_description="",
            dataset_version="",
            dataset_year=,
            dataset_contributer="",
            dataset_url=""
        )
        """
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():
    #class CONVERT:
        #convert one dataset format to another

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #class CLEAN:
        #removes nulls, performs validations, reports stats on datasets

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #class DOWNLOAD:
        #pull dataset from a source

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #class UPLOAD:
        #upload dataset to a source

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    class EXTRACT:
        #extract data from dataset archive files

        def unzip_file(zip_path, output_dir):
            """
            Unzips a file and places its contents in the specified directory.

            Args:
                zip_path (str): Path to the zip file to be unzipped
                output_dir (str): Directory where the contents should be extracted

            Returns:
                bool: True if successful, False if failed
            """
            try:
                # Convert paths to Path objects for better handling
                zip_path = Path(zip_path)
                output_dir = Path(output_dir)

                # Create output directory if it doesn't exist
                output_dir.mkdir(parents=True, exist_ok=True)

                # Check if the zip file exists
                if not zip_path.exists():
                    print(f"Error: Zip file {zip_path} does not exist")
                    return False

                # Open the zip file and extract its contents
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    print(f"Extracting contents of {zip_path} to {output_dir}...")
                    zip_ref.extractall(output_dir)

                print(f"Successfully extracted {zip_path} to {output_dir}")
                return True

            except zipfile.BadZipFile:
                print(f"Error: {zip_path} is not a valid zip file")
                return False
            except Exception as e:
                print(f"Error extracting {zip_path}: {str(e)}")
                return False

        def json_COCO(input_dir, output_dir, dataset_name="consolidated", dataset_description=None, dataset_version=None, dataset_year=None, dataset_contributer=None, dataset_url=None):
            """
            Consolidates a COCO dataset with any internal folder structure into a single dataset
            and places the output in a subfolder called "consolidated".

            Args:
                input_dir (str): Path to the directory containing the dataset with any folder structure
                output_dir (str): Path to the directory where the consolidated dataset should be saved
                dataset_name (str): Name for the consolidated dataset subfolder (default: "consolidated")

            Returns:
                dict: Information about the consolidation process
            """
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)

            # Create the consolidated subfolder
            consolidated_folder = output_dir / dataset_name
            consolidated_folder.mkdir(parents=True, exist_ok=True)

            # Track all data we'll collect
            consolidated_images = []
            consolidated_annotations = []
            consolidated_categories = []
            image_id_mapping = {}  # Original ID -> New ID
            current_id = 1  # Starting ID for consolidated dataset

            # Walk through the entire directory structure to find annotation files
            annotation_files = []
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file == "_annotations.coco.json":
                        annotation_files.append(Path(root) / file)

            if not annotation_files:
                raise FileNotFoundError(f"No COCO annotation files found in {input_dir}")

            print(f"Found {len(annotation_files)} COCO annotation files to consolidate")

            # Process each annotation file
            for annotation_file in annotation_files:
                print(f"Processing {annotation_file}")

                # Load the annotation file
                with open(annotation_file, 'r') as f:
                    data = json.load(f)

                # Process categories (avoid duplicates)
                for cat in data.get("categories", []):
                    if cat not in consolidated_categories:
                        consolidated_categories.append(cat)

                # Process images and annotations
                for img in data.get("images", []):
                    # Create new image ID
                    new_id = current_id
                    image_id_mapping[img["id"]] = new_id
                    current_id += 1

                    # Update image ID
                    img["id"] = new_id
                    consolidated_images.append(img)

                for ann in data.get("annotations", []):
                    # Update annotation IDs
                    ann["id"] = ann["id"] + image_id_mapping[ann["image_id"]] - ann["image_id"]
                    ann["image_id"] = image_id_mapping[ann["image_id"]]
                    consolidated_annotations.append(ann)

            # Create consolidated COCO structure
            consolidated_dataset = {
                "info": {
                    "description": dataset_description or "Empty COCO dataset",
                    "version": str(dataset_version) or "0.0",
                    "year": dataset_year or date.year,
                    "contributor": dataset_contributer or "",
                    "url": dataset_url or ""
                },
                "licenses": [],
                "images": consolidated_images,
                "annotations": consolidated_annotations,
                "categories": consolidated_categories
            }

            # Copy all images to the consolidated subfolder
            print("Copying images to consolidated directory...")
            image_count = 0
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_path = Path(root) / file
                        dest_path = consolidated_folder / file
                        # Handle duplicate filenames by adding a number suffix
                        if dest_path.exists():
                            base, ext = os.path.splitext(file)
                            counter = 1
                            while True:
                                new_name = f"{base}_{counter}{ext}"
                                dest_path = consolidated_folder / new_name
                                if not dest_path.exists():
                                    break
                                counter += 1
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_path, dest_path)
                        image_count += 1

            # Save the consolidated annotation file
            output_annotation_path = consolidated_folder / "_annotations.coco.json"
            with open(output_annotation_path, 'w') as f:
                json.dump(consolidated_dataset, f, indent=2)

            return {
                "input_directory": str(input_dir),
                "output_directory": str(consolidated_folder),
                "num_images": len(consolidated_images),
                "num_annotations": len(consolidated_annotations),
                "num_categories": len(consolidated_categories),
                "num_files_processed": len(annotation_files),
                "images_copied": image_count
            }
        
        def json_COCO_V0(input_dir, output_dir, dataset_name="consolidated", dataset_description=None, dataset_version=None, dataset_year=None, dataset_contributer=None, dataset_url=None):
            """
            Consolidates a COCO dataset with any internal folder structure into a single dataset
            and places the output in a new folder with the specified name containing a "consolidated" subfolder.

            Args:
                input_dir (str): Path to the directory containing the dataset with any folder structure
                output_dir (str): Path to the directory where the new dataset folder should be created
                dataset_name (str): Name for the new dataset folder (default: "consolidated_dataset")

            Returns:
                dict: Information about the consolidation process
            """
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)

            # Create the new dataset folder
            dataset_folder = output_dir / dataset_name
            dataset_folder.mkdir(parents=True, exist_ok=True)

            # Create the consolidated subfolder
            consolidated_folder = dataset_folder / "consolidated"
            consolidated_folder.mkdir(parents=True, exist_ok=True)

            # Track all data we'll collect
            consolidated_images = []
            consolidated_annotations = []
            consolidated_categories = []
            image_id_mapping = {}  # Original ID -> New ID
            current_id = 1  # Starting ID for consolidated dataset

            # Walk through the entire directory structure to find annotation files
            annotation_files = []
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file == "_annotations.coco.json":
                        annotation_files.append(Path(root) / file)

            if not annotation_files:
                raise FileNotFoundError(f"No COCO annotation files found in {input_dir}")

            print(f"Found {len(annotation_files)} COCO annotation files to consolidate")

            # Process each annotation file
            for annotation_file in annotation_files:
                print(f"Processing {annotation_file}")

                # Load the annotation file
                with open(annotation_file, 'r') as f:
                    data = json.load(f)

                # Process categories (avoid duplicates)
                for cat in data.get("categories", []):
                    if cat not in consolidated_categories:
                        consolidated_categories.append(cat)

                # Process images and annotations
                for img in data.get("images", []):
                    # Create new image ID
                    new_id = current_id
                    image_id_mapping[img["id"]] = new_id
                    current_id += 1

                    # Update image ID
                    img["id"] = new_id
                    consolidated_images.append(img)

                for ann in data.get("annotations", []):
                    # Update annotation IDs
                    ann["id"] = ann["id"] + image_id_mapping[ann["image_id"]] - ann["image_id"]
                    ann["image_id"] = image_id_mapping[ann["image_id"]]
                    consolidated_annotations.append(ann)

            # Create consolidated COCO structure
            consolidated_dataset = {
                "info": {
                    "description": dataset_description or "Empty COCO dataset",
                    "version": str(dataset_version) or "0.0",
                    "year": dataset_year or date.year,
                    "contributor": dataset_contributer or "",
                    "url": dataset_url or ""
                },
                "licenses": [],
                "images": consolidated_images,
                "annotations": consolidated_annotations,
                "categories": consolidated_categories
            }

            # Copy all images to the consolidated subfolder
            print("Copying images to consolidated directory...")
            image_count = 0
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_path = Path(root) / file
                        dest_path = consolidated_folder / file
                        # Handle duplicate filenames by adding a number suffix
                        if dest_path.exists():
                            base, ext = os.path.splitext(file)
                            counter = 1
                            while True:
                                new_name = f"{base}_{counter}{ext}"
                                dest_path = consolidated_folder / new_name
                                if not dest_path.exists():
                                    break
                                counter += 1
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_path, dest_path)
                        image_count += 1

            # Save the consolidated annotation file
            output_annotation_path = consolidated_folder / "_annotations.coco.json"
            with open(output_annotation_path, 'w') as f:
                json.dump(consolidated_dataset, f, indent=2)

            return {
                "input_directory": str(input_dir),
                "output_directory": str(dataset_folder),
                "consolidated_directory": str(consolidated_folder),
                "num_images": len(consolidated_images),
                "num_annotations": len(consolidated_annotations),
                "num_categories": len(consolidated_categories),
                "num_files_processed": len(annotation_files),
                "images_copied": image_count
            }

        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #class INGEST:
        #ingest datasets into database with symbolic links to files

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #class DEDUP:
        #deduplicate records in datasets

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #class REMOVE:
        #remove records from a dataset that is common in another dataset

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    class SPLIT:
    #DATASETS-SPLIT : take a raw or consolidated dataset and turn into train, valid, test dataset splits

        def json_COCO(input_dir="",
                    output_dir="",
                    dataset_name=None,
                    dataset_description=None,
                    dataset_version=None,
                    dataset_year=None,
                    dataset_contributer=None,
                    dataset_url=None,
                    train_ratio=0.7,
                    valid_ratio=0.15,
                    test_ratio=0.15,
                    random_seed=42):
            """
            Split a COCO dataset into train/valid/test folders while preserving the data structure.

            Args:
                input_dir: Directory containing COCO dataset (either consolidated or single dataset)
                output_dir: Directory to create the split dataset in
                dataset_name: Name of the split dataset
                dataset_description: Description for the split dataset
                dataset_version: Version for the split dataset
                dataset_year: Year for the split dataset
                dataset_contributer: Contributor for the split dataset
                dataset_url: URL for the split dataset
                train_ratio: Ratio of data for training (default: 0.7)
                valid_ratio: Ratio of data for validation (default: 0.15)
                test_ratio: Ratio of data for testing (default: 0.15)
                random_seed: Random seed for reproducibility (default: 42)
            """
            # Validate required parameters
            if not output_dir:
                raise ValueError("Output directory must be specified")
            if not dataset_name:
                raise ValueError("Dataset name must be specified")

            input_dir = Path(input_dir)
            output_dir = Path(output_dir) / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create output split directories
            split_directories = ["train", "valid", "test"]
            for split in split_directories:
                (output_dir / split).mkdir(exist_ok=True)

            # Find COCO JSON file (either in consolidated folder or root)
            coco_json_path = None
            if (input_dir / "consolidated" / "_annotations.coco.json").exists():
                coco_json_path = input_dir / "consolidated" / "_annotations.coco.json"
            elif (input_dir / "_annotations.coco.json").exists():
                coco_json_path = input_dir / "_annotations.coco.json"

            if not coco_json_path:
                raise FileNotFoundError(f"No COCO JSON file found in {input_dir}")

            # Load COCO dataset
            try:
                with open(coco_json_path, 'r') as f:
                    coco_data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {coco_json_path}: {e}")
            except Exception as e:
                raise ValueError(f"Error reading {coco_json_path}: {e}")

            # Check if dataset has images
            if not coco_data.get("images"):
                raise ValueError(f"No images found in {coco_json_path}")

            # Set random seed for reproducibility
            random.seed(random_seed)

            # Get all image IDs and shuffle them
            image_ids = [img["id"] for img in coco_data["images"]]
            random.shuffle(image_ids)

            # Calculate split points
            train_end = int(len(image_ids) * train_ratio)
            valid_end = train_end + int(len(image_ids) * valid_ratio)

            # Split image IDs
            train_ids = set(image_ids[:train_end])
            valid_ids = set(image_ids[train_end:valid_end])
            test_ids = set(image_ids[valid_end:])

            # Create split datasets
            split_datasets = {
                "train": {
                    "info": coco_data["info"].copy(),
                    "licenses": coco_data["licenses"],
                    "images": [],
                    "annotations": [],
                    "categories": coco_data["categories"]
                },
                "valid": {
                    "info": coco_data["info"].copy(),
                    "licenses": coco_data["licenses"],
                    "images": [],
                    "annotations": [],
                    "categories": coco_data["categories"]
                },
                "test": {
                    "info": coco_data["info"].copy(),
                    "licenses": coco_data["licenses"],
                    "images": [],
                    "annotations": [],
                    "categories": coco_data["categories"]
                }
            }

            # Update metadata
            for split in split_datasets:
                split_datasets[split]["info"]["description"] = (
                    dataset_description or
                    f"{split.capitalize()} split of {coco_data['info']['description']}"
                )
                split_datasets[split]["info"]["version"] = str(dataset_version or coco_data["info"]["version"])
                split_datasets[split]["info"]["year"] = dataset_year or coco_data["info"]["year"]
                split_datasets[split]["info"]["contributor"] = dataset_contributer or coco_data["info"]["contributor"]
                split_datasets[split]["info"]["url"] = dataset_url or coco_data["info"]["url"]

            # Split images and annotations
            image_id_map = {}  # Original ID -> split name
            for img in coco_data["images"]:
                if img["id"] in train_ids:
                    split_datasets["train"]["images"].append(img)
                    image_id_map[img["id"]] = "train"
                elif img["id"] in valid_ids:
                    split_datasets["valid"]["images"].append(img)
                    image_id_map[img["id"]] = "valid"
                elif img["id"] in test_ids:
                    split_datasets["test"]["images"].append(img)
                    image_id_map[img["id"]] = "test"

            # Split annotations
            for ann in coco_data["annotations"]:
                split_name = image_id_map.get(ann["image_id"])
                if split_name:
                    split_datasets[split_name]["annotations"].append(ann)

            # Copy images to appropriate split directories
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']

            # First try consolidated folder, then root directory
            image_dir = input_dir / "consolidated" if (input_dir / "consolidated").exists() else input_dir

            for ext in image_extensions:
                for img_file in image_dir.glob(ext):
                    # Find corresponding image in dataset
                    img_name = img_file.name
                    found = False

                    for img in coco_data["images"]:
                        if img["file_name"] == img_name:
                            split_name = image_id_map.get(img["id"])
                            if split_name:
                                try:
                                    dest_path = output_dir / split_name / img_name
                                    shutil.copy2(img_file, dest_path)
                                    found = True
                                except Exception as e:
                                    print(f"Error copying {img_file}: {e}")
                            break

                    if not found and img_name.startswith(("train_", "valid_", "test_")):
                        # Handle case where images were already split
                        split_name = img_name.split('_')[0]
                        if split_name in split_directories:
                            try:
                                dest_path = output_dir / split_name / img_name
                                shutil.copy2(img_file, dest_path)
                                found = True
                            except Exception as e:
                                print(f"Error copying {img_file}: {e}")

            # Save split datasets
            for split in split_directories:
                if split_datasets[split]["images"]:
                    json_path = output_dir / split / "_annotations.coco.json"
                    with open(json_path, 'w') as f:
                        json.dump(split_datasets[split], f, indent=2)
                    print(f"Saved {split} split: {len(split_datasets[split]['images'])} images, {len(split_datasets[split]['annotations'])} annotations")
                else:
                    print(f"No data for {split} split - creating empty dataset")
                    empty_dataset = {
                        "info": split_datasets[split]["info"],
                        "licenses": [],
                        "images": [],
                        "annotations": [],
                        "categories": []
                    }
                    with open(output_dir / split / "_annotations.coco.json", 'w') as f:
                        json.dump(empty_dataset, f, indent=2)

            return {
                "success": True,
                "output_dir": str(output_dir),
                "split_counts": {
                    split: {
                        "images": len(ds["images"]),
                        "annotations": len(ds["annotations"])
                    }
                    for split, ds in split_datasets.items()
                },
                "ratios": {
                    "train": train_ratio,
                    "valid": valid_ratio,
                    "test": test_ratio
                }
            }
        """
        # Example usage
        json_COCO(
            input_dir="",
            output_dir="",
            dataset_name="",
            dataset_description="",
            dataset_version="",
            dataset_year=,
            dataset_contributer="",
            dataset_url="",
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        """
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    class CONSOLIDATE:
        #turn raw or split data into a consolidated folder with all annotations and files

        def json_COCO(input_dir="", output_dir="", dataset_name=None, dataset_description=None, dataset_version=None, dataset_year=None, dataset_contributer=None, dataset_url=None):
            """
            Consolidate COCO datasets from train/valid/test splits into a single consolidated folder.

            Args:
                input_dir: Directory containing train/valid/test splits with COCO JSON files
                output_dir: Directory to create the consolidated dataset in
                dataset_name: Name of the consolidated dataset
                dataset_description: Description for the consolidated dataset
                dataset_version: Version for the consolidated dataset
                dataset_year: Year for the consolidated dataset
                dataset_contributer: Contributor for the consolidated dataset
                dataset_url: URL for the consolidated dataset
            """
            # Validate required parameters
            if not output_dir:
                raise ValueError("Output directory must be specified")
            if not dataset_name:
                raise ValueError("Dataset name must be specified")

            input_dir = Path(input_dir)
            output_dir = Path(output_dir) / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Create consolidated directory
            consolidated_dir = output_dir / "consolidated"
            consolidated_dir.mkdir(parents=True, exist_ok=True)

            # Initialize combined dataset structure with provided metadata
            combined_dataset = {
                "info": {
                    "description": dataset_description or "Consolidated COCO dataset",
                    "version": str(dataset_version) or "1.0",
                    "year": dataset_year or date.year,
                    "contributor": dataset_contributer or "",
                    "url": dataset_url or ""
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": []
            }

            # Process each split
            split_directories = ["train", "valid", "test"]
            category_ids = set()
            image_id_offset = 0
            annotation_id_offset = 0

            for split in split_directories:
                split_dir = input_dir / split

                # Check if split directory exists
                if not split_dir.exists():
                    print(f"Warning: {split} directory not found. Skipping.")
                    continue

                # Load COCO JSON file
                json_path = split_dir / "_annotations.coco.json"
                if not json_path.exists():
                    print(f"Warning: No COCO JSON file found in {split} directory. Skipping.")
                    continue

                try:
                    with open(json_path, 'r') as f:
                        split_data = json.load(f)

                    # Combine categories (avoid duplicates)
                    for cat in split_data.get("categories", []):
                        if cat["id"] not in category_ids:
                            combined_dataset["categories"].append(cat)
                            category_ids.add(cat["id"])

                    # Process images and annotations
                    for img in split_data.get("images", []):
                        # Create new image ID to avoid conflicts
                        new_img_id = img["id"] + image_id_offset
                        img["id"] = new_img_id
                        img["file_name"] = f"{split}_{img['file_name']}"  # Prefix with split name
                        combined_dataset["images"].append(img)

                    # Process annotations
                    for ann in split_data.get("annotations", []):
                        # Create new annotation ID to avoid conflicts
                        new_ann_id = ann["id"] + annotation_id_offset
                        ann["id"] = new_ann_id
                        ann["image_id"] += image_id_offset  # Update image_id reference
                        combined_dataset["annotations"].append(ann)

                    # Update offsets
                    image_id_offset += len(split_data.get("images", []))
                    annotation_id_offset += len(split_data.get("annotations", []))
                    print(f"Processed {split} split: {len(split_data.get('images', []))} images, {len(split_data.get('annotations', []))} annotations")

                except json.JSONDecodeError as e:
                    print(f"Error parsing {json_path}: {e}. Skipping.")
                    continue

            # Save consolidated JSON file
            consolidated_json_path = consolidated_dir / "_annotations.coco.json"
            with open(consolidated_json_path, 'w') as f:
                json.dump(combined_dataset, f, indent=2)

            # Copy all images to consolidated folder
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff']
            image_count = 0

            for split in split_directories:
                split_dir = input_dir / split
                if not split_dir.exists():
                    continue

                for ext in image_extensions:
                    for img_file in split_dir.glob(ext):
                        try:
                            # Create new filename with split prefix to avoid conflicts
                            new_filename = f"{split}_{img_file.name}"
                            dest_path = consolidated_dir / new_filename
                            shutil.copy2(img_file, dest_path)
                            image_count += 1
                        except Exception as e:
                            print(f"Error copying {img_file}: {e}")

            # Verify we have the right number of images
            if len(combined_dataset["images"]) != image_count:
                print(f"Warning: Image count mismatch - {len(combined_dataset['images'])} in JSON but {image_count} files copied")

            print(f"Successfully consolidated datasets to: {consolidated_dir}")
            return {
                "success": True,
                "consolidated_dir": str(consolidated_dir),
                "total_images": len(combined_dataset["images"]),
                "total_annotations": len(combined_dataset["annotations"]),
                "total_categories": len(combined_dataset["categories"]),
                "image_files_copied": image_count
            }
        
        def json_COCO_V0(input_dir="", output_dir="", dataset_name=None, dataset_description=None, dataset_version=None, dataset_year=None, dataset_contributer=None, dataset_url=None):
            """
            Consolidates a COCO dataset with any internal folder structure into a single dataset
            and places the output in a new folder with the specified name containing a "consolidated" subfolder.
            Preserves original image filenames without adding suffixes.

            Args:
                input_dir (str): Path to the directory containing the dataset with any folder structure
                output_dir (str): Path to the directory where the new dataset folder should be created
                dataset_name (str): Name for the new dataset folder (default: "consolidated_dataset")

            Returns:
                dict: Information about the consolidation process
            """
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)

            # Create the new dataset folder
            dataset_folder = output_dir / dataset_name
            dataset_folder.mkdir(parents=True, exist_ok=True)

            # Create the consolidated subfolder
            consolidated_folder = dataset_folder / "consolidated"
            consolidated_folder.mkdir(parents=True, exist_ok=True)

            # Track all data we'll collect
            consolidated_images = []
            consolidated_annotations = []
            consolidated_categories = []
            image_id_mapping = {}  # Original ID -> New ID
            current_id = 1  # Starting ID for consolidated dataset

            # Walk through the entire directory structure to find annotation files
            annotation_files = []
            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file == "_annotations.coco.json":
                        annotation_files.append(Path(root) / file)

            if not annotation_files:
                raise FileNotFoundError(f"No COCO annotation files found in {input_dir}")

            print(f"Found {len(annotation_files)} COCO annotation files to consolidate")

            # Process each annotation file
            for annotation_file in annotation_files:
                print(f"Processing {annotation_file}")

                # Load the annotation file
                with open(annotation_file, 'r') as f:
                    data = json.load(f)

                # Process categories (avoid duplicates)
                for cat in data.get("categories", []):
                    if cat not in consolidated_categories:
                        consolidated_categories.append(cat)

                # Process images and annotations
                for img in data.get("images", []):
                    # Create new image ID
                    new_id = current_id
                    image_id_mapping[img["id"]] = new_id
                    current_id += 1

                    # Update image ID
                    img["id"] = new_id
                    consolidated_images.append(img)

                for ann in data.get("annotations", []):
                    # Update annotation IDs
                    ann["id"] = ann["id"] + image_id_mapping[ann["image_id"]] - ann["image_id"]
                    ann["image_id"] = image_id_mapping[ann["image_id"]]
                    consolidated_annotations.append(ann)

            # Create consolidated COCO structure
            consolidated_dataset = {
                "info": {
                    "description": dataset_description or "Consolidated COCO dataset",
                    "version": str(dataset_version) or "1.0",
                    "year": dataset_year or date.year,
                    "contributor": dataset_contributer or "",
                    "url": dataset_url or ""
                },
                "licenses": [],
                "images": consolidated_images,
                "annotations": consolidated_annotations,
                "categories": consolidated_categories
            }

            # Copy all images to the consolidated subfolder while preserving filenames
            print("Copying images to consolidated directory...")
            image_count = 0
            image_filename_map = {}  # Track original filenames and their new paths

            for root, dirs, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src_path = Path(root) / file
                        dest_path = consolidated_folder / file

                        # Check for filename conflicts
                        if dest_path.exists():
                            # If there's a conflict, we'll need to handle it differently
                            # For now, we'll overwrite (you may want to change this behavior)
                            print(f"Warning: Overwriting existing file {dest_path}")
                        else:
                            image_count += 1

                        # Record the mapping
                        image_filename_map[file] = dest_path
                        shutil.copy2(src_path, dest_path)

            # Save the consolidated annotation file
            output_annotation_path = consolidated_folder / "_annotations.coco.json"
            with open(output_annotation_path, 'w') as f:
                json.dump(consolidated_dataset, f, indent=2)

            return {
                "input_directory": str(input_dir),
                "output_directory": str(dataset_folder),
                "consolidated_directory": str(consolidated_folder),
                "num_images": len(consolidated_images),
                "num_annotations": len(consolidated_annotations),
                "num_categories": len(consolidated_categories),
                "num_files_processed": len(annotation_files),
                "images_copied": image_count,
                "image_filename_map": image_filename_map  # Added to show filename mappings
            }

        """
        # Example usage
        json_COCO(
            input_dir="",
            output_dir="",
            dataset_name="",
            dataset_description="",
            dataset_version="",
            dataset_year=,
            dataset_contributer="",
            dataset_url=""
        )
        """
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #class REFACTOR_CATEGORY:
        #crosswalk categorical differences between datasets

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #class COPY_SINGLE
        #make an exact copy

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    class COPY:
    #make an exact copy

        def json_COCO(input_dir,
                                output_dir,
                                validate_copy=True):
            """
            Copy entire dataset folders (with all subdirectories) and validate the copy.
            Handles complex directory structures with various combinations of train/valid/test/consolidated folders.

            Args:
                input_dir: Directory containing dataset folders to copy
                output_dir: Directory to copy the datasets to
                validate_copy: Whether to validate the copy (default: True)

            Returns:
                dict: Information about the copy operation including validation results for each dataset
            """
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            results = {
                "success": False,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "datasets_copied": 0,
                "validation_results": {}
            }

            # Find all dataset folders (directories containing COCO JSON files)
            dataset_folders = []
            for item in input_dir.iterdir():
                if item.is_dir():
                    # Check if this directory contains any COCO JSON files
                    coco_files = list(item.rglob("_annotations.coco.json"))
                    if coco_files:
                        dataset_folders.append(item)

            if not dataset_folders:
                raise ValueError(f"No COCO dataset folders found in {input_dir}")

            print(f"Found {len(dataset_folders)} dataset folders to copy")

            for dataset_folder in dataset_folders:
                dataset_name = dataset_folder.name
                print(f"\nProcessing dataset: {dataset_name}")

                try:
                    # Calculate output path for this dataset
                    output_dataset_dir = output_dir / dataset_name

                    # Copy the entire directory tree
                    shutil.copytree(dataset_folder, output_dataset_dir, dirs_exist_ok=True)
                    print(f"Copied dataset {dataset_name} to {output_dataset_dir}")

                    # Validate the copy if requested
                    if validate_copy:
                        validation = DATASETS.COPY.VALIDATE_json_COCO(
                            original_dir=dataset_folder,
                            copied_dir=output_dataset_dir
                        )
                        results["validation_results"][dataset_name] = validation

                        if validation["success"]:
                            print(f"Validation successful for {dataset_name}")
                        else:
                            print(f"Validation failed for {dataset_name} with {len(validation['errors'])} error(s)")

                    results["datasets_copied"] += 1

                except Exception as e:
                    print(f"Error copying dataset {dataset_name}: {e}")
                    results["validation_results"][dataset_name] = {
                        "success": False,
                        "errors": [str(e)]
                    }

            results["success"] = results["datasets_copied"] > 0
            return results
        def VALIDATE_json_COCO(original_dir, copied_dir):
            """
            Validate that a dataset copy preserves all files and structure.

            Args:
                original_dir: Original dataset directory
                copied_dir: Copied dataset directory

            Returns:
                dict: Validation results
            """
            results = {
                "success": False,
                "errors": [],
                "file_counts": {},
                "structure_valid": False
            }

            try:
                # Walk through both directories and compare
                original_files = {}
                copied_files = {}

                for root, dirs, files in os.walk(original_dir):
                    rel_path = os.path.relpath(root, original_dir)
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_file_path = os.path.join(rel_path, file)
                        original_files[rel_file_path] = full_path

                for root, dirs, files in os.walk(copied_dir):
                    rel_path = os.path.relpath(root, copied_dir)
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_file_path = os.path.join(rel_path, file)
                        copied_files[rel_file_path] = full_path

                # Check for missing files
                missing_in_copied = set(original_files.keys()) - set(copied_files.keys())
                missing_in_original = set(copied_files.keys()) - set(original_files.keys())

                if missing_in_copied:
                    results["errors"].append(f"Files missing in copied directory: {list(missing_in_copied)}")
                if missing_in_original:
                    results["errors"].append(f"Files missing in original directory: {list(missing_in_original)}")

                # Check file contents
                if not missing_in_copied and not missing_in_original:
                    different_files = []
                    for file_path in original_files:
                        try:
                            with open(original_files[file_path], 'rb') as f1, open(copied_files[file_path], 'rb') as f2:
                                if hashlib.sha256(f1.read()).hexdigest() != hashlib.sha256(f2.read()).hexdigest():
                                    different_files.append(file_path)
                        except Exception as e:
                            results["errors"].append(f"Error comparing {file_path}: {e}")

                    if different_files:
                        results["errors"].append(f"Different files: {different_files}")

                # Check directory structure
                original_dirs = set()
                copied_dirs = set()

                for root, dirs, files in os.walk(original_dir):
                    rel_path = os.path.relpath(root, original_dir)
                    if rel_path != '.':
                        original_dirs.add(rel_path)

                for root, dirs, files in os.walk(copied_dir):
                    rel_path = os.path.relpath(root, copied_dir)
                    if rel_path != '.':
                        copied_dirs.add(rel_path)

                missing_dirs_in_copied = original_dirs - copied_dirs
                missing_dirs_in_original = copied_dirs - original_dirs

                if missing_dirs_in_copied:
                    results["errors"].append(f"Directories missing in copied directory: {list(missing_dirs_in_copied)}")
                if missing_dirs_in_original:
                    results["errors"].append(f"Directories missing in original directory: {list(missing_dirs_in_original)}")

                # Count files by type
                file_counts = {}
                for file_path in original_files:
                    ext = os.path.splitext(file_path)[1].lower()
                    file_counts[ext] = file_counts.get(ext, 0) + 1
                results["file_counts"] = file_counts

                # Final validation
                if not results["errors"]:
                    results["success"] = True
                    results["structure_valid"] = True

            except Exception as e:
                results["errors"].append(f"Validation error: {e}")

            return results
        """
        # Example usage
        json_COCO(
            input_dir="",
            output_dir="",
            validate_copy=True
        )
        """

        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    class MOVE:
        #move from one directory to another

        def json_COCO(input_dir,
                                   output_dir,
                                   validate_copy=True):
            """
            Move entire dataset folders (with all subdirectories) to a new location and validate the move.
            Handles complex directory structures with various combinations of train/valid/test/consolidated folders.

            Args:
                input_dir: Directory containing dataset folders to move
                output_dir: Directory to move the datasets to
                validate_copy: Whether to validate the copy before deleting originals (default: True)

            Returns:
                dict: Information about the move operation including validation results for each dataset
            """
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            results = {
                "success": False,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "datasets_moved": 0,
                "validation_results": {},
                "originals_deleted": False
            }

            # Find all dataset folders (directories containing COCO JSON files)
            dataset_folders = []
            for item in input_dir.iterdir():
                if item.is_dir():
                    # Check if this directory contains any COCO JSON files
                    coco_files = list(item.rglob("_annotations.coco.json"))
                    if coco_files:
                        dataset_folders.append(item)

            if not dataset_folders:
                raise ValueError(f"No COCO dataset folders found in {input_dir}")

            print(f"Found {len(dataset_folders)} dataset folders to move")

            for dataset_folder in dataset_folders:
                dataset_name = dataset_folder.name
                print(f"\nProcessing dataset: {dataset_name}")

                try:
                    # Calculate output path for this dataset
                    output_dataset_dir = output_dir / dataset_name

                    # First copy the entire directory tree
                    shutil.copytree(dataset_folder, output_dataset_dir, dirs_exist_ok=True)
                    print(f"Copied dataset {dataset_name} to {output_dataset_dir}")

                    # Validate the copy if requested
                    if validate_copy:
                        validation = DATASETS.MOVE.VALIDATE_json_COCO(
                            original_dir=dataset_folder,
                            copied_dir=output_dataset_dir
                        )
                        results["validation_results"][dataset_name] = validation

                        if not validation["success"]:
                            print(f"Validation failed for {dataset_name} - ABORTING MOVE")
                            continue

                    # If validation passed (or was skipped), delete the original
                    try:
                        shutil.rmtree(dataset_folder)
                        print(f"Deleted original dataset {dataset_name}")
                        results["datasets_moved"] += 1
                    except Exception as e:
                        print(f"Error deleting original dataset {dataset_name}: {e}")
                        results["validation_results"][dataset_name]["errors"].append(
                            f"Failed to delete original: {str(e)}"
                        )

                except Exception as e:
                    print(f"Error moving dataset {dataset_name}: {e}")
                    results["validation_results"][dataset_name] = {
                        "success": False,
                        "errors": [str(e)]
                    }

            # Mark that originals were deleted if any datasets were successfully moved
            results["originals_deleted"] = results["datasets_moved"] > 0
            results["success"] = results["datasets_moved"] > 0
            return results
        def VALIDATE_json_COCO(original_dir, copied_dir):
            """
            Validate that a dataset copy preserves all files and structure.

            Args:
                original_dir: Original dataset directory
                copied_dir: Copied dataset directory

            Returns:
                dict: Validation results
            """
            results = {
                "success": False,
                "errors": [],
                "file_counts": {},
                "structure_valid": False
            }

            try:
                # Walk through both directories and compare
                original_files = {}
                copied_files = {}

                for root, dirs, files in os.walk(original_dir):
                    rel_path = os.path.relpath(root, original_dir)
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_file_path = os.path.join(rel_path, file)
                        original_files[rel_file_path] = full_path

                for root, dirs, files in os.walk(copied_dir):
                    rel_path = os.path.relpath(root, copied_dir)
                    for file in files:
                        full_path = os.path.join(root, file)
                        rel_file_path = os.path.join(rel_path, file)
                        copied_files[rel_file_path] = full_path

                # Check for missing files
                missing_in_copied = set(original_files.keys()) - set(copied_files.keys())
                missing_in_original = set(copied_files.keys()) - set(original_files.keys())

                if missing_in_copied:
                    results["errors"].append(f"Files missing in copied directory: {list(missing_in_copied)}")
                if missing_in_original:
                    results["errors"].append(f"Files missing in original directory: {list(missing_in_original)}")

                # Check file contents
                if not missing_in_copied and not missing_in_original:
                    different_files = []
                    for file_path in original_files:
                        try:
                            with open(original_files[file_path], 'rb') as f1, open(copied_files[file_path], 'rb') as f2:
                                if hashlib.sha256(f1.read()).hexdigest() != hashlib.sha256(f2.read()).hexdigest():
                                    different_files.append(file_path)
                        except Exception as e:
                            results["errors"].append(f"Error comparing {file_path}: {e}")

                    if different_files:
                        results["errors"].append(f"Different files: {different_files}")

                # Check directory structure
                original_dirs = set()
                copied_dirs = set()

                for root, dirs, files in os.walk(original_dir):
                    rel_path = os.path.relpath(root, original_dir)
                    if rel_path != '.':
                        original_dirs.add(rel_path)

                for root, dirs, files in os.walk(copied_dir):
                    rel_path = os.path.relpath(root, copied_dir)
                    if rel_path != '.':
                        copied_dirs.add(rel_path)

                missing_dirs_in_copied = original_dirs - copied_dirs
                missing_dirs_in_original = copied_dirs - original_dirs

                if missing_dirs_in_copied:
                    results["errors"].append(f"Directories missing in copied directory: {list(missing_dirs_in_copied)}")
                if missing_dirs_in_original:
                    results["errors"].append(f"Directories missing in original directory: {list(missing_dirs_in_original)}")

                # Count files by type
                file_counts = {}
                for file_path in original_files:
                    ext = os.path.splitext(file_path)[1].lower()
                    file_counts[ext] = file_counts.get(ext, 0) + 1
                results["file_counts"] = file_counts

                # Final validation
                if not results["errors"]:
                    results["success"] = True
                    results["structure_valid"] = True

            except Exception as e:
                results["errors"].append(f"Validation error: {e}")

            return results
        """
        # Example usage
        move_result = json_COCO(
            input_dir="",
            output_dir="",
            validate_copy=True
        )
        """
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #DATASETS-CHUNK : create new datasets which are smaller chunks of an original dataset

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #DATASETS-ARCHIVE : zip and compress an aggragated dataset with its annotations and files

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #DATASETS-RESTORE : decompress an aggragated dataset into its original set

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #DATASETS-VERSION : set versioning for a dataset

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #DATASETS-DISAGGRAGATE : turn all the annotation references links to a separate file store location and move files to that location

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

    #DATASETS-AGGRAGATE : make all annotation reference links local directory and copy files to that directory

        #def json_COCO(output_dir, dataset_name):
        #def json_COCO_MMDetection():
        #def json_CreateML():
        #def json_PaliGemma():
        #def json_Florence2_Object_Detection():
        #def json_OpenAI():
        #def xml_PascalVOC():
        #def txt_YOLO_Darknet():
        #def txt_YOLO_v3_Keras():
        #def txt_YOLO_v4_PyTorch():
        #def txt_Scaled_YOLOv4():
        #def txt_YOLOv5_Oriented_Bounding_Boxes():
        #def txt_meituan_YOLOv6():
        #def txt_YOLOv5_PyTorch():
        #def txt_YOLOv7_PyTorch():
        #def txt_YOLOv8():
        #def txt_YOLOv8_Oriented_Bounding_Boxes():
        #def txt_YOLOv9():
        #def txt_YOLOv11():
        #def txt_YOLOv12():
        #def csv_Tensorflow_Object_Detection():
        #def csv_RetinaNet_Keras():
        #def csv_Multi_Label_Classification():
        #def other_OpenAI_CLIP_Classification():
        #def other_Tensorflow_TFRecord():

#DATABASE : 

#DATABASE-CATALOG-DATASET
#DATABASE-ORM
#DATABASE-SEARCH/RETRIEVAL

