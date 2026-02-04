# How to Get DETR Dataset Images

**Date:** December 26, 2025  
**Purpose:** Complete guide to downloading DETR utility detection dataset images  
**Status:** Images NOT currently available locally - only annotations exist

---

## 🎯 Quick Summary

The DETR repository contains **only annotation JSON files** (metadata for 923 images).  
The actual image files were excluded from git due to size (~200-500 MB total).

**You need to download 3 datasets from Roboflow:**
1. Insulators (599 images)
2. Crossarm (207 images)  
3. Utility-pole (218 images)

---

## 📥 Method 1: Manual Download (RECOMMENDED - No Account Needed)

### Step 1: Download Each Dataset

Visit each URL below and download in **COCO format**:

#### 1️⃣ Insulators Dataset (599 images)
- **URL:** https://universe.roboflow.com/sofia-valdivieso-von-teuber/insulators-wo6lb/dataset/3
- **Steps:**
  1. Click "Download Dataset" button
  2. Select format: **COCO**
  3. Click "Continue" → "Download ZIP"
- **Expected file:** `Insulators.v5i.coco.zip`

#### 2️⃣ Crossarm Dataset (207 images)
- **URL:** https://universe.roboflow.com/project-91iyv/song-crossarm-zqkmo
- **Steps:**
  1. Click "Download Dataset" button
  2. Select format: **COCO**
  3. Click "Continue" → "Download ZIP"
- **Expected file:** `song crossarm.v6i.coco.zip`

#### 3️⃣ Utility-pole Dataset (218 images)
- **URL:** https://universe.roboflow.com/project-6kpfk/utility-pole-hdbuh
- **Steps:**
  1. Click "Download Dataset" button
  2. Select format: **COCO**
  3. Click "Continue" → "Download ZIP"
- **Expected file:** `utility-pole.v4i.coco.zip`

### Step 2: Save ZIP Files

Save all 3 ZIP files to:
```
c:\Users\Z0057P7S\OneDrive - Siemens Energy\Documents\MenonSiemens\DETR\utility-inventory-detr-main\
```

### Step 3: Extract and Process

Open PowerShell and run:

```powershell
# Navigate to DETR directory
cd "c:\Users\Z0057P7S\OneDrive - Siemens Energy\Documents\MenonSiemens\DETR\utility-inventory-detr-main"

# Activate Python environment
# (Use your conda/venv environment)

# Step 1: Check which files exist
python scripts\01_download_datasets.py

# Step 2: Extract ZIP files
python scripts\00_extract_datasets.py

# Step 3: Clean datasets (remove invalid annotations)
python scripts\02_clean_datasets.py

# Step 4: Merge into unified dataset
python scripts\03_merge_datasets.py
```

### Step 4: Upload to S3

After processing, open the notebook and re-run upload cells:
```
AWS\DataSelection\data_global.ipynb
```

The images will automatically be detected and uploaded to S3.

---

## 📥 Method 2: Use Roboflow API (If You Have Account)

### Step 1: Get API Key

1. Sign up at https://roboflow.com/ (free account)
2. Go to **Settings** → **API Keys**
3. Copy your API key

### Step 2: Set Environment Variable

In PowerShell:
```powershell
$env:ROBOFLOW_API_KEY = "your_api_key_here"
```

### Step 3: Run Automated Download

```powershell
cd "c:\Users\Z0057P7S\OneDrive - Siemens Energy\Documents\MenonSiemens\DETR\utility-inventory-detr-main"

# This will use the API to download automatically
python scripts\01_download_datasets.py

# Then extract and process
python scripts\00_extract_datasets.py
python scripts\02_clean_datasets.py
python scripts\03_merge_datasets.py
```

---

## 📥 Method 3: Request from Team

Contact **Vijay Kovuru (ext)** who completed the DETR training.

He may have:
- ✅ Original ZIP files already downloaded
- ✅ Extracted datasets with images
- ✅ Access to shared drive/cloud storage with full dataset

**Potential locations to check:**
- Team shared drive
- Previous training machine/VM
- Cloud storage (OneDrive, Google Drive, etc.)

---

## 📂 Expected Directory Structure After Download

Once complete, you should have:

```
DETR/utility-inventory-detr-main/
├── Insulators.v5i.coco.zip          ← Downloaded ZIP 1
├── song crossarm.v6i.coco.zip       ← Downloaded ZIP 2
├── utility-pole.v4i.coco.zip        ← Downloaded ZIP 3
│
├── datasets/
│   ├── raw/                         ← Extracted from ZIPs
│   │   ├── insulators/
│   │   │   ├── train/ (images + JSON)
│   │   │   ├── valid/ (images + JSON)
│   │   │   └── test/ (images + JSON)
│   │   ├── crossarm/
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   └── test/
│   │   └── utility-pole/
│   │       ├── train/
│   │       ├── valid/
│   │       └── test/
│   │
│   └── processed/
│       ├── insulators/              ← Cleaned datasets
│       ├── crossarm/
│       ├── utility-pole/
│       └── merged/                  ← Final unified dataset
│           ├── train/               (713 .jpg images + JSON)
│           ├── valid/               (134 .jpg images + JSON)
│           └── test/                (76 .jpg images + JSON)
│
└── scripts/                         ← Processing scripts
    ├── 01_download_datasets.py
    ├── 00_extract_datasets.py
    ├── 02_clean_datasets.py
    └── 03_merge_datasets.py
```

---

## 📊 Dataset Breakdown

| Dataset | Images | Annotations | Classes |
|---------|--------|-------------|---------|
| **Insulators** | 602 | 599 | insulators |
| **Crossarm** | 182 | 207 | crossarm |
| **Utility-pole** | 139 | 218 | utility-pole |
| **MERGED TOTAL** | **923** | **1,024** | **3 classes** |

### Train/Valid/Test Splits (After Merge)
- **Train:** 713 images (77.3%)
- **Valid:** 134 images (14.5%)
- **Test:** 76 images (8.2%)

---

## 🚀 Automated Processing Script

After downloading ZIPs, you can use the automated notebook cell:

Open: `AWS\DataSelection\data_global.ipynb`

Run the cell: **"🚀 Quick Start Script (Run After Downloading ZIPs)"**

This will automatically:
1. ✅ Check for ZIP files
2. ✅ Extract them
3. ✅ Clean datasets
4. ✅ Merge into unified format
5. ✅ Prepare for S3 upload

---

## ⏱️ Estimated Time & Size

| Task | Time | Size |
|------|------|------|
| Download ZIPs (manual) | 5-10 min | ~200-300 MB |
| Download ZIPs (API) | 2-5 min | ~200-300 MB |
| Extract ZIPs | 1-2 min | ~400-600 MB |
| Clean datasets | 30 sec | Same |
| Merge datasets | 1 min | ~200-500 MB |
| Upload to S3 | 5-15 min | ~200-500 MB |

**Total:** 15-30 minutes from start to S3 upload complete

---

## ✅ Verification Checklist

After processing, verify you have:

- [ ] 3 ZIP files downloaded
- [ ] `datasets/raw/` contains 3 extracted folders
- [ ] `datasets/processed/merged/train/` has ~713 .jpg files
- [ ] `datasets/processed/merged/valid/` has ~134 .jpg files
- [ ] `datasets/processed/merged/test/` has ~76 .jpg files
- [ ] Each folder has `_annotations.coco.json` file
- [ ] Total images = 923

**Test one image:**
```powershell
# Check if images exist
Get-ChildItem "c:\Users\Z0057P7S\OneDrive - Siemens Energy\Documents\MenonSiemens\DETR\utility-inventory-detr-main\datasets\processed\merged\train\*.jpg" | Select-Object -First 5
```

---

## 🆘 Troubleshooting

### Issue: "No ZIP files found"
**Solution:** Download manually from Roboflow URLs above

### Issue: "Roboflow API key invalid"
**Solution:** Use manual download method (Method 1)

### Issue: "Extraction failed"
**Solution:** Verify ZIP files are not corrupted, re-download if needed

### Issue: "Images still not uploading to S3"
**Solution:** 
1. Verify images exist: `Get-ChildItem datasets\processed\merged\train\*.jpg`
2. Check image count matches annotations in JSON
3. Re-run consolidation cell in notebook

---

## 📞 Support

**For dataset issues:**
- Roboflow Support: https://roboflow.com/contact

**For team coordination:**
- Vijay Kovuru (ext) - Completed DETR training
- Bhargav Bompalli (ext) - Working on YOLOv11-OBB
- Erick Allage (ext) - HITL workflow design

**Documentation:**
- DETR README: `DETR/utility-inventory-detr-main/README.md`
- Dataset README: `DETR/utility-inventory-detr-main/datasets/README.md`
- S3 Upload Notebook: `AWS/DataSelection/data_global.ipynb`
- Project Review: `REVIEW-DETR-AWS-DataSelection.md`

---

**Document Version:** 1.0  
**Last Updated:** December 26, 2025  
**Status:** ⏳ Images pending download, S3 infrastructure ready
