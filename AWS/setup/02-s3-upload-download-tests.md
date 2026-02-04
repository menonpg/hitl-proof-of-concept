# S3 Upload and Download Tests

**Date:** December 17, 2025  
**Endpoint:** https://s3.ohl-inspection.com (Ceph Storage)  
**User:** menonp (Prahlad Menon)

---

## Test Results Summary

| Test | Status | Notes |
|------|--------|-------|
| Authentication | ✅ SUCCESS | Connected as menonp |
| List Buckets | ✅ SUCCESS | Empty bucket list initially |
| Create Bucket | ✅ SUCCESS | `prahlad-test-bucket` created |
| Upload File | ✅ SUCCESS | 78 bytes uploaded |
| List Objects | ✅ SUCCESS | File visible in bucket |
| Get Metadata | ✅ SUCCESS | Storage class: DEEP_ARCHIVE |
| Download File | ⚠️ PARTIAL | Blocked by DEEP_ARCHIVE storage class |

---

## Test 1: List Existing Buckets

```powershell
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 ls --endpoint-url https://s3.ohl-inspection.com
```

**Result:** No buckets initially (empty output)

**API Response:**
```powershell
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3api list-buckets --endpoint-url https://s3.ohl-inspection.com
```

```json
{
    "Buckets": [],
    "Owner": {
        "DisplayName": "\"Prahlad Menon\"",
        "ID": "menonp"
    },
    "Prefix": null
}
```

---

## Test 2: Create Test Bucket

### Attempt 1: With Location Constraint (Failed)

```powershell
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 mb s3://prahlad-test-bucket --endpoint-url https://s3.ohl-inspection.com
```

**Error:**
```
make_bucket failed: s3://prahlad-test-bucket An error occurred (InvalidLocationConstraint) when calling the CreateBucket operation: The eu-central-1 location constraint is not valid.
```

### Attempt 2: Without Location Constraint (Success)

Ceph storage doesn't use AWS regions. Remove region from config or use s3api without location constraint.

**Result:** Bucket created successfully

**Verification:**
```powershell
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 ls --endpoint-url https://s3.ohl-inspection.com
```

**Output:**
```
2025-12-17 10:23:10 prahlad-test-bucket
```

---

## Test 3: Upload File

### Create Test File

```powershell
"This is a test file for AWS S3 access verification on $(Get-Date)" | Out-File -FilePath "$env:TEMP\test-upload.txt" -Encoding UTF8
```

### Upload to S3

```powershell
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 cp "$env:TEMP\test-upload.txt" s3://prahlad-test-bucket/test-upload.txt --endpoint-url https://s3.ohl-inspection.com
```

**Output:**
```
Completed 78 Bytes/78 Bytes (55 Bytes/s) with 1 file(s)
upload: ..\..\..\AppData\Local\Temp\test-upload.txt to s3://prahlad-test-bucket/test-upload.txt
```

**Status:** ✅ SUCCESS (78 bytes uploaded)

---

## Test 4: List Bucket Contents

```powershell
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 ls s3://prahlad-test-bucket/ --endpoint-url https://s3.ohl-inspection.com
```

**Output:**
```
2025-12-17 10:25:19         78 test-upload.txt
```

**Status:** ✅ File visible in bucket

---

## Test 5: Get Object Metadata

```powershell
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3api head-object --bucket prahlad-test-bucket --key test-upload.txt --endpoint-url https://s3.ohl-inspection.com
```

**Output:**
```json
{
    "AcceptRanges": "bytes",
    "LastModified": "Wed, 17 Dec 2025 15:25:19 GMT",
    "ContentLength": 78,
    "ETag": "\"b7da4a23306cb1f685a6641b80959afd\"",
    "ContentEncoding": "aws-chunked",
    "ContentType": "text/plain",
    "Metadata": {},
    "StorageClass": "DEEP_ARCHIVE"
}
```

**Key Finding:** Default storage class is `DEEP_ARCHIVE`

---

## Test 6: Download File

### Attempt: Standard Download

```powershell
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 cp s3://prahlad-test-bucket/test-upload.txt "$env:TEMP\test-download.txt" --endpoint-url https://s3.ohl-inspection.com
```

**Error:**
```
warning: Skipping file s3://prahlad-test-bucket/test-upload.txt. 
Object is of storage class GLACIER. Unable to perform download operations on GLACIER objects. 
You must restore the object to be able to perform the operation.
```

**Status:** ⚠️ BLOCKED by DEEP_ARCHIVE storage class

---

## Storage Class Issue Analysis

### What is DEEP_ARCHIVE?

- **AWS Equivalent:** S3 Glacier Deep Archive
- **Purpose:** Long-term cold storage (lowest cost, highest retrieval time)
- **Retrieval:** Requires restoration before download (12-48 hours typically)
- **Cost:** Cheapest storage, but expensive retrieval

### Why is Ceph Using DEEP_ARCHIVE?

The SIEAERO Ceph storage appears to be configured with DEEP_ARCHIVE as the default storage class. This is unusual for active datasets.

### Solutions

#### Option 1: Request Standard Storage Class

Contact Juergen Hatzl to configure buckets with `STANDARD` storage class:

```bash
# When creating bucket, specify storage class
--storage-class STANDARD
```

#### Option 2: Restore Objects Before Download

For GLACIER/DEEP_ARCHIVE objects, must initiate restore:

```powershell
# Initiate restore (not tested - may not work with Ceph)
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3api restore-object `
  --bucket prahlad-test-bucket `
  --key test-upload.txt `
  --restore-request Days=7 `
  --endpoint-url https://s3.ohl-inspection.com
```

#### Option 3: Upload with Explicit Storage Class

```powershell
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 cp "$env:TEMP\test-upload.txt" `
  s3://prahlad-test-bucket/test-standard.txt `
  --storage-class STANDARD `
  --endpoint-url https://s3.ohl-inspection.com
```

---

## Complete Upload/Download Test Script

```powershell
# Set endpoint as variable for convenience
$ENDPOINT = "https://s3.ohl-inspection.com"
$AWS_CMD = "C:\Users\Z0057P7S\miniconda3\python.exe -m awscli"

# Test 1: Create test file
Write-Host "`n=== Creating test file ===" -ForegroundColor Cyan
$testContent = "Test upload at $(Get-Date)`nUser: $env:USERNAME"
$testFile = "$env:TEMP\s3-test-$(Get-Date -Format 'yyyyMMdd-HHmmss').txt"
$testContent | Out-File -FilePath $testFile -Encoding UTF8
Write-Host "Created: $testFile" -ForegroundColor Green

# Test 2: Upload with STANDARD storage class
Write-Host "`n=== Uploading to S3 ===" -ForegroundColor Cyan
& $AWS_CMD s3 cp $testFile "s3://prahlad-test-bucket/test-$(Get-Date -Format 'yyyyMMdd-HHmmss').txt" `
  --storage-class STANDARD `
  --endpoint-url $ENDPOINT

# Test 3: List bucket contents
Write-Host "`n=== Listing bucket contents ===" -ForegroundColor Cyan
& $AWS_CMD s3 ls "s3://prahlad-test-bucket/" --endpoint-url $ENDPOINT

# Test 4: Download file
Write-Host "`n=== Downloading from S3 ===" -ForegroundColor Cyan
$downloadFile = "$env:TEMP\s3-download-$(Get-Date -Format 'yyyyMMdd-HHmmss').txt"
& $AWS_CMD s3 cp "s3://prahlad-test-bucket/test-$(Get-Date -Format 'yyyyMMdd-HHmmss').txt" $downloadFile `
  --endpoint-url $ENDPOINT

# Test 5: Verify download
if (Test-Path $downloadFile) {
    Write-Host "`n=== Download successful ===" -ForegroundColor Green
    Get-Content $downloadFile
} else {
    Write-Host "`n=== Download failed ===" -ForegroundColor Red
}

# Cleanup
Write-Host "`n=== Cleanup ===" -ForegroundColor Cyan
Remove-Item $testFile -ErrorAction SilentlyContinue
Remove-Item $downloadFile -ErrorAction SilentlyContinue
```

---

## Recommendations

1. **Contact Juergen Hatzl** to configure bucket with `STANDARD` storage class for active development
2. **Use `--storage-class STANDARD`** flag when uploading files for immediate access
3. **Reserve DEEP_ARCHIVE** for long-term archival datasets (SIEAERO historical imagery)
4. **Document storage class policy** for team (when to use STANDARD vs DEEP_ARCHIVE)

---

## Next Steps

- Test with `STANDARD` storage class after configuration change
- Verify download functionality with STANDARD objects
- Test large file uploads (>1GB) for SIEAERO imagery datasets
- Document bucket naming conventions and access policies

---

## Contact

**For Storage Configuration Issues:**  
Juergen Hatzl  
Email: juergen.hatzl@siemens-energy.com  
Role: Ceph Storage Administrator, SIEAERO Team
