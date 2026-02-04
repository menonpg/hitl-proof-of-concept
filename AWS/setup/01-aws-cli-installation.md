# AWS CLI Installation and Setup Guide

**Date:** December 15-17, 2025  
**Purpose:** Configure AWS CLI for accessing SIEAERO Ceph S3 Storage  
**Endpoint:** https://s3.ohl-inspection.com

---

## Prerequisites

- Miniconda3 installed at `C:\Users\Z0057P7S\miniconda3`
- PowerShell 5.1 or later
- Internet connection
- Valid AWS credentials from SIEAERO team

---

## Installation Steps

### Step 1: Install AWS CLI via pip

Organization policies may block MSI installations, so use pip from Miniconda:

```powershell
# Install AWS CLI using pip from Miniconda
C:\Users\Z0057P7S\miniconda3\Scripts\pip.exe install awscli
```

**Expected Output:**
```
Successfully installed PyYAML-6.0.3 awscli-1.44.1 botocore-1.42.11 docutils-0.19 jmespath-1.0.1 pyasn1-0.6.1 rsa-4.7.2 s3transfer-0.16.0
```

### Step 2: Verify Installation

```powershell
# Check AWS CLI version
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli --version
```

**Expected Output:**
```
aws-cli/1.44.1 Python/3.13.9 Windows/11 botocore/1.42.11
```

---

## Configuration

### Step 3: Configure AWS Credentials

**Credentials Location:** See `C:\Users\Z0057P7S\OneDrive - Siemens Energy\Documents\MenonSiemens\AWS\S3\testS3.md`

```powershell
# Configure access key
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli configure set aws_access_key_id "YOUR_ACCESS_KEY"

# Configure secret key
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli configure set aws_secret_access_key "YOUR_SECRET_KEY"

# Set default region (optional, not critical for Ceph)
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli configure set region "eu-central-1"
```

### Step 4: Verify Configuration

```powershell
# List configuration
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli configure list
```

**Expected Output:**
```
      Name                    Value             Type    Location
      ----                    -----             ----    --------
   profile                <not set>             None    None
access_key     ****************05IPH shared-credentials-file
secret_key     ****************Iyak shared-credentials-file
    region                eu-central-1      config-file    ~/.aws/config
```

---

## Testing Connection

### Test 1: Verify Authentication

```powershell
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3api list-buckets --endpoint-url https://s3.ohl-inspection.com
```

**Expected Success:**
```json
{
    "Buckets": [],
    "Owner": {
        "DisplayName": "\"Prahlad Menon\"",
        "ID": "menonp"
    }
}
```

**Expected Failure (No Credentials):**
```
Unable to locate credentials. You can configure credentials by running "aws configure".
```

---

## Common Issues and Solutions

### Issue 1: Organization Policy Blocking MSI Installation

**Error:**
```
Organization policies are preventing installation. Contact your admin.
Installer failed with exit code: 1625
```

**Solution:** Use pip installation instead of MSI/winget

---

### Issue 2: AWS CLI Not in PATH

**Error:**
```
aws : The term 'aws' is not recognized as the name of a cmdlet
```

**Solution:** Use full path to python module:
```powershell
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli <command>
```

---

### Issue 3: Conda Terms of Service Not Accepted

**Error:**
```
CondaToSNonInteractiveError: Terms of Service have not been accepted
```

**Solution:** Use pip instead of conda for installation

---

## Creating Command Aliases (Optional)

To simplify commands, create a PowerShell alias:

```powershell
# Add to PowerShell profile
function aws { C:\Users\Z0057P7S\miniconda3\python.exe -m awscli $args }

# Or create a batch file
@echo off
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli %*
```

---

## Security Best Practices

1. ✅ **Never commit credentials** to version control
2. ✅ **Store credentials** in AWS credentials file (`~/.aws/credentials`)
3. ✅ **Use environment variables** for temporary access
4. ✅ **Rotate credentials** periodically
5. ✅ **Use IAM roles** when possible (for EC2/Lambda)

---

## Next Steps

- See `02-s3-upload-download-tests.md` for file operations testing
- See `03-bucket-management.md` for bucket creation and management
- Contact Juergen Hatzl (juergen.hatzl@siemens-energy.com) for credential issues

---

## References

- AWS CLI Documentation: https://docs.aws.amazon.com/cli/
- Ceph S3 Compatibility: https://docs.ceph.com/en/latest/radosgw/s3/
- SIEAERO S3 Endpoint: https://s3.ohl-inspection.com
