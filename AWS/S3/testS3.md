Dear SIEAERO Ceph Storage User

 

Here are the credentials you need to access our S3 Ceph storage (https://s3.ohl-inspection.com/).

{

    "user_id": "menonp",

    "email": prahlad.menon.ext@siemens-energy.com,

    "access_key": "PAWDFXP6IEDAFD305IPH",

    "secret_key": "txXMiiEC7sHfUyjTj6iShCVm6kC4KhiYFQUmIyak"

}


Please note that these credentials are essential for accessing your S3 buckets and managing your data on the Ceph platform. It is crucial that you keep these details secure and confidential at all times.

 

We would like to remind you of the importance of safeguarding your access credentials. Please adhere to the following security best practices:

 

1. Do not share your S3 access credentials with anyone.

2. Avoid sending credentials via unsecured channels.

3. Store your credentials in a safe place, such as a password manager.

 

If you have any questions about the new user management system or require assistance with your credentials, please do not hesitate to contact me.

 

Thank you!

 

With best regards,
Juergen Hatzl

Siemens Energy Austria GmbH
SE GT DG GET PPM R&D ED
Strassganger Str. 315
8054 Graz, AUSTRIA

---

## AWS S3 Access Test Results

**Date:** December 15, 2025  
**Tested by:** Prahlad Menon  
**Endpoint:** https://s3.ohl-inspection.com (Ceph Storage)

### Installation
- ✅ AWS CLI installed via pip in Miniconda environment
- Version: aws-cli/1.44.1 Python/3.13.9 Windows/11 botocore/1.42.11
- Location: `C:\Users\Z0057P7S\miniconda3\python.exe -m awscli`

### Credentials Configuration
```bash
AWS Access Key ID: PAWDFXP6IEDAFD305IPH
AWS Secret Access Key: [configured]
Endpoint URL: https://s3.ohl-inspection.com
```

### Test Results

#### 1. Authentication Test
```bash
Command: aws s3api list-buckets --endpoint-url https://s3.ohl-inspection.com
Status: ✅ SUCCESS
Owner: "Prahlad Menon" (menonp)
```

#### 2. Bucket Creation Test
```bash
Command: aws s3api create-bucket --bucket prahlad-test-bucket
Status: ✅ SUCCESS
Created: 2025-12-17 10:23:10
```

#### 3. File Upload Test
```bash
Command: aws s3 cp test-upload.txt s3://prahlad-test-bucket/
Status: ✅ SUCCESS
Size: 78 Bytes
Upload Time: 2025-12-17 10:25:19
```

#### 4. File Metadata Test
```bash
Command: aws s3api head-object --bucket prahlad-test-bucket --key test-upload.txt
Status: ✅ SUCCESS
Storage Class: DEEP_ARCHIVE (default for this Ceph storage)
Content Type: text/plain
ETag: "b7da4a23306cb1f685a6641b80959afd"
```

### Key Findings

1. **✅ Full S3 Access Confirmed** - Authentication, bucket creation, and file upload all working
2. **⚠️ Storage Class** - Default storage class is `DEEP_ARCHIVE`, which requires restoration before downloads
3. **✅ Endpoint Compatibility** - Ceph S3 storage compatible with AWS CLI
4. **📝 Usage Pattern** - Must specify `--endpoint-url https://s3.ohl-inspection.com` for all commands

### AWS CLI Command Reference

**Set endpoint as environment variable (optional):**
```powershell
$env:AWS_ENDPOINT_URL = "https://s3.ohl-inspection.com"
```

**Common commands:**
```bash
# List buckets
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 ls --endpoint-url https://s3.ohl-inspection.com

# List bucket contents
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 ls s3://bucket-name/ --endpoint-url https://s3.ohl-inspection.com

# Upload file
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 cp local-file.txt s3://bucket-name/ --endpoint-url https://s3.ohl-inspection.com

# Download file (if not in DEEP_ARCHIVE)
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 cp s3://bucket-name/file.txt ./ --endpoint-url https://s3.ohl-inspection.com

# Get object metadata
C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3api head-object --bucket bucket-name --key file.txt --endpoint-url https://s3.ohl-inspection.com
```

### Next Steps

- Contact Juergen Hatzl if different storage class needed (STANDARD vs DEEP_ARCHIVE)
- Set up profile-based configuration for easier access
- Investigate SIEAERO dataset availability on this storage
Mobile: +43 66488554084
mailto:juergen.hatzl@siemens-energy.com
siemens-energy.com
Logo
Company Name: Siemens Energy Austria GmbH; Managing Directors: Aleš Prešern, Michaela Länger; Legal Form: Limited Liability Company; Corporate Seat: Vienna, Austria; Commercial Register Number: FN 518270 m; Companies’ Register: Commercial Court Vienna
Siemens Energy is a trademark licensed by Siemens AG.

 