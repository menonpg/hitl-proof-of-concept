(base) PS C:\Users\Z0057P7S\OneDrive - Siemens Energy\Documents\MenonSiemens> C:\Users\Z0057P7S\miniconda3\python.exe -m awscli configure list --profile ceph
      Name                    Value             Type    Location
      ----                    -----             ----    --------
   profile                     ceph           manual    --profile        

The config profile (ceph) could not be found
(base) PS C:\Users\Z0057P7S\OneDrive - Siemens Energy\Documents\MenonSiemens> C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 ls s3://2024-04--teco--data --endpoint-url https://s3.ohl-inspection.com
                           PRE 2024-04--teco_Segment-04-2_20250527/
                           PRE 2024-04--teco_Segment-05_20250527/
                           PRE 2024-04--teco_Segment-13_20250604/        
                           PRE 2024-04--teco_Segment-14_20250604/        
                           PRE 2024-04_TECO_Segment-04-1_20240402/       
                           PRE 2024-04_TECO_Segment-04-2_20240614/       
                           PRE 2024-04_TECO_Segment-05_20240614/
                           PRE 2024-04_TECO_Segment-13_20240530/
                           PRE 2024-04_TECO_Segment-13_20250604/
                           PRE 2024-04_TECO_Segment-14_20240528/
                           PRE 2024-04_TECO_Segment-14_20250604/
                           PRE 2024-04_TECO_Segment-15_20240402/
2025-06-04 08:12:30         24 test.txt
(base) PS C:\Users\Z0057P7S\OneDrive - Siemens Energy\Documents\MenonSiemens> C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 ls s3://2024-04--teco--data --endpoint-url https://s3.ohl-inspection.com --recursive --human-readable --summarize | Select-Object -First 50
2025-05-27 09:12:56    3.8 KiB 2024-04--teco_Segment-04-2_20250527/OHLI_config.json
2025-05-27 09:12:56   17.4 KiB 2024-04--teco_Segment-04-2_20250527/data/div/Segment-04-2.geojson
2025-05-27 09:12:56    0 Bytes 2024-04--teco_Segment-04-2_20250527/data/div/crossings.csv
2025-05-27 09:12:56  152 Bytes 2024-04--teco_Segment-04-2_20250527/data/div/images_crs.json
2025-06-06 02:04:17  408.2 MiB 2024-04--teco_Segment-04-2_20250527/data/div/images_ir.json
2025-06-06 02:04:17  214.3 MiB 2024-04--teco_Segment-04-2_20250527/data/div/images_rgb.json
2025-05-27 09:12:56   68 Bytes 2024-04--teco_Segment-04-2_20250527/data/div/images_uv.json
2025-05-27 09:12:56  805 Bytes 2024-04--teco_Segment-04-2_20250527/data/div/pylon_connectivity.csv
2025-05-27 09:12:56    1.6 KiB 2024-04--teco_Segment-04-2_20250527/data/div/substations.geojson
2025-05-27 09:12:56    3.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/IR/M0000090/M0000090_20240629.npy
2025-06-06 02:04:17  115.2 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/IR/M0000090/image_orientations.json
2025-05-27 09:12:56    3.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/IR/M0000095/M0000095_20240629.npy
2025-06-06 02:04:19  111.3 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/IR/M0000095/image_orientations.json
2025-05-27 09:12:56    3.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/IR/M0000107/M0000107_20240629.npy
2025-06-06 02:04:19  111.4 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/IR/M0000107/image_orientations.json
2025-05-27 09:13:05    3.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/IR/M0000108/M0000108_20240629.npy
2025-06-06 02:04:20  115.3 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/IR/M0000108/image_orientations.json
2025-05-27 09:13:10    3.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/IR/M0000204/M0000204_20240629.npy
2025-06-06 02:04:21  111.4 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/IR/M0000204/image_orientations.json
2025-06-06 02:04:22   13.7 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40143959/image_orientations.json
2025-06-06 02:04:22   13.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40143965/image_orientations.json
2025-06-06 02:04:22   13.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40143970/image_orientations.json
2025-06-06 02:04:23   13.7 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40143976/image_orientations.json
2025-06-06 02:04:23   13.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144430/image_orientations.json
2025-06-06 02:04:24   13.7 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144431/image_orientations.json
2025-06-06 02:04:24   13.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144436/image_orientations.json
2025-06-06 02:04:25   13.7 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144437/image_orientations.json
2025-06-06 02:04:25   13.7 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144438/image_orientations.json
2025-06-06 02:04:26   13.7 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144439/image_orientations.json
2025-06-06 02:04:27   13.7 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144441/image_orientations.json
2025-06-06 02:04:28   13.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144442/image_orientations.json
2025-06-06 02:04:28   13.7 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144443/image_orientations.json
2025-06-06 02:04:29   13.7 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144445/image_orientations.json
2025-06-06 02:04:29   13.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144446/image_orientations.json
2025-06-06 02:04:30   13.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144447/image_orientations.json
2025-06-06 02:04:31   13.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144449/image_orientations.json
2025-06-06 02:04:32   13.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144450/image_orientations.json
2025-06-06 02:04:33   13.5 MiB 2024-04--teco_Segment-04-2_20250527/data/images/2024-03-04/RGB/40144454/image_orientations.json
2025-06-06 02:04:33  803.3 MiB 2024-04--teco_Segment-04-2_20250527/data/pointclouds/20240304_002_151556-151631_Segment-04-2.las
2025-06-06 02:04:33   31.2 GiB 2024-04--teco_Segment-04-2_20250527/data/pointclouds/20240304_002_151845-154424_Segment-04-2.las
2025-06-06 02:04:34    1.9 GiB 2024-04--teco_Segment-04-2_20250527/data/pointclouds/20240304_002_154425-154544_Segment-04-2.las
2025-06-06 02:04:35  516.1 MiB 2024-04--teco_Segment-04-2_20250527/data/pointclouds/20240304_002_154545-154587_Segment-04-2.las
2025-06-06 02:04:35  621.1 MiB 2024-04--teco_Segment-04-2_20250527/data/pointclouds/20240304_002_154588-154648_Segment-04-2.las
2025-06-06 02:04:36  613.7 MiB 2024-04--teco_Segment-04-2_20250527/data/pointclouds/20240304_002_154649-154701_Segment-04-2.las
2025-06-06 02:04:36  329.2 MiB 2024-04--teco_Segment-04-2_20250527/data/pointclouds/20240304_002_154702-154727_Segment-04-2.las
2025-06-06 02:04:37   15.4 GiB 2024-04--teco_Segment-04-2_20250527/data/pointclouds/20240304_002_154728-155915_Segment-04-2.las
2025-06-06 02:04:38    7.4 GiB 2024-04--teco_Segment-04-2_20250527/data/pointclouds/20240304_002_155916-156503_Segment-04-2.las
2025-06-06 02:04:39    8.5 GiB 2024-04--teco_Segment-04-2_20250527/data/pointclouds/20240304_002_156504-157156_Segment-04-2.las
2025-06-06 02:04:39    1.6 GiB 2024-04--teco_Segment-04-2_20250527/data/pointclouds/20240304_002_157157-157303_Segment-04-2.las
2025-06-06 02:04:40   64.7 MiB 2024-04--teco_Segment-04-2_20250527/data/pointclouds/corrected_trajectory_GPS_ToW_Segment-04-2.trj
(base) PS C:\Users\Z0057P7S\OneDrive - Siemens Energy\Documents\MenonSiemens> Write-Host "`n========== TECO Bucket Access Summary ==========`n" -ForegroundColor Green; C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 ls s3://2024-04--teco--data --endpoint-url https://s3.ohl-inspection.com | Format-Table -AutoSize

========== TECO Bucket Access Summary ==========

                           PRE 2024-04--teco_Segment-04-2_20250527/
                           PRE 2024-04--teco_Segment-05_20250527/        
                           PRE 2024-04--teco_Segment-13_20250604/        
                           PRE 2024-04--teco_Segment-14_20250604/        
                           PRE 2024-04_TECO_Segment-04-1_20240402/       
                           PRE 2024-04_TECO_Segment-04-2_20240614/       
                           PRE 2024-04_TECO_Segment-05_20240614/
                           PRE 2024-04_TECO_Segment-13_20240530/
                           PRE 2024-04_TECO_Segment-13_20250604/
                           PRE 2024-04_TECO_Segment-14_20240528/
                           PRE 2024-04_TECO_Segment-14_20250604/
                           PRE 2024-04_TECO_Segment-15_20240402/
2025-06-04 08:12:30         24 test.txt
(base) PS C:\Users\Z0057P7S\OneDrive - Siemens Energy\Documents\MenonSiemens>

(base) PS C:\Users\Z0057P7S\OneDrive - Siemens Energy\Documents\MenonSiemens> Write-Host "`nExploring Sample Segment Structure:`n" -ForegroundColor Cyan; C:\Users\Z0057P7S\miniconda3\python.exe -m awscli s3 ls s3://2024-04--teco--data/2024-04--teco_Segment-04-2_20250527/ --endpoint-url https://s3.ohl-inspection.com

Exploring Sample Segment Structure:

                           PRE data/
                           PRE dev-results/
                           PRE results/
2025-05-27 09:12:56       3936 OHLI_config.json
(base) PS C:\Users\Z0057P7S\OneDrive - Siemens Energy\Documents\MenonSiemens>