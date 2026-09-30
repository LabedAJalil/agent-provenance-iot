# Dataset

CIC-APT-IIoT-2024 is not redistributed here. Download it from the Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/iiot-dataset-2024.html

Only the provenance logs are used. Place them as:

```
Dataset/Provenance_Logs/Phase1_Provenance.csv   md5 df3da3d0cbd8b0afbc3eb3965388508e
Dataset/Provenance_Logs/Phase2_Provenance.csv   md5 620138736b62d0dec299a995d4ced32f
```

Check with `md5 Dataset/Provenance_Logs/*.csv` (macOS) or `md5sum Dataset/Provenance_Logs/*.csv` (Linux).
Every `metrics.json` records these sums in `manifest.dataset_md5`.
