# Data access (ADNI)

The authors cannot redistribute ADNI data.

1. Go to https://adni.loni.usc.edu and choose **Access Data → Apply for access**.
2. Complete the online application and accept the ADNI Data Use Agreement.
3. Once approved, download the source tables listed in `manifests/stage47_feature_source_and_circularity_inventory.csv` (for example DXSUM, PTDEMOG, CDR, FAQ, MMSE, MoCA and the CSF biomarker tables).
4. Place them in `data/` (git-ignored). The scripts check file hashes against `results/stage4_4_temporal_alignment/source_sha256.csv`.
