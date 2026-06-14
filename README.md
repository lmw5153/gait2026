# OpenCap Gait Analysis App - cycle-wide input version

This Streamlit app is for analysis steps after raw MOT/TRC preprocessing.
It does not parse raw MOT/TRC files.

## Supported gait input

Institution-specific preprocessed cycle-wide MOT tables:

```text
id, side, cycle, t0, t1, to, to_pct, phase, time,
pelvis_tilt, pelvis_list, pelvis_rotation, ...
```

The app internally converts cycle-wide rows into subject-level mean curves:

```text
institution, subject_id, feature, grid_pct, value
```

The original MOT variable names are preserved as feature names.

## Main workflow

1. Upload UNI/UUH/JBH cycle-wide gait CSV or ZIP files.
2. Upload institution-specific CRF files.
3. Click `① 데이터 로드/매핑`.
4. Check subject mapping and group labels.
5. Click `② 분석 시작`.
6. Review FDA/fPCA/clinical/ML results.
7. Download CSV result ZIP.

## Notes

- FDA tests are computed for all selected features.
- The FDA screen shows one selected feature graph at a time to avoid excessive rendering.
- Full FDA grid-level test results and significant intervals are included as CSV files in the result ZIP.
- Yellow FDA regions indicate feature-wise FDR-significant grid ranges after grid-wise Welch t-tests.
