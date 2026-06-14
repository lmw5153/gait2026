# OpenCap Gait 분석 웹서비스 - Analysis-only 버전

이 버전은 Streamlit Cloud 메모리 한계를 피하기 위해 **원본 MOT/TRC 파싱과 웹 분석을 분리**합니다.
웹앱은 원본 `.mot`, `.trc` ZIP을 받지 않고, 별도 전처리 파이프라인에서 생성한 **환자별 mean gait curve**와 기관별 CRF만 업로드받아 분석합니다.

## 입력 자료

### 1) 전처리된 gait curve 데이터

CSV, XLSX 또는 여러 CSV/XLSX를 묶은 ZIP을 업로드할 수 있습니다.
필수 long-format 컬럼은 아래 4개입니다.

```text
subject_id, feature, grid_pct, value
```

예시:

```text
subject_id,feature,grid_pct,value,n_trials_total,n_trials_kept
UNI1,hip_flexion_r,0,10.23,3,3
UNI1,hip_flexion_r,1,10.45,3,3
UNI1,hip_flexion_r,2,10.82,3,3
UNI2,hip_flexion_r,0,8.91,2,2
```

이 데이터는 원본 MOT/TRC에서 다음 작업이 끝난 결과여야 합니다.

```text
원본 MOT/TRC
→ walking 6m trial만 선별
→ 결측 보정
→ spline smoothing
→ 0~100% gait cycle 정규화
→ 환자 내 이상 trajectory 제외
→ 환자당 feature별 mean curve 생성
→ preprocessed_gait_curves_long.csv 저장
```

### 2) 기관별 CRF

기관별 CRF를 각각 업로드합니다.

```text
UNI_CRF.xlsx
UUH_CRF.xlsx
JBH_CRF.xlsx
```

그룹 분리는 기관명이 아니라 CRF의 `피험자군` 컬럼을 기본으로 사용합니다.

```text
피험자군 = Control
피험자군 = Parkinson
```

## 웹앱 분석 기능

```text
전처리 gait curve + CRF 업로드
→ subject_id 매핑 확인
→ 그룹/공변량/feature 선택
→ FDA mean curve 및 유의구간
→ fPCA score/loading/explained variance
→ FPC1-FPC2 분포 검정
→ 질환군 내 임상척도 상관분석
→ fold 내부 공변량 보정 + PCA 기반 leakage-free ML
→ 전체 결과 ZIP 다운로드
```

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 왜 analysis-only로 분리했나?

Streamlit Cloud에서 300MB 이상의 원본 MOT/TRC ZIP을 직접 파싱하면 압축 해제 및 DataFrame 변환 과정에서 메모리가 수 GB까지 증가할 수 있습니다. 따라서 대용량 원자료 처리는 오프라인/서버 전처리 단계로 분리하고, 웹앱은 전처리 결과만 받아 분석하도록 구성합니다.
