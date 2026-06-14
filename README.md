# OpenCap Gait 분석 웹서비스 - 2~5번 분석 전용

이 저장소의 `app.py`는 **분석 전용 Streamlit 앱**입니다. 원본 MOT/TRC 전처리(1번 작업)는 이 웹앱에서 수행하지 않습니다.

## 전체 작업 분리

```text
1번 작업: 로컬/서버 전처리
- 원본 MOT/TRC 수집
- 6m walking trial 선별
- 결측 보정
- spline smoothing
- 0~100% gait cycle 정규화
- 환자 내 이상 trajectory 제외
- 환자별 feature mean curve 생성
- preprocessed_gait_curves_long.csv 생성

2~5번 작업: 웹앱 분석
- CRF와 전처리 gait curve 매핑
- FDA/fPCA 분석
- 질환군 내 HY/UPDRS 임상척도 연결
- leakage-free ML 분석
- 전체 결과 ZIP 다운로드
```

## 웹앱 입력자료

웹앱에는 원본 `.mot`, `.trc` 파일을 올리지 않습니다.

기관별로 다음 파일을 업로드합니다.

```text
UNI 전처리 gait curve CSV/XLSX/ZIP + UNI CRF
UUH 전처리 gait curve CSV/XLSX/ZIP + UUH CRF
JBH 전처리 gait curve CSV/XLSX/ZIP + JBH CRF
```

전처리 gait curve 파일의 필수 컬럼은 다음 4개입니다.

```text
subject_id, feature, grid_pct, value
```

예시:

```csv
subject_id,feature,grid_pct,value,n_trials_total,n_trials_kept
UNI1,hip_flexion_r,0,10.23,3,3
UNI1,hip_flexion_r,1,10.45,3,3
UNI2,hip_flexion_r,0,8.91,2,2
```

## 그룹 분리

기관명으로 정상군/질환군을 나누지 않습니다. CRF의 `피험자군` 컬럼을 기본 그룹 컬럼으로 사용합니다.

예:

```text
피험자군 = Control
피험자군 = Parkinson
```

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```
