# OpenCap Gait Analysis App

이 앱은 1단계 전처리 산출물을 받아 2~5번 분석(FDA/fPCA/임상척도/ML)을 수행하는 Streamlit 앱입니다.

## 지원 입력

### 1) cycle별 MOT wide table
샘플 형식:

```text
id, side, cycle, t0, t1, to, to_pct, phase, time, pelvis_tilt, ..., hip_flexion_r, ...
```

앱 내부에서 `id` 예: `UUH1_6m_1`에서 `UUH1`을 subject_id로 추출하고, cycle별 time을 0~100% grid로 정규화한 뒤 원본 MOT 변수명을 그대로 feature로 유지하여 subject-level mean curve를 만듭니다.

### 2) subject mean long table

```text
subject_id, feature, grid_pct, value
```

## FDA

FDA 화면은 선택형이 아니라 전체 feature를 표시합니다. 그룹 간 grid별 Welch t-test 후 feature별 FDR q-value가 alpha 미만인 구간을 노란색 영역으로 표시하고, grid별 검정 테이블과 유의구간 테이블을 제공합니다.

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```
