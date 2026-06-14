# OpenCap Gait Analysis App - Sequential Analysis

이 앱은 2~5번 분석 전용 Streamlit 앱입니다.

권장 입력은 로컬 전처리에서 생성한 subject-level mean curve long table입니다.

필수 컬럼:

- `subject_id`
- `feature`
- `grid_pct`
- `value`

선택 컬럼:

- `institution`
- `n_cycles_total`
- `n_cycles_kept`
- `input_format`

## 실행 순서

사이드바 버튼을 위에서 아래로 순차 실행합니다.

1. `① 데이터 로드/매핑`
2. `② FDA 실행`
3. `③ fPCA 실행`
4. `④ 임상 분석 실행`
5. `⑤ ML 실행`

각 단계가 끝나야 다음 단계 버튼이 활성화됩니다. FDA를 다시 실행하면 fPCA/임상/ML 결과는 자동으로 초기화됩니다. fPCA를 다시 실행하면 임상/ML 결과는 자동으로 초기화됩니다.

## 다운로드

다운로드 탭에서 현재 완료된 단계의 CSV 결과만 ZIP에 포함됩니다.

- FDA 완료 후: `01_fda/`
- fPCA 완료 후: `02_fpca/`
- 임상 완료 후: `03_clinical/`
- ML 완료 후: `04_ml/`

cycle-level `*_mot_by_cycle.csv`는 웹에서 변환할 수 있지만 느리므로 기본적으로 막아두었습니다. 실제 운영에서는 로컬에서 `*_subject_mean_curve_long.csv` 또는 `*_subject_mean_curve_upload_ready.zip`를 생성한 뒤 업로드하세요.

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```
