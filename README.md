# OpenCap Gait Analysis Web Service

Streamlit 기반 OpenCap walking MOT/TRC + CRF 분석 웹서비스입니다.

## 권장 업로드 방식

온라인 웹앱에서는 폴더를 그대로 업로드할 수 없으므로 **기관별 검사 데이터 폴더를 ZIP으로 압축해서 업로드**합니다. GitHub에는 데이터가 아니라 코드만 올립니다.

앱 사이드바에서 기관별로 다음 2가지를 각각 업로드합니다.

1. 기관 검사 데이터 ZIP
2. 기관 CRF 파일(xlsx/xls/csv)

예시:

```text
UNI_gait.zip
├─ UNI1/
│  ├─ 6m_1.mot
│  ├─ 6m_1.trc
│  ├─ 6m_2.mot
│  └─ 6m_2.trc
├─ UNI2/
│  ├─ 6m_1.mot
│  └─ 6m_1.trc
└─ UNI3/
   ├─ 6m_1(1).mot
   └─ 6m_1(1).trc
```

또는 ZIP 내부에 기관 폴더가 한 번 더 있어도 됩니다.

```text
UNI_gait.zip
└─ UNI/
   ├─ UNI1/
   │  ├─ 6m_1.mot
   │  └─ 6m_1.trc
   └─ UNI2/
      ├─ 6m_1.mot
      └─ 6m_1.trc
```

## 매핑 규칙

- `institution`: 앱에서 선택한 기관 코드, 예: `UNI`, `UUH`, `JBH`
- `subject_id`: ZIP 내부 비식별 폴더명에서 추출, 예: `UNI1`, `UUH1`, `JBH1`
- `trial_id`: 파일명에서 추출, 예: `6m_1`, `6m_2`, `6m_3`
- MOT/TRC 구분: 같은 폴더 내 파일 확장자 `.mot`, `.trc`로 구분
- walking 선별: 기본 정규식 `^6m[_\- ]*\d*`에 맞는 파일만 분석에 포함
- `TUG_1`, `TUG_2`, standing 등은 자동 제외

## 분석 단위

분석은 trial 단위가 아니라 **환자 단위 평균 curve**를 사용합니다.

```text
각 subject의 모든 6m walking trial
→ trial별 결측 보정
→ 0~100% gait cycle 정규화
→ spline smoothing
→ subject 내부 outlier trial 제외
→ 남은 trial 평균
→ subject-feature mean curve 생성
→ FDA/fPCA/임상척도/ML 분석
```

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 배포

1. 이 저장소를 GitHub에 push
2. Streamlit Community Cloud에서 New app
3. Repository 선택
4. Branch: `main`
5. Main file path: `app.py`
6. Deploy

## 주의

- 개인정보/원자료(MOT/TRC/CRF)는 GitHub에 올리지 않습니다.
- 온라인 웹앱에서 업로드할 때만 ZIP으로 올립니다.
- 대용량 자료는 서버 메모리 제한에 걸릴 수 있으므로, 기관별로 나누어 업로드하는 것을 권장합니다.
