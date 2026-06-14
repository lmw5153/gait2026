# OpenCap Gait 분석 웹서비스

OpenCap walking 검사 자료(`.mot`, `.trc`)와 기관별 CRF를 업로드하여 gait curve 전처리, FDA/fPCA, 임상척도 연결, 누수 방지 ML 분석을 수행하는 Streamlit 앱입니다.

## 입력 구조

온라인 웹앱에서는 폴더를 직접 올리지 않고 **기관별 검사 데이터 ZIP**과 **기관별 CRF**를 업로드합니다.

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

- MOT와 TRC는 같은 비식별 폴더 안에 있어도 됩니다.
- 앱은 확장자로 `.mot`/`.trc`를 구분합니다.
- `6m`, `6m_1`, `6m_2`, `6m_1(1)`처럼 walking trial만 분석합니다.
- `TUG`, `standing`, 기타 검사는 자동 제외됩니다.

CRF는 기관별로 업로드합니다.

```text
UNI_CRF.xlsx
UUH_CRF.xlsx
JBH_CRF.xlsx
```

## 그룹 분리

기관 폴더명으로 정상군/질환군을 나누지 않습니다. 기관 내부에 정상군과 질환군이 섞여 있을 수 있으므로, CRF의 **`피험자군`** 변수를 기본 그룹 컬럼으로 사용합니다.

예:

```text
피험자군 = Control
피험자군 = Parkinson
```

## CRF trial 상태값

CRF의 `6m_1`, `6m_2`, `6m_3`에 기록된 사용 가능 여부를 진단합니다.

- 기본값: `O`, `△` trial만 분석 포함
- `X`, `-`, blank 등은 분석 제외 또는 확인 대상으로 표시

## 업로드 용량

`.streamlit/config.toml`에서 Streamlit 기본 200MB 업로드 제한을 4096MB로 올렸습니다.

```toml
[server]
maxUploadSize = 4096
```

단, 온라인 배포 서버의 실제 메모리/디스크 제한은 별도로 영향을 줄 수 있습니다. 300MB 이상의 ZIP을 자주 처리한다면 Streamlit Community Cloud보다 Docker/VPS/기관 내부 서버 배포가 더 안정적일 수 있습니다.

## 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 분석 흐름

```text
기관별 ZIP + 기관별 CRF 업로드
→ ZIP 내부 비식별 폴더에서 subject_id 추출
→ 6m walking trial만 선별
→ MOT/TRC pair 진단
→ CRF subject/trial 매핑 진단
→ 환자별 모든 usable walking MOT trial 통합
→ 결측 보정, spline smoothing, 0~100% 정규화
→ 환자 내 이상 trajectory 제외
→ 환자당 feature별 mean curve 생성
→ 공변량 보정 후 FDA/fPCA
→ 질환군 내 HY/UPDRS 임상 연결
→ fold 내부 fPCA 기반 leakage-free ML
→ 전체 테이블/그래프/데이터 ZIP 다운로드
```
