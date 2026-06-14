# OpenCap Gait 분석 웹서비스

OpenCap walking 검사 자료(`.mot`, `.trc`)와 기관별 CRF를 업로드하여 gait curve 전처리, FDA/fPCA, 임상척도 연결, 누수 방지 ML 분석을 수행하는 Streamlit 앱입니다.

## 핵심 실행 방식

대용량 MOT/TRC ZIP을 매번 다시 파싱하지 않도록 실행 단계를 분리했습니다.

1. 기관별 검사 데이터 ZIP과 CRF를 업로드합니다.
2. 사이드바의 **① 업로드 자료 파싱/매핑 실행** 버튼을 누릅니다.
   - ZIP 내부 파일을 한 번만 파싱합니다.
   - 6m walking trial만 선별합니다.
   - MOT/TRC pair와 CRF 매핑을 진단합니다.
   - 파싱 결과는 `st.session_state`에 저장됩니다.
3. 파라미터와 분석 변수를 조절합니다.
4. 사이드바의 **② 분석 시작** 버튼을 누릅니다.
   - 저장된 파싱 결과를 재사용합니다.
   - ZIP을 다시 풀거나 MOT/TRC를 다시 읽지 않습니다.
   - 전처리와 FDA/fPCA가 실행됩니다.
5. ML은 계산량이 커서 기존처럼 ML 탭에서 별도 버튼으로 실행합니다.

업로드 파일이나 walking/CRF 필터 옵션을 변경하면 다시 **① 업로드 자료 파싱/매핑 실행**을 눌러야 합니다.

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
→ ① 업로드 자료 파싱/매핑 실행
→ ZIP 내부 비식별 폴더에서 subject_id 추출
→ 6m walking trial만 선별
→ MOT/TRC pair 진단
→ CRF subject/trial 매핑 진단
→ 파싱 결과 session_state 저장
→ 파라미터/feature/covariate 조절
→ ② 분석 시작
→ 저장된 파싱 결과 재사용
→ 환자별 모든 usable walking MOT trial 통합
→ 결측 보정, spline smoothing, 0~100% 정규화
→ 환자 내 이상 trajectory 제외
→ 환자당 feature별 mean curve 생성
→ 공변량 보정 후 FDA/fPCA
→ 질환군 내 HY/UPDRS 임상 연결
→ fold 내부 fPCA 기반 leakage-free ML
→ 전체 분석 자료 ZIP 생성/갱신 버튼 클릭
→ 전체 테이블/그래프/데이터 ZIP 다운로드
```

## 성능 관련 변경

- 업로드 ZIP/CRF 파싱은 사이드바의 **① 업로드 자료 파싱/매핑 실행**을 눌렀을 때만 수행됩니다.
- 전처리/FDA/fPCA는 사이드바의 **② 분석 시작** 또는 각 탭의 개별 실행 버튼을 눌렀을 때만 수행됩니다.
- 전체 결과 ZIP도 **전체 분석 자료 ZIP 생성/갱신** 버튼을 눌렀을 때만 생성됩니다.
