# OpenCap Gait 분석 웹서비스

Streamlit 기반 OpenCap walking gait 분석 웹서비스입니다. 기관별 OpenCap MOT/TRC 파일과 CRF 파일을 업로드하면 walking trial(`6m`, `6m_1`, `6m_2` 등)만 선별하고, 환자별 여러 trial을 통합하여 mean curve를 만든 뒤 FDA, fPCA, 임상척도 연계 분석, ML 분석을 수행합니다.

## 1. 권장 입력 구조

데이터는 GitHub에 올리지 말고, 앱 화면에서 ZIP으로 업로드하는 것을 권장합니다. ZIP 내부 구조는 아래처럼 구성합니다.

```text
OpenCap_DATA.zip
├─ UNI/
│  └─ UNI1/
│     ├─ 6m_1.mot
│     ├─ 6m_1.trc
│     ├─ 6m_2.mot
│     └─ 6m_2.trc
├─ UUH/
│  └─ UUH1/
│     ├─ 6m_1.mot
│     └─ 6m_1.trc
├─ JBH/
│  └─ JBH1/
│     ├─ 6m_1.mot
│     └─ 6m_1.trc
└─ CRF/
   └─ OpenCap협력연구_피험자정보.xlsx
```

앱은 ZIP 내부 경로에서 `institution`, `subject_id`, `trial_id`를 자동 추출합니다.

```text
UNI/UNI1/6m_1.mot → institution=UNI, subject_id=UNI1, trial_id=6m_1
```

## 2. 분석 흐름

```text
기관폴더-비식별폴더-검사 데이터
→ walking 6m trial만 선별
→ 환자별 모든 walking trial 통합
→ 결측/이상치 처리
→ subject-level mean curve 생성
→ 공변량 보정
→ FDA/fPCA
→ 질환군 내 임상척도 연계
→ fPCA 기반 ML 분석
→ 전체 결과 ZIP 다운로드
```

## 3. 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Streamlit Community Cloud 배포

1. 이 저장소를 GitHub에 업로드합니다.
2. Streamlit Community Cloud에 GitHub 계정으로 로그인합니다.
3. 저장소, branch, entrypoint file을 선택합니다.
4. entrypoint file은 `app.py`로 지정합니다.
5. Deploy를 클릭합니다.

## 5. 주의사항

- 원자료 MOT/TRC/CRF는 개인정보 및 연구데이터일 수 있으므로 GitHub에 올리지 마세요.
- GitHub에는 코드만 올리고, 실제 데이터는 앱 실행 후 업로드하세요.
- Streamlit 업로드 제한은 `.streamlit/config.toml`에서 `maxUploadSize = 2048` MB로 설정했습니다.
- 데이터가 매우 크면 Community Cloud의 메모리/실행시간 한계가 있을 수 있으므로, 병원 내부 서버 또는 Docker 배포가 더 안정적일 수 있습니다.
