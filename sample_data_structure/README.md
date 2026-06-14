# 업로드 예시 구조

기관별 검사 데이터는 ZIP으로 압축해서 업로드합니다.

```text
UNI_gait.zip
├─ UNI1/
│  ├─ 6m_1.mot
│  ├─ 6m_1.trc
│  ├─ 6m_2.mot
│  └─ 6m_2.trc
└─ UNI2/
   ├─ 6m_1.mot
   └─ 6m_1.trc
```

MOT와 TRC는 같은 폴더 안에 있어도 되며, 앱이 확장자로 구분합니다.
