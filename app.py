# -*- coding: utf-8 -*-
"""
OpenCap Gait 분석 웹서비스 - Streamlit App

주요 기능
1) MOT/TRC + CRF 업로드 및 비식별번호 매핑
2) Gait 구간 전처리: 결측 보정, 스플라인/시간 정규화, 이상 궤적 제외, stance/swing 라벨링
3) 공변량 보정 후 FDA/fPCA 분석 및 시각화
4) 질환군 내 HY/UPDRS 임상척도 연결
5) fPCA 기반 누수 방지 5-fold 로지스틱 회귀 ML 분석

실행:
    streamlit run opencap_gait_streamlit_app.py

필수 패키지:
    streamlit pandas numpy scipy scikit-learn statsmodels matplotlib openpyxl
"""

from __future__ import annotations

import io
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import fdrcorrection


# =============================================================================
# 기본 설정
# =============================================================================

st.set_page_config(
    page_title="OpenCap Gait 분석 서비스",
    page_icon="🚶",
    layout="wide",
)

DEFAULT_ID_REGEX = r"^([^_\-. ]+)"
TIME_COL_CANDIDATES = ["time", "Time", "t", "timestamp", "seconds"]
INSTITUTIONS = ["UNI", "UUH", "JBH"]
DEFAULT_WALKING_REGEX = r"^6m(?:[_\- ]*\d+)?(?:\(\d+\))?$"
VALID_TRIAL_STATUS = {"O", "△", "A", "OK", "가능", "사용", "사용가능"}


# =============================================================================
# 공통 유틸리티
# =============================================================================


def _decode_bytes(file_bytes: bytes) -> str:
    """MOT/TRC 텍스트 파일을 최대한 안전하게 디코딩한다."""
    for enc in ("utf-8-sig", "utf-8", "cp949", "latin1"):
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("latin1", errors="ignore")


@st.cache_data(show_spinner=False)
def parse_opensim_table_cached(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    OpenSim 계열 .mot/.trc/storage 형태 파일을 DataFrame으로 읽는다.

    - .mot: 보통 endheader 다음 줄이 컬럼명이다.
    - .trc: 파일마다 header가 다르므로 완벽한 marker 파싱보다는 QC/확인용으로 읽는다.
    - 본 앱의 주 분석 곡선은 MOT의 kinematics 컬럼을 우선 사용한다.
    """
    text = _decode_bytes(file_bytes)
    lines = [ln.rstrip("\n\r") for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        return pd.DataFrame()

    lower_lines = [ln.lower().strip() for ln in lines]

    # MOT/STO 파일: endheader 다음 행이 컬럼명인 경우가 많다.
    if any("endheader" in ln for ln in lower_lines):
        end_idx = next(i for i, ln in enumerate(lower_lines) if "endheader" in ln)
        header_idx = end_idx + 1
        while header_idx < len(lines) and not lines[header_idx].strip():
            header_idx += 1
        if header_idx >= len(lines):
            return pd.DataFrame()
        table_text = "\n".join(lines[header_idx:])
        try:
            df = pd.read_csv(io.StringIO(table_text), sep=r"\s+", engine="python")
        except Exception:
            df = pd.read_csv(io.StringIO(table_text), sep=None, engine="python")
        return _clean_numeric_dataframe(df)

    # TRC 파일: 일반적으로 4~5번째 줄에 Frame#, Time, marker headers가 있다.
    frame_line_idx = None
    for i, ln in enumerate(lines[:15]):
        if "Frame#" in ln or re.search(r"\bFrame\b", ln):
            frame_line_idx = i
            break

    if frame_line_idx is not None:
        # TRC는 marker 이름과 X/Y/Z 줄이 분리될 수 있으므로, numeric data 시작점을 탐색한다.
        data_start = None
        for i in range(frame_line_idx + 1, min(len(lines), frame_line_idx + 8)):
            parts = re.split(r"\s+|\t+", lines[i].strip())
            if parts and re.match(r"^\d+(\.0)?$", parts[0]):
                data_start = i
                break
        if data_start is None:
            data_start = min(frame_line_idx + 2, len(lines) - 1)

        header_parts = re.split(r"\s+|\t+", lines[frame_line_idx].strip())
        # 데이터 컬럼 수에 맞게 임시 컬럼명 생성
        first_data_parts = re.split(r"\s+|\t+", lines[data_start].strip())
        n_cols = len(first_data_parts)
        if len(header_parts) < n_cols:
            header_parts = header_parts + [f"col_{i}" for i in range(len(header_parts), n_cols)]
        header_parts = header_parts[:n_cols]
        data_text = "\n".join(lines[data_start:])
        try:
            df = pd.read_csv(
                io.StringIO(data_text),
                sep=r"\s+",
                engine="python",
                names=header_parts,
            )
        except Exception:
            df = pd.DataFrame()
        return _clean_numeric_dataframe(df)

    # 마지막 fallback: 구분자 자동 추정
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")
        except Exception:
            df = pd.DataFrame()
    return _clean_numeric_dataframe(df)



def _clean_numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """중복 컬럼명을 정리하고, 숫자로 변환 가능한 컬럼은 numeric으로 변환한다."""
    if df.empty:
        return df
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 중복 컬럼명 방지
    seen = {}
    new_cols = []
    for c in df.columns:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
    df.columns = new_cols

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df


@st.cache_data(show_spinner=False)
def read_excel_sheets_cached(file_bytes: bytes) -> List[str]:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def read_excel_sheet_cached(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


def _clean_col_name(x) -> str:
    """CRF 컬럼명을 화면/분석에서 쓰기 좋게 정리한다."""
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).replace("\n", " ")).strip()


def _make_unique_columns(cols: Sequence[str]) -> List[str]:
    out = []
    seen = {}
    for i, c in enumerate(cols):
        c = _clean_col_name(c) or f"col_{i}"
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out


def safe_filename(name: str, max_len: int = 120) -> str:
    """파일/ZIP 내부 경로에 안전하게 쓸 수 있도록 이름을 정리한다."""
    s = str(name)
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", "_", s).strip("._ ")
    if not s:
        s = "unnamed"
    return s[:max_len]


def choose_crf_sheet(file_bytes: bytes, institution: str) -> str:
    """기관별 CRF 파일 또는 통합 CRF에서 해당 기관 sheet를 우선 선택한다."""
    sheets = read_excel_sheets_cached(file_bytes)
    inst = institution.upper()
    preferred = {
        "UNI": ["UNI분석가능여부현황", "UNIST", "UNI"],
        "UUH": ["UUH분석가능여부현황", "울산대병원", "UUH"],
        "JBH": ["JBH분석가능여부현황", "전북대병원", "JBH"],
    }.get(inst, [])
    for key in preferred:
        for s in sheets:
            if key.lower() in str(s).lower():
                return s
    return sheets[0]


def normalize_subject_id_value(value, institution: str) -> str:
    """CRF/폴더의 피험자번호를 UNI1, UUH1, JBH1 형태로 표준화한다."""
    inst = str(institution).upper().strip()
    if pd.isna(value):
        return ""
    raw = str(value).strip()
    raw = re.sub(r"\.0$", "", raw)
    raw = raw.replace(" ", "")
    m = re.search(rf"({inst}\d+)", raw, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"([A-Za-z]{2,5}\d+)", raw)
    if m:
        return m.group(1).upper()
    m = re.search(r"\d+", raw)
    if m:
        return f"{inst}{int(m.group(0))}"
    return raw.upper()


def normalize_trial_id_from_name(filename: str) -> str:
    """6m_1(1).mot 같은 파일명을 CRF의 6m_1 trial명과 맞춘다."""
    stem = Path(filename).stem.strip()
    stem = re.sub(r"\(\d+\)$", "", stem)  # 재촬영/복사 suffix 제거
    stem = stem.replace("-", "_").replace(" ", "_")
    m = re.match(r"(?i)^(6m)(?:_*(\d+))?", stem)
    if m:
        return f"6m_{m.group(2)}" if m.group(2) else "6m"
    return stem


def normalize_subject_id_from_path(parts: Sequence[str], institution: str) -> str:
    """ZIP 내부 경로에서 비식별 폴더명을 찾는다."""
    inst = institution.upper()
    # 1순위: 경로 전체에서 UNI1, UUH12, JBH3 형태 탐색
    for part in parts:
        m = re.search(rf"({inst}\d+)", str(part), flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    # 2순위: 파일 바로 위 폴더를 subject로 간주. 예: 1 -> UNI1, UNI1_재촬영 -> UNI1
    if len(parts) >= 2:
        return normalize_subject_id_value(parts[-2], inst)
    return ""


def is_walking_trial(filename: str, walking_regex: str) -> bool:
    trial_id = normalize_trial_id_from_name(filename)
    try:
        return bool(re.search(walking_regex, trial_id, flags=re.IGNORECASE))
    except re.error:
        return trial_id.lower().startswith("6m")


def read_institution_crf(file_obj, institution: str) -> pd.DataFrame:
    """기관별 CRF를 읽어 subject_id와 피험자군 중심으로 표준화한다."""
    if file_obj is None:
        return pd.DataFrame()
    b = file_obj.getvalue()
    inst = institution.upper()
    if file_obj.name.lower().endswith(".csv"):
        raw = pd.read_csv(io.BytesIO(b), header=None)
    else:
        sheet = choose_crf_sheet(b, inst)
        raw = pd.read_excel(io.BytesIO(b), sheet_name=sheet, header=None)

    # 피험자군과 피험자번호가 있는 header row 탐색
    header_idx = None
    for i in range(min(20, len(raw))):
        vals = [_clean_col_name(v) for v in raw.iloc[i].tolist()]
        joined = " ".join(vals)
        has_group = "피험자군" in joined or re.search(r"\bgroup\b", joined, flags=re.IGNORECASE)
        has_subject = ("피험자번호" in joined) or ("Session Name" in joined) or ("Sub Num" in joined) or re.search(r"\bsubject\b|\bID\b", joined, flags=re.IGNORECASE)
        if has_group and has_subject:
            header_idx = i
            break
    if header_idx is None:
        # fallback: 첫 non-empty row
        header_idx = 0

    cols = _make_unique_columns(raw.iloc[header_idx].tolist())
    df = raw.iloc[header_idx + 1:].copy()
    df.columns = cols
    df = df.dropna(how="all").reset_index(drop=True)
    df["institution"] = inst

    # subject 후보
    subj_candidates = [c for c in df.columns if ("피험자번호" in c) or ("Session Name" in c) or ("Sub Num" in c) or re.search(r"subject|\bID\b", c, flags=re.IGNORECASE)]
    subj_col = subj_candidates[0] if subj_candidates else df.columns[0]
    df["subject_id"] = df[subj_col].apply(lambda x: normalize_subject_id_value(x, inst))
    df = df[df["subject_id"].astype(str).str.len() > 0].copy()

    # group 후보. 반드시 피험자군을 우선한다.
    group_candidates = [c for c in df.columns if "피험자군" in c]
    if not group_candidates:
        group_candidates = [c for c in df.columns if re.search(r"group|diagnosis|군|그룹", c, flags=re.IGNORECASE)]
    if group_candidates:
        gcol = group_candidates[0]
        if gcol != "피험자군":
            df["피험자군"] = df[gcol]
    else:
        df["피험자군"] = np.nan

    # 자주 쓰는 공변량/임상척도 alias 추가
    alias_map = {
        "성별": "sex_auto",
        "만 나이": "age_auto",
        "키": "height_auto",
        "체중": "weight_auto",
        "주손": "dominant_hand_auto",
        "주발": "dominant_foot_auto",
        "호엔": "hy_stage_auto",
        "UPDRS": "updrs_auto",
    }
    for pattern, alias in alias_map.items():
        cand = [c for c in df.columns if pattern in c]
        if cand and alias not in df.columns:
            df[alias] = df[cand[0]]

    # trial status 컬럼은 CRF 원 컬럼명 6m_1, 6m_2, 6m_3 그대로 쓰되, 공백 제거 보완
    for c in list(df.columns):
        clean = _clean_col_name(c)
        if re.fullmatch(r"6m[_ ]*\d+", clean, flags=re.IGNORECASE) and clean != c:
            df[clean.replace(" ", "_")] = df[c]

    return df.reset_index(drop=True)


def extract_subject_id(filename: str, regex_pattern: str = DEFAULT_ID_REGEX) -> str:
    stem = Path(filename).stem
    try:
        m = re.search(regex_pattern, stem)
        if m:
            return str(m.group(1) if m.groups() else m.group(0))
    except re.error:
        pass
    return re.split(r"[_\-. ]+", stem)[0]



def guess_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        first = numeric_cols[0]
        # 첫 numeric 컬럼이 단조 증가하면 time으로 간주
        vals = pd.to_numeric(df[first], errors="coerce").dropna().values
        if len(vals) > 3 and np.all(np.diff(vals) >= 0):
            return first
    return None



def numeric_feature_columns(df: pd.DataFrame, time_col: Optional[str]) -> List[str]:
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    drop_like = {"frame", "frame#", "time", "timestamp", "seconds"}
    out = []
    for c in cols:
        if c == time_col:
            continue
        if str(c).lower() in drop_like:
            continue
        out.append(c)
    return out



def kalman_impute_1d(y: Sequence[float], process_var: float = 1e-4, obs_var: float = 1e-2) -> np.ndarray:
    """
    단순 local-level Kalman filter 기반 결측 보정.
    외부 filterpy 없이 동작하도록 최소 구현했다.
    """
    arr = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    n = len(arr)
    if n == 0:
        return arr
    valid = np.isfinite(arr)
    if valid.sum() == 0:
        return np.zeros(n, dtype=float)

    x = arr[valid][0]
    p = 1.0
    out = np.empty(n, dtype=float)
    for i in range(n):
        # prediction
        p = p + process_var
        if np.isfinite(arr[i]):
            k = p / (p + obs_var)
            x = x + k * (arr[i] - x)
            p = (1 - k) * p
        out[i] = x

    # backward pass로 약간 부드럽게
    back = np.empty(n, dtype=float)
    x = arr[valid][-1]
    p = 1.0
    for i in range(n - 1, -1, -1):
        p = p + process_var
        if np.isfinite(arr[i]):
            k = p / (p + obs_var)
            x = x + k * (arr[i] - x)
            p = (1 - k) * p
        back[i] = x
    return (out + back) / 2.0



def resample_curve(
    x: np.ndarray,
    y: np.ndarray,
    n_grid: int = 101,
    use_kalman: bool = True,
    spline_smoothing: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """한 trial 곡선을 0~100% grid로 정규화한다."""
    x = np.asarray(x, dtype=float)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)

    ok_x = np.isfinite(x)
    if ok_x.sum() < 3:
        x = np.arange(len(y), dtype=float)
        ok_x = np.isfinite(x)

    # x 중복 제거와 정렬
    tmp = pd.DataFrame({"x": x, "y": y}).loc[ok_x].copy()
    tmp = tmp.sort_values("x")
    tmp = tmp.groupby("x", as_index=False)["y"].mean()
    x = tmp["x"].to_numpy(dtype=float)
    y = tmp["y"].to_numpy(dtype=float)

    if len(y) < 5 or np.isfinite(y).sum() < 5:
        raise ValueError("유효한 관측치가 너무 적습니다.")

    if use_kalman:
        y_filled = kalman_impute_1d(y)
    else:
        y_filled = pd.Series(y).interpolate(limit_direction="both").to_numpy(dtype=float)

    # 0~1 정규화
    denom = x.max() - x.min()
    if denom <= 0:
        x_norm = np.linspace(0, 1, len(x))
    else:
        x_norm = (x - x.min()) / denom

    grid = np.linspace(0, 1, n_grid)
    try:
        s_val = float(spline_smoothing) * len(x_norm)
        spline = UnivariateSpline(x_norm, y_filled, s=s_val)
        y_grid = spline(grid)
    except Exception:
        y_grid = np.interp(grid, x_norm, y_filled)
    return grid * 100.0, y_grid



def make_download_button_csv(df: pd.DataFrame, label: str, filename: str) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )



def dataframe_to_zip_bytes(files: Dict[str, pd.DataFrame]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in files.items():
            zf.writestr(name, df.to_csv(index=False, encoding="utf-8-sig"))
    mem.seek(0)
    return mem.read()


# =============================================================================
# 전처리: trial curve -> subject mean curve
# =============================================================================


@dataclass
class ParsedUpload:
    name: str
    subject_id: str
    kind: str
    df: pd.DataFrame
    time_col: Optional[str]
    n_rows: int
    n_cols: int
    institution: str = ""
    trial_id: str = ""
    relative_path: str = ""


def build_upload_index_from_institution_zips(
    institution_zip_files: Dict[str, object],
    walking_regex: str = DEFAULT_WALKING_REGEX,
) -> Tuple[pd.DataFrame, Dict[str, ParsedUpload], pd.DataFrame, pd.DataFrame]:
    """
    기관별 ZIP을 읽고 walking 6m MOT/TRC만 분석 대상으로 index화한다.

    ZIP 내부 권장 구조:
        UNI1/6m_1.mot
        UNI1/6m_1.trc
        UNI2/6m_2.mot
        ...
    또는:
        UNI/UNI1/6m_1.mot

    MOT/TRC는 같은 폴더에 있어도 확장자로 구분한다.
    """
    parsed: Dict[str, ParsedUpload] = {}
    rows = []
    skipped = []

    for inst, zfile in institution_zip_files.items():
        if zfile is None:
            continue
        inst = inst.upper()
        try:
            zf = zipfile.ZipFile(io.BytesIO(zfile.getvalue()))
        except Exception as e:
            skipped.append({"institution": inst, "relative_path": getattr(zfile, "name", ""), "reason": f"zip_read_error: {e}"})
            continue

        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = info.filename.replace("\\", "/")
            fname = Path(rel).name
            suffix = Path(fname).suffix.lower()
            if suffix not in {".mot", ".sto", ".trc", ".txt", ".csv"}:
                skipped.append({"institution": inst, "relative_path": rel, "file_name": fname, "reason": "unsupported_extension"})
                continue

            # .txt/.csv는 명확하지 않으면 제외한다. 실제 OpenCap 분석은 .mot/.trc 기준.
            if suffix not in {".mot", ".sto", ".trc"}:
                skipped.append({"institution": inst, "relative_path": rel, "file_name": fname, "reason": "not_mot_trc"})
                continue

            if not is_walking_trial(fname, walking_regex):
                skipped.append({"institution": inst, "relative_path": rel, "file_name": fname, "reason": "not_walking_6m"})
                continue

            parts = [p for p in rel.split("/") if p]
            subject_id = normalize_subject_id_from_path(parts, inst)
            trial_id = normalize_trial_id_from_name(fname)
            kind = "MOT" if suffix in {".mot", ".sto"} else "TRC"
            try:
                file_bytes = zf.read(info)
                df = parse_opensim_table_cached(file_bytes, fname)
                time_col = guess_time_col(df) if not df.empty else None
                key = f"{inst}:{subject_id}:{trial_id}:{kind}:{rel}"
                parsed[key] = ParsedUpload(
                    name=fname,
                    subject_id=subject_id,
                    kind=kind,
                    df=df,
                    time_col=time_col,
                    n_rows=df.shape[0],
                    n_cols=df.shape[1],
                    institution=inst,
                    trial_id=trial_id,
                    relative_path=rel,
                )
                n_features = len(numeric_feature_columns(df, time_col)) if not df.empty else 0
                rows.append(
                    {
                        "institution": inst,
                        "subject_id": subject_id,
                        "trial_id": trial_id,
                        "file_name": fname,
                        "relative_path": rel,
                        "file_type": kind,
                        "n_rows": df.shape[0],
                        "n_cols": df.shape[1],
                        "time_col_guess": time_col,
                        "numeric_feature_count": n_features,
                    }
                )
            except Exception as e:
                skipped.append({"institution": inst, "relative_path": rel, "file_name": fname, "reason": f"parse_error: {e}"})

    file_index = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped)
    pair_df = build_trial_pair_diagnostics(file_index)
    return file_index, parsed, skipped_df, pair_df


def build_trial_pair_diagnostics(file_index: pd.DataFrame) -> pd.DataFrame:
    """subject-trial 단위로 MOT/TRC pair 상태를 진단한다."""
    if file_index.empty:
        return pd.DataFrame()
    tmp = file_index.copy()
    tmp["has_mot"] = tmp["file_type"].eq("MOT")
    tmp["has_trc"] = tmp["file_type"].eq("TRC")
    out = (
        tmp.groupby(["institution", "subject_id", "trial_id"], as_index=False)
        .agg(
            n_mot=("has_mot", "sum"),
            n_trc=("has_trc", "sum"),
            files=("file_name", lambda x: "; ".join(map(str, x))),
        )
    )
    out["pair_status"] = np.select(
        [out["n_mot"].gt(0) & out["n_trc"].gt(0), out["n_mot"].gt(0), out["n_trc"].gt(0)],
        ["paired", "mot_only", "trc_only"],
        default="missing",
    )
    return out


def build_crf_mapping_diagnostics(pair_df: pd.DataFrame, crf: pd.DataFrame) -> pd.DataFrame:
    """업로드 walking trial과 CRF subject/trial 상태값 매핑을 확인한다."""
    if pair_df.empty or crf.empty or "subject_id" not in crf.columns:
        return pd.DataFrame()
    meta = crf.copy()
    meta["subject_id"] = meta["subject_id"].astype(str)
    trial_cols = {re.sub(r"\s+", "_", str(c)).lower(): c for c in meta.columns if re.fullmatch(r"6m[_ ]*\d+", str(c), flags=re.IGNORECASE)}
    rows = []
    for _, r in pair_df.iterrows():
        sid = str(r["subject_id"])
        trial = str(r["trial_id"]).lower()
        m = meta[meta["subject_id"].astype(str) == sid]
        crf_matched = not m.empty
        status_col_present = trial in trial_cols
        status = np.nan
        if crf_matched and status_col_present:
            status = m.iloc[0][trial_cols[trial]]
        status_str = "" if pd.isna(status) else str(status).strip()
        trial_usable_by_crf = (
            (not status_col_present)
            or status_str in VALID_TRIAL_STATUS
            or status_str.upper() in VALID_TRIAL_STATUS
        )
        if not crf_matched:
            mapping_status = "crf_subject_unmatched"
        elif r["pair_status"] not in {"paired", "mot_only"}:
            mapping_status = "mot_missing"
        elif status_col_present and not status_str:
            mapping_status = "crf_trial_status_blank"
        elif status_col_present and not trial_usable_by_crf:
            mapping_status = "crf_trial_not_usable"
        else:
            mapping_status = "ok"
        rows.append(
            {
                "institution": r.get("institution", ""),
                "subject_id": sid,
                "trial_id": r["trial_id"],
                "pair_status": r["pair_status"],
                "crf_subject_matched": crf_matched,
                "crf_trial_status_col_present": bool(status_col_present),
                "crf_trial_status": status_str,
                "trial_usable_by_crf": bool(trial_usable_by_crf),
                "mapping_status": mapping_status,
            }
        )
    return pd.DataFrame(rows)



def preprocess_to_subject_curves(
    parsed: Dict[str, ParsedUpload],
    selected_features: Sequence[str],
    n_grid: int,
    use_kalman: bool,
    spline_smoothing: float,
    outlier_percentile: float,
    stance_pct: float,
    min_points: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    MOT trial들을 subject-feature별 평균 curve로 만든다.

    반환:
    - long_df: subject_id, feature, grid_pct, value, phase, n_trials_kept 등
    - qc_df: subject-feature-file별 QC 기록
    - excluded_df: 이상 trajectory 제외 기록
    """
    trial_rows = []
    qc_rows = []

    for item in parsed.values():
        if item.kind != "MOT" or item.df.empty:
            continue
        df = item.df
        time_col = item.time_col
        x = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float) if time_col else np.arange(len(df))

        for feat in selected_features:
            if feat not in df.columns:
                continue
            y = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float)
            n_valid = int(np.isfinite(y).sum())
            missing_rate = float(1 - n_valid / max(len(y), 1))
            if len(y) < min_points or n_valid < min_points:
                qc_rows.append(
                    {
                        "institution": item.institution,
                        "subject_id": item.subject_id,
                        "trial_id": item.trial_id,
                        "file_name": item.name,
                        "feature": feat,
                        "status": "skip_too_few_points",
                        "n_points": len(y),
                        "n_valid": n_valid,
                        "missing_rate": missing_rate,
                    }
                )
                continue
            try:
                grid_pct, y_grid = resample_curve(
                    x,
                    y,
                    n_grid=n_grid,
                    use_kalman=use_kalman,
                    spline_smoothing=spline_smoothing,
                )
                trial_rows.append(
                    {
                        "institution": item.institution,
                        "subject_id": item.subject_id,
                        "trial_id": item.trial_id,
                        "file_name": item.name,
                        "feature": feat,
                        "grid_pct": grid_pct,
                        "curve": y_grid,
                        "missing_rate_raw": missing_rate,
                    }
                )
                qc_rows.append(
                    {
                        "institution": item.institution,
                        "subject_id": item.subject_id,
                        "trial_id": item.trial_id,
                        "file_name": item.name,
                        "feature": feat,
                        "status": "ok",
                        "n_points": len(y),
                        "n_valid": n_valid,
                        "missing_rate": missing_rate,
                    }
                )
            except Exception as e:
                qc_rows.append(
                    {
                        "institution": item.institution,
                        "subject_id": item.subject_id,
                        "trial_id": item.trial_id,
                        "file_name": item.name,
                        "feature": feat,
                        "status": f"skip_error: {e}",
                        "n_points": len(y),
                        "n_valid": n_valid,
                        "missing_rate": missing_rate,
                    }
                )

    if not trial_rows:
        return pd.DataFrame(), pd.DataFrame(qc_rows), pd.DataFrame()

    # subject-feature별 이상 trajectory 제외 후 평균화
    long_rows = []
    excluded_rows = []
    trial_df = pd.DataFrame(trial_rows)

    for (sid, feat), sub in trial_df.groupby(["subject_id", "feature"]):
        curves = np.vstack(sub["curve"].to_numpy())
        file_names = sub["file_name"].to_list()
        trial_ids = sub["trial_id"].to_list() if "trial_id" in sub.columns else [""] * len(file_names)
        institutions = sub["institution"].to_list() if "institution" in sub.columns else [""] * len(file_names)
        missing_rates = sub["missing_rate_raw"].to_list()

        if curves.shape[0] >= 3:
            med = np.nanmedian(curves, axis=0)
            dist = np.sqrt(np.nanmean((curves - med) ** 2, axis=1))
            cutoff = np.nanpercentile(dist, outlier_percentile)
            keep = dist <= cutoff
        else:
            dist = np.zeros(curves.shape[0])
            cutoff = np.nan
            keep = np.ones(curves.shape[0], dtype=bool)

        for inst, tid, fn, d, k, miss in zip(institutions, trial_ids, file_names, dist, keep, missing_rates):
            excluded_rows.append(
                {
                    "institution": inst,
                    "subject_id": sid,
                    "trial_id": tid,
                    "feature": feat,
                    "file_name": fn,
                    "trajectory_distance": float(d),
                    "outlier_cutoff": float(cutoff) if np.isfinite(cutoff) else np.nan,
                    "excluded": bool(not k),
                    "missing_rate_raw": float(miss),
                }
            )

        kept_curves = curves[keep]
        if kept_curves.shape[0] == 0:
            kept_curves = curves
        mean_curve = np.nanmean(kept_curves, axis=0)
        grid_pct = trial_df.iloc[0]["grid_pct"]
        for gp, val in zip(grid_pct, mean_curve):
            long_rows.append(
                {
                    "subject_id": sid,
                    "feature": feat,
                    "grid_pct": float(gp),
                    "value": float(val),
                    "phase": "Stance" if gp <= stance_pct else "Swing",
                    "n_trials_total": int(curves.shape[0]),
                    "n_trials_kept": int(kept_curves.shape[0]),
                }
            )

    return pd.DataFrame(long_rows), pd.DataFrame(qc_rows), pd.DataFrame(excluded_rows)


# =============================================================================
# 공변량 보정 / FDA / fPCA
# =============================================================================



def make_design_matrix(
    meta: pd.DataFrame,
    covariates: Sequence[str],
    fit_columns: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    공변량 design matrix 생성.
    숫자는 median impute, 범주는 Missing 포함 one-hot 처리.
    fit_columns가 있으면 train에서 만든 컬럼 구조에 맞춰 test를 정렬한다.
    """
    if not covariates:
        X = pd.DataFrame(index=meta.index)
        if fit_columns is not None:
            for c in fit_columns:
                X[c] = 0.0
            return X[list(fit_columns)], list(fit_columns)
        return X, []

    X_parts = []
    for c in covariates:
        if c not in meta.columns:
            continue
        s = meta[c]
        numeric = pd.to_numeric(s, errors="coerce")
        # numeric으로 충분히 변환되면 연속형으로 처리
        if numeric.notna().mean() >= 0.8:
            med = numeric.median()
            if not np.isfinite(med):
                med = 0.0
            X_parts.append(pd.DataFrame({c: numeric.fillna(med).astype(float)}, index=meta.index))
        else:
            cat = s.astype("object").where(s.notna(), "Missing").astype(str)
            d = pd.get_dummies(cat, prefix=c, drop_first=True, dtype=float)
            d.index = meta.index
            X_parts.append(d)

    if X_parts:
        X = pd.concat(X_parts, axis=1)
    else:
        X = pd.DataFrame(index=meta.index)

    if fit_columns is not None:
        for c in fit_columns:
            if c not in X.columns:
                X[c] = 0.0
        X = X[list(fit_columns)]
        return X, list(fit_columns)

    return X, X.columns.tolist()



def long_to_feature_matrices(long_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """long curve table을 feature별 subject x grid matrix로 변환한다."""
    matrices: Dict[str, pd.DataFrame] = {}
    if long_df.empty:
        return matrices
    for feat, sub in long_df.groupby("feature"):
        mat = sub.pivot_table(index="subject_id", columns="grid_pct", values="value", aggfunc="mean")
        mat = mat.sort_index(axis=1)
        matrices[str(feat)] = mat
    return matrices



def adjust_matrix_all_subjects(
    mat: pd.DataFrame,
    meta: pd.DataFrame,
    subject_col: str,
    covariates: Sequence[str],
    add_back_grid_mean: bool = True,
) -> pd.DataFrame:
    """전체 subject 기준 공변량 보정 curve matrix를 만든다. 탐색/시각화용."""
    common_subjects = [sid for sid in mat.index if sid in set(meta[subject_col].astype(str))]
    if not common_subjects:
        return pd.DataFrame()
    mat2 = mat.loc[common_subjects].copy()
    meta2 = meta.set_index(meta[subject_col].astype(str)).loc[common_subjects]
    X, _ = make_design_matrix(meta2, covariates)

    adjusted = pd.DataFrame(index=mat2.index, columns=mat2.columns, dtype=float)
    for grid in mat2.columns:
        y = pd.to_numeric(mat2[grid], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y)
        if ok.sum() < max(3, X.shape[1] + 2):
            adjusted[grid] = y
            continue
        if X.shape[1] == 0:
            pred = np.nanmean(y[ok])
            resid = y - pred
        else:
            model = LinearRegression()
            model.fit(X.iloc[ok].to_numpy(), y[ok])
            pred = model.predict(X.to_numpy())
            resid = y - pred
        if add_back_grid_mean:
            adjusted[grid] = resid + np.nanmean(y[ok])
        else:
            adjusted[grid] = resid
    return adjusted



def adjust_all_feature_matrices(
    matrices: Dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    subject_col: str,
    covariates: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    out = {}
    for feat, mat in matrices.items():
        adj = adjust_matrix_all_subjects(mat, meta, subject_col, covariates)
        if not adj.empty:
            out[feat] = adj
    return out



def matrix_to_long(matrices: Dict[str, pd.DataFrame], value_col: str = "adjusted_value") -> pd.DataFrame:
    rows = []
    for feat, mat in matrices.items():
        tmp = mat.copy()
        tmp.index.name = "subject_id"
        long = tmp.reset_index().melt(id_vars="subject_id", var_name="grid_pct", value_name=value_col)
        long["feature"] = feat
        rows.append(long)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["grid_pct"] = pd.to_numeric(out["grid_pct"], errors="coerce")
    return out[["subject_id", "feature", "grid_pct", value_col]]



def fit_fpca_for_feature(mat: pd.DataFrame, n_components: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """feature 하나에 대해 PCA를 fPCA처럼 적용한다."""
    mat_num = mat.apply(pd.to_numeric, errors="coerce")
    # subject별 결측이 너무 많으면 제외. 나머지는 grid 평균으로 impute.
    valid_subject = mat_num.notna().mean(axis=1) >= 0.8
    mat_num = mat_num.loc[valid_subject]
    if mat_num.shape[0] < 3:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    imp = SimpleImputer(strategy="mean")
    X = imp.fit_transform(mat_num)
    n_comp = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_comp, random_state=42)
    scores = pca.fit_transform(X)

    score_df = pd.DataFrame({"subject_id": mat_num.index.astype(str)})
    for j in range(n_comp):
        score_df[f"FPC{j+1}"] = scores[:, j]

    load_rows = []
    grid = mat_num.columns.astype(float).to_numpy()
    for j in range(n_comp):
        for gp, loading in zip(grid, pca.components_[j]):
            load_rows.append(
                {
                    "component": f"FPC{j+1}",
                    "grid_pct": float(gp),
                    "loading": float(loading),
                }
            )
    loading_df = pd.DataFrame(load_rows)
    evr_df = pd.DataFrame(
        {
            "component": [f"FPC{j+1}" for j in range(n_comp)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    return score_df, loading_df, evr_df



def run_fpca_all_features(
    matrices: Dict[str, pd.DataFrame],
    n_components: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows = []
    loading_rows = []
    evr_rows = []
    wide_parts = []

    for feat, mat in matrices.items():
        score_df, loading_df, evr_df = fit_fpca_for_feature(mat, n_components=n_components)
        if score_df.empty:
            continue
        score_df["feature"] = feat
        score_rows.append(score_df)
        loading_df["feature"] = feat
        loading_rows.append(loading_df)
        evr_df["feature"] = feat
        evr_rows.append(evr_df)

        wide = score_df[["subject_id"]].copy()
        for c in [c for c in score_df.columns if c.startswith("FPC")]:
            wide[f"{feat}__{c}"] = score_df[c]
        wide_parts.append(wide)

    if score_rows:
        scores_long = pd.concat(score_rows, ignore_index=True)
        loadings_long = pd.concat(loading_rows, ignore_index=True)
        evr = pd.concat(evr_rows, ignore_index=True)
        scores_wide = wide_parts[0]
        for part in wide_parts[1:]:
            scores_wide = scores_wide.merge(part, on="subject_id", how="outer")
    else:
        scores_long = pd.DataFrame()
        loadings_long = pd.DataFrame()
        evr = pd.DataFrame()
        scores_wide = pd.DataFrame()
    return scores_long, loadings_long, evr, scores_wide


# =============================================================================
# 통계 검정: 2D fPCA 분포 차이
# =============================================================================



def hotelling_t2_2group(X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    y = np.asarray(y)
    X0 = X[y == 0]
    X1 = X[y == 1]
    n0, n1 = X0.shape[0], X1.shape[0]
    p = X.shape[1]
    if n0 <= p or n1 <= p:
        return np.nan, np.nan, np.nan
    m0 = X0.mean(axis=0)
    m1 = X1.mean(axis=0)
    S0 = np.cov(X0, rowvar=False)
    S1 = np.cov(X1, rowvar=False)
    Sp = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)
    diff = m1 - m0
    T2 = (n0 * n1 / (n0 + n1)) * float(diff.T @ np.linalg.pinv(Sp) @ diff)
    F_stat = ((n0 + n1 - p - 1) / ((n0 + n1 - 2) * p)) * T2
    p_val = 1 - stats.f.cdf(F_stat, p, n0 + n1 - p - 1)
    return float(T2), float(F_stat), float(p_val)



def permanova_2group_euclidean(
    X: np.ndarray, y: np.ndarray, n_perm: int = 1000, random_state: int = 42
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    n = X.shape[0]
    grand = X.mean(axis=0)
    groups = np.unique(y)
    if len(groups) != 2:
        return np.nan, np.nan, np.nan

    def stat_for(labels: np.ndarray) -> Tuple[float, float]:
        ss_total = float(((X - grand) ** 2).sum())
        ss_between = 0.0
        for g in groups:
            Xg = X[labels == g]
            if Xg.shape[0] == 0:
                continue
            mg = Xg.mean(axis=0)
            ss_between += Xg.shape[0] * float(((mg - grand) ** 2).sum())
        ss_within = ss_total - ss_between
        df_between = len(groups) - 1
        df_within = n - len(groups)
        F_stat = (ss_between / df_between) / (ss_within / df_within) if ss_within > 0 else np.inf
        R2 = ss_between / ss_total if ss_total > 0 else np.nan
        return float(F_stat), float(R2)

    obs_F, obs_R2 = stat_for(y)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        Fp, _ = stat_for(yp)
        if Fp >= obs_F:
            cnt += 1
    p_val = (cnt + 1) / (n_perm + 1)
    return float(obs_F), float(obs_R2), float(p_val)



def energy_distance_test_2group(
    X: np.ndarray, y: np.ndarray, n_perm: int = 1000, random_state: int = 42
) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    X0, X1 = X[y == 0], X[y == 1]
    if X0.shape[0] < 2 or X1.shape[0] < 2:
        return np.nan, np.nan

    def energy_stat(a: np.ndarray, b: np.ndarray) -> float:
        d_ab = cdist(a, b).mean()
        d_aa = cdist(a, a).mean()
        d_bb = cdist(b, b).mean()
        return float(2 * d_ab - d_aa - d_bb)

    obs = energy_stat(X0, X1)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        stat_p = energy_stat(X[yp == 0], X[yp == 1])
        if stat_p >= obs:
            cnt += 1
    return float(obs), float((cnt + 1) / (n_perm + 1))



def fpca_2d_tests(
    scores_long: pd.DataFrame,
    meta: pd.DataFrame,
    subject_col: str,
    group_col: str,
    control_label: str,
    disease_label: str,
    n_perm: int,
) -> pd.DataFrame:
    if scores_long.empty:
        return pd.DataFrame()
    meta2 = meta[[subject_col, group_col]].copy()
    meta2[subject_col] = meta2[subject_col].astype(str)
    rows = []
    for feat, sub in scores_long.groupby("feature"):
        if not {"FPC1", "FPC2"}.issubset(sub.columns):
            continue
        tmp = sub.merge(meta2, left_on="subject_id", right_on=subject_col, how="left")
        tmp = tmp[tmp[group_col].isin([control_label, disease_label])].copy()
        tmp = tmp.dropna(subset=["FPC1", "FPC2", group_col])
        if tmp[group_col].nunique() != 2 or tmp.shape[0] < 6:
            continue
        y = (tmp[group_col].astype(str) == str(disease_label)).astype(int).to_numpy()
        X = tmp[["FPC1", "FPC2"]].to_numpy(dtype=float)
        t2, fstat, p_hot = hotelling_t2_2group(X, y)
        f_perm, r2_perm, p_perm = permanova_2group_euclidean(X, y, n_perm=n_perm)
        e_stat, p_energy = energy_distance_test_2group(X, y, n_perm=n_perm)
        rows.append(
            {
                "feature": feat,
                "n_control": int((y == 0).sum()),
                "n_disease": int((y == 1).sum()),
                "control_mean_FPC1": float(X[y == 0, 0].mean()),
                "control_mean_FPC2": float(X[y == 0, 1].mean()),
                "disease_mean_FPC1": float(X[y == 1, 0].mean()),
                "disease_mean_FPC2": float(X[y == 1, 1].mean()),
                "hotelling_T2": t2,
                "hotelling_F": fstat,
                "hotelling_p": p_hot,
                "permanova_F": f_perm,
                "permanova_R2": r2_perm,
                "permanova_p": p_perm,
                "energy_stat": e_stat,
                "energy_p": p_energy,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        for p_col in ["hotelling_p", "permanova_p", "energy_p"]:
            ok = out[p_col].notna()
            q = np.full(out.shape[0], np.nan)
            if ok.sum() > 0:
                _, q_vals = fdrcorrection(out.loc[ok, p_col].to_numpy())
                q[np.where(ok)[0]] = q_vals
            out[p_col.replace("_p", "_q_fdr")] = q
        out = out.sort_values(["permanova_q_fdr", "permanova_R2"], ascending=[True, False])
    return out


# =============================================================================
# 그림 함수
# =============================================================================



def bootstrap_mean_ci(arr: np.ndarray, n_boot: int = 500, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr).mean(axis=1) > 0.8]
    if arr.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    mean = np.nanmean(arr, axis=0)
    if arr.shape[0] < 3:
        return mean, mean, mean
    boots = []
    n = arr.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(np.nanmean(arr[idx], axis=0))
    boots = np.vstack(boots)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return mean, lo, hi



def plot_fda_group_mean(
    mat: pd.DataFrame,
    meta: pd.DataFrame,
    subject_col: str,
    group_col: str,
    groups_to_plot: Sequence[str],
    title: str,
    show_significance: bool = True,
    alpha: float = 0.05,
    stance_pct: float = 60.0,
) -> plt.Figure:
    meta2 = meta.copy()
    meta2[subject_col] = meta2[subject_col].astype(str)
    meta2 = meta2.set_index(subject_col)
    common = [sid for sid in mat.index.astype(str) if sid in meta2.index]
    mat = mat.loc[common]
    grid = mat.columns.astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for g in groups_to_plot:
        sids = [sid for sid in common if str(meta2.loc[sid, group_col]) == str(g)]
        if not sids:
            continue
        arr = mat.loc[sids].to_numpy(dtype=float)
        mean, lo, hi = bootstrap_mean_ci(arr)
        if len(mean) == 0:
            continue
        ax.plot(grid, mean, label=f"{g} (n={len(sids)})")
        ax.fill_between(grid, lo, hi, alpha=0.16)

    if show_significance and len(groups_to_plot) == 2:
        g0, g1 = groups_to_plot
        s0 = [sid for sid in common if str(meta2.loc[sid, group_col]) == str(g0)]
        s1 = [sid for sid in common if str(meta2.loc[sid, group_col]) == str(g1)]
        if len(s0) >= 3 and len(s1) >= 3:
            pvals = []
            for col in mat.columns:
                a = mat.loc[s0, col].to_numpy(dtype=float)
                b = mat.loc[s1, col].to_numpy(dtype=float)
                try:
                    p = stats.ttest_ind(a, b, nan_policy="omit", equal_var=False).pvalue
                except Exception:
                    p = np.nan
                pvals.append(p)
            pvals = np.asarray(pvals, dtype=float)
            ok = np.isfinite(pvals)
            sig = np.zeros_like(pvals, dtype=bool)
            if ok.sum() > 0:
                _, q = fdrcorrection(pvals[ok])
                sig[np.where(ok)[0]] = q < alpha
            if sig.any():
                ymin, ymax = ax.get_ylim()
                y_sig = ymin + 0.04 * (ymax - ymin)
                ax.scatter(grid[sig], np.full(sig.sum(), y_sig), s=10, marker="|", label=f"FDR q<{alpha}")

    ax.axvline(stance_pct, linestyle="--", linewidth=1, alpha=0.55)
    ax.text(stance_pct, ax.get_ylim()[1], "  stance/swing", va="top", fontsize=9)
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Adjusted curve value")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig



def add_cov_ellipse(ax, x: np.ndarray, y: np.ndarray, n_std: float = 1.8) -> None:
    pts = np.column_stack([x, y])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 4:
        return
    cov = np.cov(pts, rowvar=False)
    if not np.isfinite(cov).all():
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=pts.mean(axis=0), width=width, height=height, angle=angle, fill=False, linewidth=1.5, alpha=0.65)
    ax.add_patch(ell)



def plot_fpca_scatter(
    scores_long: pd.DataFrame,
    meta: pd.DataFrame,
    subject_col: str,
    group_col: str,
    feature: str,
    groups_to_plot: Sequence[str],
    title: str,
) -> plt.Figure:
    sub = scores_long[scores_long["feature"] == feature].copy()
    meta2 = meta[[subject_col, group_col]].copy()
    meta2[subject_col] = meta2[subject_col].astype(str)
    tmp = sub.merge(meta2, left_on="subject_id", right_on=subject_col, how="left")
    tmp = tmp[tmp[group_col].astype(str).isin([str(g) for g in groups_to_plot])]
    fig, ax = plt.subplots(figsize=(7.3, 5.8))
    for g in groups_to_plot:
        ss = tmp[tmp[group_col].astype(str) == str(g)]
        if ss.empty:
            continue
        ax.scatter(ss["FPC1"], ss["FPC2"], label=f"{g} (n={len(ss)})", alpha=0.82, s=55)
        add_cov_ellipse(ax, ss["FPC1"].to_numpy(), ss["FPC2"].to_numpy())
        ax.scatter(ss["FPC1"].mean(), ss["FPC2"].mean(), marker="X", s=160, edgecolor="black", linewidth=0.8)
    ax.axhline(0, linewidth=0.8, alpha=0.35)
    ax.axvline(0, linewidth=0.8, alpha=0.35)
    ax.set_xlabel("FPC1 score")
    ax.set_ylabel("FPC2 score")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig



def plot_fpca_box(scores_long: pd.DataFrame, meta: pd.DataFrame, subject_col: str, group_col: str, feature: str, pc: str, groups: Sequence[str]) -> plt.Figure:
    sub = scores_long[scores_long["feature"] == feature].copy()
    meta2 = meta[[subject_col, group_col]].copy()
    meta2[subject_col] = meta2[subject_col].astype(str)
    tmp = sub.merge(meta2, left_on="subject_id", right_on=subject_col, how="left")
    data = [tmp.loc[tmp[group_col].astype(str) == str(g), pc].dropna().to_numpy() for g in groups]
    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    ax.boxplot(data, labels=[str(g) for g in groups], showfliers=False)
    for i, vals in enumerate(data, start=1):
        if len(vals):
            jitter = np.random.default_rng(42 + i).normal(i, 0.035, size=len(vals))
            ax.scatter(jitter, vals, alpha=0.7, s=35)
    ax.set_title(f"{feature} - {pc} 그룹별 분포")
    ax.set_ylabel(f"{pc} score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig



def plot_loading(loadings_long: pd.DataFrame, feature: str) -> plt.Figure:
    sub = loadings_long[loadings_long["feature"] == feature]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for comp, ss in sub.groupby("component"):
        ss = ss.sort_values("grid_pct")
        ax.plot(ss["grid_pct"], ss["loading"], label=comp)
    ax.axhline(0, linewidth=0.8, alpha=0.5)
    ax.axvline(60, linestyle="--", linewidth=1, alpha=0.55)
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Loading")
    ax.set_title(f"fPCA component loading: {feature}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# =============================================================================
# 임상척도 연결
# =============================================================================



def fpca_clinical_correlation(
    scores_wide: pd.DataFrame,
    meta: pd.DataFrame,
    subject_col: str,
    disease_mask: pd.Series,
    clinical_vars: Sequence[str],
) -> pd.DataFrame:
    if scores_wide.empty or not clinical_vars:
        return pd.DataFrame()
    meta2 = meta.copy()
    meta2[subject_col] = meta2[subject_col].astype(str)
    meta2 = meta2.loc[disease_mask].copy()
    dat = scores_wide.merge(meta2, left_on="subject_id", right_on=subject_col, how="inner")
    score_cols = [c for c in scores_wide.columns if c != "subject_id"]
    rows = []
    for cv in clinical_vars:
        if cv not in dat.columns:
            continue
        y = pd.to_numeric(dat[cv], errors="coerce")
        for sc in score_cols:
            x = pd.to_numeric(dat[sc], errors="coerce")
            ok = x.notna() & y.notna()
            if ok.sum() < 5:
                continue
            rho, p = stats.spearmanr(x[ok], y[ok])
            rows.append(
                {
                    "clinical_variable": cv,
                    "fpca_feature": sc,
                    "n": int(ok.sum()),
                    "spearman_rho": float(rho),
                    "p_value": float(p),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        _, q = fdrcorrection(out["p_value"].fillna(1).to_numpy())
        out["q_value_fdr"] = q
        out = out.sort_values(["q_value_fdr", "p_value"])
    return out


# =============================================================================
# ML: train/test leakage 방지 fPCA + logistic regression
# =============================================================================



def transform_feature_train_test_no_leakage(
    mat: pd.DataFrame,
    meta: pd.DataFrame,
    subject_col: str,
    covariates: Sequence[str],
    train_subjects: Sequence[str],
    test_subjects: Sequence[str],
    n_components: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    한 feature에 대해 train fold에서만 공변량 보정모형과 PCA를 학습하고 test를 변환한다.
    """
    train_subjects = [str(s) for s in train_subjects if str(s) in mat.index.astype(str)]
    test_subjects = [str(s) for s in test_subjects if str(s) in mat.index.astype(str)]
    if len(train_subjects) < 4 or len(test_subjects) == 0:
        return pd.DataFrame(index=train_subjects), pd.DataFrame(index=test_subjects), []

    mat2 = mat.copy()
    mat2.index = mat2.index.astype(str)
    Y_train = mat2.loc[train_subjects].apply(pd.to_numeric, errors="coerce")
    Y_test = mat2.loc[test_subjects].apply(pd.to_numeric, errors="coerce")

    meta_idx = meta.copy()
    meta_idx[subject_col] = meta_idx[subject_col].astype(str)
    meta_idx = meta_idx.set_index(subject_col)
    train_meta = meta_idx.loc[train_subjects]
    test_meta = meta_idx.loc[test_subjects]

    X_train_cov, cov_cols = make_design_matrix(train_meta, covariates)
    X_test_cov, _ = make_design_matrix(test_meta, covariates, fit_columns=cov_cols)

    train_adj = pd.DataFrame(index=train_subjects, columns=Y_train.columns, dtype=float)
    test_adj = pd.DataFrame(index=test_subjects, columns=Y_train.columns, dtype=float)

    for col in Y_train.columns:
        ytr = Y_train[col].to_numpy(dtype=float)
        ok = np.isfinite(ytr)
        if ok.sum() < max(3, X_train_cov.shape[1] + 2):
            mean_val = np.nanmean(ytr) if np.isfinite(ytr).any() else 0.0
            train_adj[col] = ytr - mean_val
            test_adj[col] = Y_test[col].to_numpy(dtype=float) - mean_val
            continue
        if X_train_cov.shape[1] == 0:
            mean_val = np.nanmean(ytr[ok])
            train_adj[col] = ytr - mean_val
            test_adj[col] = Y_test[col].to_numpy(dtype=float) - mean_val
        else:
            lm = LinearRegression()
            lm.fit(X_train_cov.iloc[ok].to_numpy(), ytr[ok])
            train_adj[col] = ytr - lm.predict(X_train_cov.to_numpy())
            test_adj[col] = Y_test[col].to_numpy(dtype=float) - lm.predict(X_test_cov.to_numpy())

    # train 평균으로 imputation. test도 train 평균만 사용.
    imp = SimpleImputer(strategy="mean")
    Xtr = imp.fit_transform(train_adj)
    Xte = imp.transform(test_adj)

    max_comp = min(n_components, Xtr.shape[0] - 1, Xtr.shape[1])
    if max_comp < 1:
        return pd.DataFrame(index=train_subjects), pd.DataFrame(index=test_subjects), []
    pca = PCA(n_components=max_comp, random_state=42)
    Ztr = pca.fit_transform(Xtr)
    Zte = pca.transform(Xte)
    cols = [f"FPC{i+1}" for i in range(max_comp)]
    return (
        pd.DataFrame(Ztr, index=train_subjects, columns=cols),
        pd.DataFrame(Zte, index=test_subjects, columns=cols),
        cols,
    )



def compute_binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)
    ok = np.isfinite(prob) & np.isfinite(y_true)
    y_true = y_true[ok].astype(int)
    prob = prob[ok]
    if len(y_true) == 0:
        return {"AUC": np.nan, "Accuracy": np.nan, "Sensitivity": np.nan, "Specificity": np.nan, "TP": 0, "FP": 0, "TN": 0, "FN": 0}
    pred = (prob >= threshold).astype(int)
    if len(np.unique(y_true)) == 2:
        auc_val = roc_auc_score(y_true, prob)
    else:
        auc_val = np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "AUC": float(auc_val),
        "Accuracy": float(accuracy_score(y_true, pred)),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }



def run_no_leakage_ml(
    raw_matrices: Dict[str, pd.DataFrame],
    meta: pd.DataFrame,
    subject_col: str,
    group_col: str,
    control_label: str,
    disease_label: str,
    covariates: Sequence[str],
    selected_features: Sequence[str],
    n_components: int,
    n_splits: int,
    c_value: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """fPCA 정보 누수를 막는 fold 내부 PCA + 로지스틱 회귀."""
    meta2 = meta.copy()
    meta2[subject_col] = meta2[subject_col].astype(str)
    meta2 = meta2[meta2[group_col].astype(str).isin([str(control_label), str(disease_label)])].copy()

    # 모든 선택 feature에 curve가 있는 subject 우선. 부족하면 일부 missing은 fold 내 imputer가 처리한다.
    subjects = meta2[subject_col].astype(str).to_list()
    y = (meta2[group_col].astype(str) == str(disease_label)).astype(int).to_numpy()

    # class 최소 수보다 split이 클 수 없도록 조정
    _, counts = np.unique(y, return_counts=True)
    max_splits = int(counts.min()) if len(counts) == 2 else 0
    n_splits = max(2, min(int(n_splits), max_splits))
    if max_splits < 2:
        raise ValueError("두 그룹 중 한쪽 표본 수가 2명 미만이라 5-fold 분석을 할 수 없습니다.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_prob = np.full(len(subjects), np.nan, dtype=float)
    fold_rows = []
    coef_rows = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(subjects, y), start=1):
        train_subjects = [subjects[i] for i in tr_idx]
        test_subjects = [subjects[i] for i in te_idx]
        y_train = y[tr_idx]
        y_test = y[te_idx]

        train_parts = []
        test_parts = []
        feature_names = []
        for feat in selected_features:
            if feat not in raw_matrices:
                continue
            ztr, zte, cols = transform_feature_train_test_no_leakage(
                raw_matrices[feat],
                meta2,
                subject_col,
                covariates,
                train_subjects,
                test_subjects,
                n_components=n_components,
            )
            if not cols:
                continue
            # 전체 train/test subject 순서로 reindex
            ztr = ztr.reindex(train_subjects)
            zte = zte.reindex(test_subjects)
            ztr.columns = [f"{feat}__{c}" for c in ztr.columns]
            zte.columns = [f"{feat}__{c}" for c in zte.columns]
            train_parts.append(ztr)
            test_parts.append(zte)
            feature_names.extend(ztr.columns.tolist())

        if not train_parts:
            raise ValueError("ML에 사용할 fPCA feature를 만들 수 없습니다. 선택 feature와 표본 수를 확인하세요.")

        X_train = pd.concat(train_parts, axis=1)
        X_test = pd.concat(test_parts, axis=1)
        # 컬럼 정렬 및 결측 imputation/scaling/logistic
        X_test = X_test.reindex(columns=X_train.columns)

        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=float(c_value),
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_test)[:, 1]
        oof_prob[te_idx] = prob

        metrics = compute_binary_metrics(y_test, prob)
        metrics.update({"fold": fold, "n_train": len(train_subjects), "n_test": len(test_subjects)})
        fold_rows.append(metrics)

        coefs = pipe.named_steps["clf"].coef_[0]
        for name, coef in zip(X_train.columns, coefs):
            coef_rows.append({"fold": fold, "feature": name, "coef_standardized": float(coef)})

    oof_df = pd.DataFrame({"subject_id": subjects, "y_true": y, "prob_disease": oof_prob})
    oof_metrics = compute_binary_metrics(y, oof_prob)
    oof_metrics_df = pd.DataFrame([{"metric_scope": "OOF", **oof_metrics, "n": len(y)}])

    fold_df = pd.DataFrame(fold_rows)
    coef_df = pd.DataFrame(coef_rows)
    if not coef_df.empty:
        coef_summary = (
            coef_df.groupby("feature")
            .agg(
                coef_mean=("coef_standardized", "mean"),
                coef_sd=("coef_standardized", "std"),
                abs_coef_mean=("coef_standardized", lambda x: float(np.mean(np.abs(x)))),
                selected_folds=("fold", "nunique"),
            )
            .reset_index()
            .sort_values("abs_coef_mean", ascending=False)
        )
    else:
        coef_summary = pd.DataFrame()

    return fold_df, oof_metrics_df, oof_df, coef_summary



def plot_roc_from_oof(oof_df: pd.DataFrame) -> plt.Figure:
    y = oof_df["y_true"].to_numpy(dtype=float)
    p = oof_df["prob_disease"].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y = y[ok].astype(int)
    p = p[ok]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if len(np.unique(y)) == 2 and len(y) > 1:
        fpr, tpr, _ = roc_curve(y, p)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"OOF ROC AUC={auc_val:.3f}")
    else:
        ax.text(0.5, 0.5, "ROC 계산에 필요한 두 그룹의 OOF 예측이 부족합니다.", ha="center", va="center")
    ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.6)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("5-fold OOF ROC")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig




# =============================================================================
# Streamlit 실행 상태 관리
# =============================================================================

PARSED_STATE_KEYS = [
    "file_index", "parsed_uploads", "skipped_files", "pair_df", "crf", "crf_mapping_df",
    "upload_signature", "parse_done",
]

ANALYSIS_STATE_KEYS = [
    "curve_long", "qc_df", "excluded_df", "raw_matrices", "adjusted_matrices", "adjusted_long",
    "scores_long", "scores_wide", "loadings_long", "evr_df", "fpca_tests_df",
    "clinical_corr_df", "ml_fold_df", "ml_oof_metrics_df", "ml_oof_df", "ml_coef_summary",
    "full_results_zip_bytes",
    "last_preprocess_run_id", "last_fpca_run_id", "pending_analysis_run_id",
]


def clear_analysis_state() -> None:
    """파싱 결과는 유지하고, 전처리 이후 분석 결과만 초기화한다."""
    for key in ANALYSIS_STATE_KEYS:
        st.session_state.pop(key, None)


def clear_parsed_and_analysis_state() -> None:
    """업로드 파싱 결과와 분석 결과를 모두 초기화한다."""
    for key in PARSED_STATE_KEYS + ANALYSIS_STATE_KEYS:
        st.session_state.pop(key, None)


def make_upload_signature(
    institution_zip_files: Dict[str, object],
    institution_crf_files: Dict[str, object],
    walking_regex: str,
    use_crf_trial_filter: bool,
) -> str:
    """업로드 파일/파싱 옵션 변경 여부를 감지하기 위한 가벼운 signature."""
    rows = [
        ("walking_regex", str(walking_regex)),
        ("use_crf_trial_filter", bool(use_crf_trial_filter)),
    ]
    for inst in INSTITUTIONS:
        z = institution_zip_files.get(inst)
        c = institution_crf_files.get(inst)
        rows.append((inst, "zip", getattr(z, "name", None), getattr(z, "size", None)))
        rows.append((inst, "crf", getattr(c, "name", None), getattr(c, "size", None)))
    return repr(rows)


def parse_uploaded_inputs_once(
    institution_zip_files: Dict[str, object],
    institution_crf_files: Dict[str, object],
    walking_regex: str,
    use_crf_trial_filter: bool,
) -> Tuple[pd.DataFrame, Dict[str, ParsedUpload], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """기관별 ZIP/CRF를 한 번 파싱하고, CRF trial filter까지 반영한다."""
    file_index, parsed, skipped_files, pair_df = build_upload_index_from_institution_zips(
        institution_zip_files,
        walking_regex=walking_regex,
    )

    crf_frames = []
    for inst, f in institution_crf_files.items():
        if f is not None:
            crf_frames.append(read_institution_crf(f, inst))
    crf = pd.concat(crf_frames, ignore_index=True, sort=False) if crf_frames else pd.DataFrame()
    crf_mapping_df = build_crf_mapping_diagnostics(pair_df, crf) if not crf.empty else pd.DataFrame()

    if use_crf_trial_filter and not crf_mapping_df.empty:
        has_any_trial_status_col = bool(crf_mapping_df.get("crf_trial_status_col_present", pd.Series(dtype=bool)).any())
        if has_any_trial_status_col:
            allowed = crf_mapping_df.loc[
                crf_mapping_df["trial_usable_by_crf"] & crf_mapping_df["mapping_status"].isin(["ok"]),
                ["institution", "subject_id", "trial_id"],
            ]
            allowed_keys = set(map(tuple, allowed.astype(str).to_numpy()))
            if not file_index.empty:
                file_index = file_index[
                    file_index[["institution", "subject_id", "trial_id"]].astype(str).apply(tuple, axis=1).isin(allowed_keys)
                ].reset_index(drop=True)
            pair_df = build_trial_pair_diagnostics(file_index)
            parsed = {
                k: v for k, v in parsed.items()
                if (str(v.institution), str(v.subject_id), str(v.trial_id)) in allowed_keys
            }

    return file_index, parsed, skipped_files, pair_df, crf, crf_mapping_df

# =============================================================================
# Streamlit UI
# =============================================================================


st.title("🚶 OpenCap Gait 분석 웹서비스")
st.caption("MOT/TRC와 CRF를 비식별번호로 매핑하고, 전처리 → FDA/fPCA → 임상척도 연결 → ML까지 한 흐름으로 분석합니다.")

# -------------------------------
# Sidebar: 업로드 및 하이퍼파라미터
# -------------------------------

st.sidebar.header("1) 기관별 데이터 업로드")
st.sidebar.caption("온라인 배포 환경에서는 폴더를 그대로 올리는 것이 아니라, 기관별 검사 데이터 폴더를 ZIP으로 압축해서 업로드합니다.")

institution_zip_files = {}
institution_crf_files = {}
with st.sidebar.expander("UNI 업로드", expanded=True):
    institution_zip_files["UNI"] = st.file_uploader("UNI 검사 데이터 ZIP", type=["zip"], key="uni_zip")
    institution_crf_files["UNI"] = st.file_uploader("UNI CRF", type=["xlsx", "xls", "csv"], key="uni_crf")
with st.sidebar.expander("UUH 업로드", expanded=True):
    institution_zip_files["UUH"] = st.file_uploader("UUH 검사 데이터 ZIP", type=["zip"], key="uuh_zip")
    institution_crf_files["UUH"] = st.file_uploader("UUH CRF", type=["xlsx", "xls", "csv"], key="uuh_crf")
with st.sidebar.expander("JBH 업로드", expanded=True):
    institution_zip_files["JBH"] = st.file_uploader("JBH 검사 데이터 ZIP", type=["zip"], key="jbh_zip")
    institution_crf_files["JBH"] = st.file_uploader("JBH CRF", type=["xlsx", "xls", "csv"], key="jbh_crf")

st.sidebar.header("2) Walking trial 선별 설정")
walking_regex = st.sidebar.text_input("walking 파일명 정규식", value=DEFAULT_WALKING_REGEX)
st.sidebar.caption("기본값은 6m, 6m_1, 6m_2, 6m_1(1)만 포함하고 TUG/standing은 제외합니다.")
use_crf_trial_filter = st.sidebar.checkbox("CRF에서 사용 가능(O/△)인 6m trial만 분석", value=True)
st.sidebar.info("업로드 제한은 .streamlit/config.toml에서 4096MB로 설정했습니다. Streamlit Cloud의 실제 메모리 한계는 별도 영향을 받을 수 있습니다.")

st.sidebar.header("3) 전처리 하이퍼파라미터")
n_grid = st.sidebar.slider("x축 정규화 grid 수", 51, 201, 101, 10)
outlier_pct = st.sidebar.slider("환자 내 이상 궤적 cutoff percentile", 80, 99, 95, 1)
use_kalman = st.sidebar.checkbox("결측치 Kalman filter 보정", value=True)
spline_smoothing = st.sidebar.number_input("스플라인 smoothing 강도", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
stance_pct = st.sidebar.slider("Stance/Swing 분기점 (%)", 40, 70, 60, 1)

st.sidebar.header("4) fPCA / 검정 / ML 설정")
n_fpc = st.sidebar.slider("feature당 fPC 개수", 1, 5, 2, 1)
n_perm = st.sidebar.slider("Permutation 횟수", 200, 10000, 1000, 200)
cv_splits = st.sidebar.slider("ML Stratified K-fold", 2, 10, 5, 1)
logistic_c = st.sidebar.number_input("Logistic L2 C", min_value=0.001, max_value=100.0, value=1.0, step=0.1)
random_seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

# -------------------------------
# 실행 제어
# -------------------------------

st.sidebar.header("5) 실행 제어")
current_upload_signature = make_upload_signature(
    institution_zip_files,
    institution_crf_files,
    walking_regex,
    use_crf_trial_filter,
)

parse_clicked = st.sidebar.button("① 업로드 자료 파싱/매핑 실행", type="primary", use_container_width=True)
analysis_clicked = st.sidebar.button("② 분석 시작", use_container_width=True)
clear_clicked = st.sidebar.button("파싱/분석 결과 초기화", use_container_width=True)

if clear_clicked:
    clear_parsed_and_analysis_state()
    st.sidebar.success("파싱 및 분석 결과를 초기화했습니다.")

if parse_clicked:
    if not any(institution_zip_files.values()):
        st.sidebar.error("최소 1개 기관의 검사 데이터 ZIP을 업로드하세요.")
    else:
        try:
            with st.spinner("기관별 ZIP/CRF를 파싱하고 매핑 진단을 생성하는 중..."):
                file_index_p, parsed_p, skipped_files_p, pair_df_p, crf_p, crf_mapping_df_p = parse_uploaded_inputs_once(
                    institution_zip_files=institution_zip_files,
                    institution_crf_files=institution_crf_files,
                    walking_regex=walking_regex,
                    use_crf_trial_filter=use_crf_trial_filter,
                )
            st.session_state["file_index"] = file_index_p
            st.session_state["parsed_uploads"] = parsed_p
            st.session_state["skipped_files"] = skipped_files_p
            st.session_state["pair_df"] = pair_df_p
            st.session_state["crf"] = crf_p
            st.session_state["crf_mapping_df"] = crf_mapping_df_p
            st.session_state["upload_signature"] = current_upload_signature
            st.session_state["parse_done"] = True
            clear_analysis_state()
            st.sidebar.success("파싱/매핑 완료. 이제 파라미터와 변수 선택 후 분석을 시작하세요.")
        except Exception as e:
            st.sidebar.error(f"파싱/매핑 중 오류: {e}")

# 파싱된 결과는 session_state에서만 가져온다. rerun마다 ZIP을 다시 풀지 않는다.
parsed: Dict[str, ParsedUpload] = st.session_state.get("parsed_uploads", {})
file_index = st.session_state.get("file_index", pd.DataFrame())
skipped_files = st.session_state.get("skipped_files", pd.DataFrame())
pair_df = st.session_state.get("pair_df", pd.DataFrame())
crf = st.session_state.get("crf", pd.DataFrame())
crf_mapping_df = st.session_state.get("crf_mapping_df", pd.DataFrame())

if st.session_state.get("parse_done", False):
    if st.session_state.get("upload_signature") != current_upload_signature:
        st.sidebar.warning("업로드 파일 또는 walking/CRF 필터 옵션이 변경되었습니다. 다시 파싱/매핑 실행을 눌러주세요.")
    else:
        st.sidebar.success("파싱된 업로드 자료를 사용 중입니다. 분석 시 ZIP을 다시 파싱하지 않습니다.")

if analysis_clicked:
    if not st.session_state.get("parse_done", False) or not parsed:
        st.sidebar.error("먼저 ① 업로드 자료 파싱/매핑 실행을 완료하세요.")
    else:
        clear_analysis_state()
        next_run_id = int(st.session_state.get("analysis_run_id", 0)) + 1
        st.session_state["analysis_run_id"] = next_run_id
        st.session_state["pending_analysis_run_id"] = next_run_id
        st.sidebar.success("분석 시작 신호를 보냈습니다. 현재 선택된 파라미터와 변수로 전처리/FDA/fPCA를 실행합니다.")

# -------------------------------
# Sidebar sample size
st.sidebar.header("6) 데이터 샘플 사이즈")
if not file_index.empty:
    mot_counts = file_index[file_index["file_type"] == "MOT"].groupby(["institution", "subject_id"]).size().rename("n_mot")
    trc_counts = file_index[file_index["file_type"] == "TRC"].groupby(["institution", "subject_id"]).size().rename("n_trc")
    sample_size = pd.concat([mot_counts, trc_counts], axis=1).fillna(0).astype(int).reset_index()
    st.sidebar.metric("총 subject 수", sample_size["subject_id"].nunique())
    st.sidebar.metric("총 MOT 파일 수", int((file_index["file_type"] == "MOT").sum()))
    st.sidebar.metric("총 TRC 파일 수", int((file_index["file_type"] == "TRC").sum()))
    with st.sidebar.expander("환자당 파일 개수"):
        st.dataframe(sample_size, use_container_width=True, height=260)
else:
    st.sidebar.info("기관별 ZIP을 업로드하면 환자당 파일 개수가 표시됩니다.")

# -------------------------------
# Main tabs
# -------------------------------

tab_upload, tab_pre, tab_fpca, tab_clinical, tab_ml = st.tabs(
    ["1. 업로드 자료", "2. Gait 구간 전처리", "3. FDA / fPCA", "4. 임상척도 연결", "5. ML 적합"]
)

# =============================================================================
# 1. 업로드 자료
# =============================================================================

with tab_upload:
    st.subheader("1. 업로드 자료 및 비식별번호 매핑")

    if file_index.empty:
        st.info("왼쪽 사이드바에서 기관별 검사 데이터 ZIP을 업로드하세요. MOT/TRC는 같은 비식별 폴더 안에 있어도 확장자로 구분합니다.")
    else:
        st.markdown("#### 업로드 파일 index: 6m walking trial만 포함")
        st.dataframe(file_index, use_container_width=True)

        if not pair_df.empty:
            st.markdown("#### MOT/TRC trial pair 진단")
            st.dataframe(pair_df, use_container_width=True)

        if not skipped_files.empty:
            with st.expander("제외된 파일 목록: non-walking/TUG/파싱 실패"):
                st.dataframe(skipped_files, use_container_width=True)

        mot_feature_union = []
        for item in parsed.values():
            if item.kind == "MOT":
                mot_feature_union.extend(numeric_feature_columns(item.df, item.time_col))
        mot_feature_union = sorted(set(mot_feature_union))

        st.markdown("#### MOT에서 감지된 numeric gait 변수")
        st.write(f"감지된 후보 변수 수: **{len(mot_feature_union)}개**")
        with st.expander("후보 변수 보기"):
            st.write(mot_feature_union)

    if crf.empty:
        st.info("기관별 CRF 엑셀 또는 CSV를 업로드하면 subject/group/covariate를 선택할 수 있습니다. 그룹은 CRF의 `피험자군` 변수를 우선 사용합니다.")
    else:
        st.markdown("#### CRF 표준화 미리보기")
        st.dataframe(crf.head(50), use_container_width=True)

        crf_cols = crf.columns.tolist()
        default_subject_idx = 0
        for cand in ["subject_id", "피험자번호", "SubjectID", "ID", "id", "비식별번호"]:
            if cand in crf_cols:
                default_subject_idx = crf_cols.index(cand)
                break
        subject_col = st.selectbox("CRF subject ID 컬럼", crf_cols, index=default_subject_idx, key="subject_col")

        default_group_idx = 0
        for cand in ["피험자군", "group", "Group", "diagnosis", "Diagnosis", "군", "그룹"]:
            if cand in crf_cols:
                default_group_idx = crf_cols.index(cand)
                break
        group_col = st.selectbox("그룹 컬럼", crf_cols, index=default_group_idx, key="group_col")

        crf_tmp = crf.copy()
        crf_tmp[subject_col] = crf_tmp[subject_col].astype(str)
        if not file_index.empty:
            upload_sids = set(file_index["subject_id"].astype(str))
            crf_sids = set(crf_tmp[subject_col].astype(str))
            matched = upload_sids & crf_sids
            col1, col2, col3 = st.columns(3)
            col1.metric("업로드 subject", len(upload_sids))
            col2.metric("CRF subject", len(crf_sids))
            col3.metric("매핑 성공 subject", len(matched))
            if len(matched) == 0:
                st.warning("기관별 ZIP 내부 비식별 폴더명과 CRF subject ID가 매칭되지 않습니다. 폴더명이 UNI1/UUH1/JBH1 형식인지, CRF의 피험자번호가 맞는지 확인하세요.")

        if not crf_mapping_df.empty:
            st.markdown("#### CRF ↔ walking trial 매핑 진단")
            st.dataframe(crf_mapping_df, use_container_width=True)
            bad_map = crf_mapping_df[~crf_mapping_df["crf_subject_matched"]]
            if not bad_map.empty:
                st.error(f"CRF subject와 매칭되지 않은 walking trial이 {bad_map.shape[0]}개 있습니다.")

        group_values = crf_tmp[group_col].dropna().astype(str).unique().tolist()
        if group_values:
            control_default_idx = next(
                (i for i, v in enumerate(group_values) if re.search(r"control|normal|healthy|hc|정상", v, flags=re.IGNORECASE)),
                0,
            )
            disease_default_idx = next(
                (i for i, v in enumerate(group_values) if re.search(r"parkinson|pd|disease|patient|환자|질환", v, flags=re.IGNORECASE)),
                1 if len(group_values) > 1 else 0,
            )
            if disease_default_idx == control_default_idx and len(group_values) > 1:
                disease_default_idx = 1 - control_default_idx if control_default_idx in [0, 1] else 1
            control_label = st.selectbox("정상군 label", group_values, index=control_default_idx, key="control_label")
            disease_label = st.selectbox("질환군 label", group_values, index=disease_default_idx, key="disease_label")
        else:
            st.warning("그룹 컬럼에 값이 없습니다.")

        likely_covs = [c for c in ["age_auto", "sex_auto", "height_auto", "weight_auto", "dominant_hand_auto", "dominant_foot_auto", "만 나이", "성별", "키 (m)", "체중 (kg)", "age", "sex", "height", "weight", "dominant_hand", "dominant_foot", "Age", "Sex", "Height", "Weight"] if c in crf_cols]
        covariates = st.multiselect(
            "공변량 선택: 나이, 성별, 키, 체중, 주손, 주발 등",
            crf_cols,
            default=likely_covs,
            key="covariates",
        )

# Store shared config after tab widgets exist
subject_col = st.session_state.get("subject_col", None)
group_col = st.session_state.get("group_col", None)
control_label = st.session_state.get("control_label", None)
disease_label = st.session_state.get("disease_label", None)
covariates = st.session_state.get("covariates", [])

# =============================================================================
# 2. Gait 구간 전처리
# =============================================================================

with tab_pre:
    st.subheader("2. Gait 구간 전처리")
    st.markdown(
        "MOT 곡선을 환자별 평균 curve로 변환합니다. 결측은 Kalman filter 또는 interpolation으로 보정하고, "
        "환자 내 trial 거리 기준 상위 outlier를 제외한 뒤 평균 curve를 계산합니다."
    )

    if not parsed:
        st.info("먼저 MOT 파일을 업로드하세요.")
    else:
        mot_feature_union = []
        for item in parsed.values():
            if item.kind == "MOT":
                mot_feature_union.extend(numeric_feature_columns(item.df, item.time_col))
        mot_feature_union = sorted(set(mot_feature_union))

        default_features = mot_feature_union[: min(10, len(mot_feature_union))]
        selected_features = st.multiselect(
            "분석할 gait 변수 선택",
            mot_feature_union,
            default=default_features,
            key="selected_features",
        )

        st.caption("처음 실행은 5~15개 주요 관절 변수로 확인하고, 이후 전체 변수로 확장하는 것을 권장합니다.")

        pending_run_id = st.session_state.get("pending_analysis_run_id", None)
        run_pre_from_sidebar = pending_run_id is not None and st.session_state.get("last_preprocess_run_id") != pending_run_id
        run_pre_now = st.button("전처리 실행", type="primary") or run_pre_from_sidebar

        if run_pre_now:
            if not selected_features:
                st.error("분석할 gait 변수를 최소 1개 선택하세요.")
                if run_pre_from_sidebar:
                    st.session_state["pending_analysis_run_id"] = None
            else:
                with st.spinner("Gait curve 전처리 중..."):
                    long_df, qc_df, excluded_df = preprocess_to_subject_curves(
                        parsed=parsed,
                        selected_features=selected_features,
                        n_grid=n_grid,
                        use_kalman=use_kalman,
                        spline_smoothing=spline_smoothing,
                        outlier_percentile=outlier_pct,
                        stance_pct=stance_pct,
                    )
                    st.session_state["curve_long"] = long_df
                    st.session_state["qc_df"] = qc_df
                    st.session_state["excluded_df"] = excluded_df
                    st.session_state["raw_matrices"] = long_to_feature_matrices(long_df)
                    if pending_run_id is not None:
                        st.session_state["last_preprocess_run_id"] = pending_run_id
                st.success("전처리 완료")

        long_df = st.session_state.get("curve_long", pd.DataFrame())
        qc_df = st.session_state.get("qc_df", pd.DataFrame())
        excluded_df = st.session_state.get("excluded_df", pd.DataFrame())

        if not long_df.empty:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("전처리 subject", long_df["subject_id"].nunique())
            c2.metric("분석 feature", long_df["feature"].nunique())
            c3.metric("grid points", long_df["grid_pct"].nunique())
            c4.metric("long rows", len(long_df))

            st.markdown("#### FDA 분석용 tidy/tablet 데이터")
            st.dataframe(long_df.head(300), use_container_width=True)
            make_download_button_csv(long_df, "전처리 long table 다운로드", "opencap_preprocessed_curves_long.csv")

            with st.expander("QC 결과"):
                st.dataframe(qc_df, use_container_width=True)
                make_download_button_csv(qc_df, "QC CSV 다운로드", "opencap_preprocessing_qc.csv")

            with st.expander("이상 trajectory 제외 기록"):
                st.dataframe(excluded_df, use_container_width=True)
                make_download_button_csv(excluded_df, "이상 trajectory 기록 다운로드", "opencap_excluded_trajectories.csv")
        else:
            st.info("전처리 실행 후 결과가 여기에 표시됩니다.")

# =============================================================================
# 3. FDA / fPCA
# =============================================================================

with tab_fpca:
    st.subheader("3. FDA / fPCA 적합")
    long_df = st.session_state.get("curve_long", pd.DataFrame())
    raw_matrices = st.session_state.get("raw_matrices", {})

    if long_df.empty or crf.empty or subject_col is None or group_col is None:
        st.info("전처리 결과와 CRF 설정이 필요합니다.")
    else:
        crf_meta = crf.copy()
        crf_meta[subject_col] = crf_meta[subject_col].astype(str)

        pending_run_id = st.session_state.get("pending_analysis_run_id", None)
        run_fpca_from_sidebar = (
            pending_run_id is not None
            and st.session_state.get("last_preprocess_run_id") == pending_run_id
            and st.session_state.get("last_fpca_run_id") != pending_run_id
        )
        run_fpca_now = st.button("공변량 보정 후 FDA/fPCA 실행", type="primary") or run_fpca_from_sidebar

        if run_fpca_now:
            with st.spinner("공변량 보정 및 fPCA 적합 중..."):
                adjusted_matrices = adjust_all_feature_matrices(
                    raw_matrices,
                    crf_meta,
                    subject_col=subject_col,
                    covariates=covariates,
                )
                scores_long, loadings_long, evr_df, scores_wide = run_fpca_all_features(adjusted_matrices, n_components=n_fpc)
                tests_df = fpca_2d_tests(
                    scores_long,
                    crf_meta,
                    subject_col=subject_col,
                    group_col=group_col,
                    control_label=control_label,
                    disease_label=disease_label,
                    n_perm=n_perm,
                )
                st.session_state["adjusted_matrices"] = adjusted_matrices
                st.session_state["adjusted_long"] = matrix_to_long(adjusted_matrices)
                st.session_state["scores_long"] = scores_long
                st.session_state["scores_wide"] = scores_wide
                st.session_state["loadings_long"] = loadings_long
                st.session_state["evr_df"] = evr_df
                st.session_state["fpca_tests_df"] = tests_df
                if pending_run_id is not None:
                    st.session_state["last_fpca_run_id"] = pending_run_id
                    st.session_state["pending_analysis_run_id"] = None
            st.success("FDA/fPCA 분석 완료")

        adjusted_matrices = st.session_state.get("adjusted_matrices", {})
        scores_long = st.session_state.get("scores_long", pd.DataFrame())
        scores_wide = st.session_state.get("scores_wide", pd.DataFrame())
        loadings_long = st.session_state.get("loadings_long", pd.DataFrame())
        evr_df = st.session_state.get("evr_df", pd.DataFrame())
        tests_df = st.session_state.get("fpca_tests_df", pd.DataFrame())

        if adjusted_matrices:
            features = list(adjusted_matrices.keys())
            selected_feat = st.selectbox("시각화할 feature", features, key="fpca_selected_feat")

            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.markdown("#### FDA mean curve + bootstrap CI + 유의구간")
                fig = plot_fda_group_mean(
                    adjusted_matrices[selected_feat],
                    crf_meta,
                    subject_col=subject_col,
                    group_col=group_col,
                    groups_to_plot=[control_label, disease_label],
                    title=f"Adjusted FDA curve: {selected_feat}",
                    show_significance=True,
                    stance_pct=stance_pct,
                )
                st.pyplot(fig, clear_figure=True)
            with c2:
                st.markdown("#### fPCA loading")
                if not loadings_long.empty:
                    st.pyplot(plot_loading(loadings_long, selected_feat), clear_figure=True)
                else:
                    st.info("loading 결과가 없습니다.")

            c3, c4 = st.columns([1.1, 1])
            with c3:
                st.markdown("#### fPCA 2차원 분포")
                if not scores_long.empty and {"FPC1", "FPC2"}.issubset(scores_long.columns):
                    st.pyplot(
                        plot_fpca_scatter(
                            scores_long,
                            crf_meta,
                            subject_col,
                            group_col,
                            selected_feat,
                            [control_label, disease_label],
                            title=f"{selected_feat}: FPC1-FPC2 distribution",
                        ),
                        clear_figure=True,
                    )
                else:
                    st.info("FPC1/FPC2 score가 없습니다.")
            with c4:
                st.markdown("#### 특정 fPC 그룹별 boxplot")
                pc_options = [f"FPC{i}" for i in range(1, n_fpc + 1) if f"FPC{i}" in scores_long.columns]
                if pc_options:
                    selected_pc = st.selectbox("fPC 선택", pc_options)
                    st.pyplot(
                        plot_fpca_box(scores_long, crf_meta, subject_col, group_col, selected_feat, selected_pc, [control_label, disease_label]),
                        clear_figure=True,
                    )

            st.markdown("#### fPCA 설명분산")
            st.dataframe(evr_df, use_container_width=True)

            st.markdown("#### FPC1+FPC2 2차원 그룹 분포 차이 검정")
            st.caption("Hotelling T², PERMANOVA, Energy distance를 함께 제공합니다. q값은 FDR 보정 결과입니다.")
            st.dataframe(tests_df, use_container_width=True)

            zip_files = {
                "adjusted_curves_long.csv": st.session_state.get("adjusted_long", pd.DataFrame()),
                "fpca_scores_long.csv": scores_long,
                "fpca_scores_wide.csv": scores_wide,
                "fpca_loadings_long.csv": loadings_long,
                "fpca_explained_variance.csv": evr_df,
                "fpca_2d_group_tests.csv": tests_df,
            }
            st.download_button(
                "FDA/fPCA 결과 ZIP 다운로드",
                data=dataframe_to_zip_bytes(zip_files),
                file_name="opencap_fda_fpca_results.zip",
                mime="application/zip",
            )
        else:
            st.info("공변량 보정 후 FDA/fPCA 실행 버튼을 누르세요.")

# =============================================================================
# 4. 임상척도 연결 및 해석
# =============================================================================

with tab_clinical:
    st.subheader("4. 임상척도 연결 및 해석")
    adjusted_matrices = st.session_state.get("adjusted_matrices", {})
    scores_wide = st.session_state.get("scores_wide", pd.DataFrame())

    if not adjusted_matrices or scores_wide.empty or crf.empty or subject_col is None:
        st.info("먼저 FDA/fPCA 분석을 실행하세요.")
    else:
        crf_meta = crf.copy()
        crf_meta[subject_col] = crf_meta[subject_col].astype(str)
        disease_mask = crf_meta[group_col].astype(str) == str(disease_label)
        disease_meta = crf_meta.loc[disease_mask].copy()

        st.caption("정상군에는 UPDRS/HY가 없을 수 있으므로, 임상척도 분석은 기본적으로 질환군 내부에서만 수행합니다.")

        crf_cols = crf_meta.columns.tolist()
        default_hy = next((c for c in ["hy_stage_auto", "호엔-야 단계", "호엔-야 단계_1", "HY_stage", "HY", "Hoehn_Yahr", "HoehnYahr", "hoehn_yahr"] if c in crf_cols), crf_cols[0])
        hy_col = st.selectbox("HY / 호엔야 등급 컬럼", crf_cols, index=crf_cols.index(default_hy))

        features = list(adjusted_matrices.keys())
        selected_feat_clin = st.selectbox("HY별 FDA curve 확인 feature", features, key="clin_feat")

        hy_levels_all = disease_meta[hy_col].dropna().astype(str).unique().tolist() if hy_col in disease_meta.columns else []
        selected_hy_levels = st.multiselect("표시할 HY 등급", hy_levels_all, default=hy_levels_all[: min(4, len(hy_levels_all))])

        if selected_hy_levels:
            # group_col 대신 임시 HY 표시 컬럼 사용
            plot_meta = crf_meta.copy()
            plot_meta["__HY_LEVEL__"] = plot_meta[hy_col].astype(str)
            plot_meta = plot_meta.loc[disease_mask].copy()
            # 질환군 subject만 matrix에서 선택
            mat = adjusted_matrices[selected_feat_clin].copy()
            disease_sids = set(plot_meta[subject_col].astype(str))
            mat = mat.loc[[sid for sid in mat.index.astype(str) if sid in disease_sids]]
            fig = plot_fda_group_mean(
                mat,
                plot_meta,
                subject_col=subject_col,
                group_col="__HY_LEVEL__",
                groups_to_plot=selected_hy_levels,
                title=f"Disease-only FDA curve by HY: {selected_feat_clin}",
                show_significance=False,
                stance_pct=stance_pct,
            )
            st.pyplot(fig, clear_figure=True)

        st.markdown("#### UPDRS 등 임상점수와 fPCA score 상관성")
        likely_updrs = [c for c in crf_cols if re.search("UPDRS|MDS|motor|score|updrs_auto", str(c), flags=re.IGNORECASE)]
        clinical_vars = st.multiselect("상관분석할 임상척도", crf_cols, default=likely_updrs[: min(3, len(likely_updrs))])

        if st.button("임상척도 상관성 계산", type="primary"):
            corr_df = fpca_clinical_correlation(scores_wide, crf_meta, subject_col, disease_mask, clinical_vars)
            st.session_state["clinical_corr_df"] = corr_df

        corr_df = st.session_state.get("clinical_corr_df", pd.DataFrame())
        if not corr_df.empty:
            st.dataframe(corr_df, use_container_width=True)
            make_download_button_csv(corr_df, "임상척도 상관성 CSV 다운로드", "opencap_fpca_clinical_correlations.csv")
        else:
            st.info("임상척도와 fPCA score의 Spearman 상관성 테이블이 여기에 표시됩니다.")

# =============================================================================
# 5. ML 적합
# =============================================================================

with tab_ml:
    st.subheader("5. ML 적합: fPCA 기반 로지스틱 회귀")
    st.markdown(
        "이 단계는 버튼을 눌렀을 때만 계산합니다. 각 fold 내부에서만 공변량 보정과 PCA를 학습하므로, "
        "train/test 간 fPCA 분포 정보 누수를 방지합니다. 데이터 불균형은 `class_weight='balanced'`로 처리합니다."
    )

    raw_matrices = st.session_state.get("raw_matrices", {})
    if not raw_matrices or crf.empty or subject_col is None or group_col is None:
        st.info("전처리 결과와 CRF 설정이 필요합니다.")
    else:
        all_features = list(raw_matrices.keys())
        ml_features = st.multiselect("ML에 사용할 gait feature", all_features, default=all_features[: min(20, len(all_features))])
        st.caption("feature가 너무 많고 표본이 적으면 불안정할 수 있습니다. 먼저 상위/주요 feature로 시작하세요.")

        if st.button("누수 방지 5-fold ML 실행", type="primary"):
            if not ml_features:
                st.error("ML에 사용할 feature를 선택하세요.")
            else:
                try:
                    with st.spinner("Fold 내부 fPCA + 로지스틱 회귀 실행 중..."):
                        fold_df, oof_metrics_df, oof_df, coef_summary = run_no_leakage_ml(
                            raw_matrices=raw_matrices,
                            meta=crf,
                            subject_col=subject_col,
                            group_col=group_col,
                            control_label=control_label,
                            disease_label=disease_label,
                            covariates=covariates,
                            selected_features=ml_features,
                            n_components=n_fpc,
                            n_splits=cv_splits,
                            c_value=logistic_c,
                            random_state=int(random_seed),
                        )
                        st.session_state["ml_fold_df"] = fold_df
                        st.session_state["ml_oof_metrics_df"] = oof_metrics_df
                        st.session_state["ml_oof_df"] = oof_df
                        st.session_state["ml_coef_summary"] = coef_summary
                    st.success("ML 분석 완료")
                except Exception as e:
                    st.error(f"ML 분석 중 오류: {e}")

        fold_df = st.session_state.get("ml_fold_df", pd.DataFrame())
        oof_metrics_df = st.session_state.get("ml_oof_metrics_df", pd.DataFrame())
        oof_df = st.session_state.get("ml_oof_df", pd.DataFrame())
        coef_summary = st.session_state.get("ml_coef_summary", pd.DataFrame())

        if not fold_df.empty:
            st.markdown("#### Fold별 성능")
            st.dataframe(fold_df, use_container_width=True)

            metric_cols = ["AUC", "Accuracy", "Sensitivity", "Specificity"]
            avg_rows = []
            for m in metric_cols:
                avg_rows.append(
                    {
                        "metric": m,
                        "mean": fold_df[m].mean(),
                        "sd": fold_df[m].std(),
                    }
                )
            avg_df = pd.DataFrame(avg_rows)
            st.markdown("#### 5-fold 평균(표준편차)")
            st.dataframe(avg_df, use_container_width=True)

            st.markdown("#### OOF 성능")
            st.dataframe(oof_metrics_df, use_container_width=True)
            st.pyplot(plot_roc_from_oof(oof_df), clear_figure=True)

            st.markdown("#### 표준화 계수 정보")
            st.dataframe(coef_summary, use_container_width=True)

            zip_files = {
                "ml_fold_metrics.csv": fold_df,
                "ml_fold_mean_sd.csv": avg_df,
                "ml_oof_metrics.csv": oof_metrics_df,
                "ml_oof_predictions.csv": oof_df,
                "ml_logistic_coefficients.csv": coef_summary,
            }
            st.download_button(
                "ML 결과 ZIP 다운로드",
                data=dataframe_to_zip_bytes(zip_files),
                file_name="opencap_ml_results.zip",
                mime="application/zip",
            )
        else:
            st.info("ML 실행 버튼을 누르면 ROC, AUC, 정확도, 민감도, 특이도, 계수 테이블이 표시됩니다.")

# =============================================================================
# 최하단 전체 결과 ZIP 다운로드
# =============================================================================

st.divider()
st.subheader("전체 분석 자료 ZIP 다운로드")
include_figures_in_zip = st.checkbox("ZIP에 현재 선택된 주요 그래프 PNG 포함", value=True)

def fig_to_png_bytes(fig) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", dpi=180, bbox_inches="tight")
    bio.seek(0)
    return bio.read()


def make_full_results_zip(include_figures: bool = True) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        tables = {
            "00_upload/file_index_walking_trials.csv": file_index,
            "00_upload/mot_trc_pair_diagnostics.csv": pair_df,
            "00_upload/skipped_nonwalking_or_failed_files.csv": skipped_files,
            "00_upload/crf_standardized.csv": crf,
            "00_upload/crf_mapping_diagnostics.csv": crf_mapping_df,
            "01_preprocessing/preprocessed_curves_long.csv": st.session_state.get("curve_long", pd.DataFrame()),
            "01_preprocessing/preprocessing_qc.csv": st.session_state.get("qc_df", pd.DataFrame()),
            "01_preprocessing/excluded_trajectories.csv": st.session_state.get("excluded_df", pd.DataFrame()),
            "02_fda_fpca/adjusted_curves_long.csv": st.session_state.get("adjusted_long", pd.DataFrame()),
            "02_fda_fpca/fpca_scores_long.csv": st.session_state.get("scores_long", pd.DataFrame()),
            "02_fda_fpca/fpca_scores_wide.csv": st.session_state.get("scores_wide", pd.DataFrame()),
            "02_fda_fpca/fpca_loadings_long.csv": st.session_state.get("loadings_long", pd.DataFrame()),
            "02_fda_fpca/fpca_explained_variance.csv": st.session_state.get("evr_df", pd.DataFrame()),
            "02_fda_fpca/fpca_2d_group_tests.csv": st.session_state.get("fpca_tests_df", pd.DataFrame()),
            "03_clinical/fpca_clinical_correlations.csv": st.session_state.get("clinical_corr_df", pd.DataFrame()),
            "04_ml/ml_fold_metrics.csv": st.session_state.get("ml_fold_df", pd.DataFrame()),
            "04_ml/ml_oof_metrics.csv": st.session_state.get("ml_oof_metrics_df", pd.DataFrame()),
            "04_ml/ml_oof_predictions.csv": st.session_state.get("ml_oof_df", pd.DataFrame()),
            "04_ml/ml_logistic_coefficients.csv": st.session_state.get("ml_coef_summary", pd.DataFrame()),
        }
        settings = pd.DataFrame([
            {"setting": "walking_regex", "value": walking_regex},
            {"setting": "use_crf_trial_filter", "value": use_crf_trial_filter},
            {"setting": "n_grid", "value": n_grid},
            {"setting": "outlier_percentile", "value": outlier_pct},
            {"setting": "use_kalman", "value": use_kalman},
            {"setting": "spline_smoothing", "value": spline_smoothing},
            {"setting": "stance_pct", "value": stance_pct},
            {"setting": "n_fpc", "value": n_fpc},
            {"setting": "n_perm", "value": n_perm},
            {"setting": "cv_splits", "value": cv_splits},
            {"setting": "logistic_c", "value": logistic_c},
            {"setting": "subject_col", "value": subject_col},
            {"setting": "group_col", "value": group_col},
            {"setting": "control_label", "value": control_label},
            {"setting": "disease_label", "value": disease_label},
            {"setting": "covariates", "value": ";".join(map(str, covariates))},
        ])
        tables["00_upload/analysis_settings.csv"] = settings
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                zf.writestr(name, df.to_csv(index=False, encoding="utf-8-sig"))
        if include_figures:
            try:
                adjusted_matrices = st.session_state.get("adjusted_matrices", {})
                scores_long = st.session_state.get("scores_long", pd.DataFrame())
                loadings_long = st.session_state.get("loadings_long", pd.DataFrame())
                if adjusted_matrices and subject_col and group_col and control_label is not None and disease_label is not None:
                    feat = list(adjusted_matrices.keys())[0]
                    crf_meta = crf.copy()
                    crf_meta[subject_col] = crf_meta[subject_col].astype(str)
                    fig = plot_fda_group_mean(adjusted_matrices[feat], crf_meta, subject_col, group_col, [control_label, disease_label], f"FDA curve: {feat}", True, stance_pct=stance_pct)
                    zf.writestr(f"figures/fda_mean_curve_{safe_filename(feat)}.png", fig_to_png_bytes(fig))
                    plt.close(fig)
                    if not loadings_long.empty:
                        fig = plot_loading(loadings_long, feat)
                        zf.writestr(f"figures/fpca_loading_{safe_filename(feat)}.png", fig_to_png_bytes(fig))
                        plt.close(fig)
                    if not scores_long.empty and {"FPC1", "FPC2"}.issubset(scores_long.columns):
                        fig = plot_fpca_scatter(scores_long, crf_meta, subject_col, group_col, feat, [control_label, disease_label], f"FPC1-FPC2: {feat}")
                        zf.writestr(f"figures/fpca_scatter_{safe_filename(feat)}.png", fig_to_png_bytes(fig))
                        plt.close(fig)
                oof_df = st.session_state.get("ml_oof_df", pd.DataFrame())
                if not oof_df.empty:
                    fig = plot_roc_from_oof(oof_df)
                    zf.writestr("figures/ml_oof_roc.png", fig_to_png_bytes(fig))
                    plt.close(fig)
            except Exception as e:
                zf.writestr("figures/_figure_export_error.txt", str(e))
    mem.seek(0)
    return mem.read()

if st.button("전체 분석 자료 ZIP 생성/갱신", use_container_width=True):
    with st.spinner("전체 결과 ZIP을 생성하는 중..."):
        st.session_state["full_results_zip_bytes"] = make_full_results_zip(include_figures=include_figures_in_zip)
    st.success("전체 결과 ZIP 생성 완료")

full_zip_bytes = st.session_state.get("full_results_zip_bytes", None)
if full_zip_bytes is not None:
    st.download_button(
        "전체 분석 자료 ZIP 다운로드",
        data=full_zip_bytes,
        file_name="opencap_gait_full_analysis_results.zip",
        mime="application/zip",
        use_container_width=True,
    )
else:
    st.info("전체 결과 ZIP은 버튼을 눌렀을 때만 생성합니다. 대용량 결과에서 불필요한 재생성을 막기 위한 설정입니다.")

# =============================================================================
# 하단 주의사항
# =============================================================================

st.divider()
st.markdown(
    """
### 구현상 중요한 주의사항
- 입력은 기관별 검사 데이터 ZIP + 기관별 CRF입니다. GitHub에는 원자료를 올리지 않고, 웹앱에서 업로드합니다.
- MOT/TRC는 같은 비식별 폴더 안에 있어도 확장자로 구분합니다. 분석에는 `6m` walking trial만 포함하고 TUG/standing은 제외합니다.
- 그룹 분리는 기관명이 아니라 CRF의 `피험자군` 변수를 기준으로 합니다. 기관 안에 정상군과 질환군이 섞여 있어도 문제없습니다.
- MOT의 관절각/kinematics 컬럼을 주 분석 곡선으로 사용합니다. TRC는 현재 업로드/QC용으로 읽으며, marker 기반 gait event 자동 탐지는 별도 고도화가 필요합니다.
- Stance/Swing은 force plate나 heel-strike event가 없으면 기본적으로 사용자가 지정한 gait cycle percentage 기준으로 나눕니다.
- FDA/fPCA 탐색 탭은 전체 자료 기준 보정/분해 결과를 보여주는 시각화용입니다.
- ML 탭은 각 fold 내부에서 공변량 보정과 PCA를 다시 학습하여 test fold로 변환하므로, fPCA 분포 정보 누수를 방지합니다.
- 표본 수가 작으면 permutation p-value, fPCA score, ML 성능의 변동성이 큽니다. OOF 결과와 fold 평균/표준편차를 함께 보세요.
"""
)
