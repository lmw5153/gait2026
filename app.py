# -*- coding: utf-8 -*-
"""
OpenCap Gait 분석 웹서비스 - Analysis-only Streamlit App

이 앱은 원본 MOT/TRC를 직접 파싱하지 않는다.
오프라인/별도 전처리 단계에서 생성한 환자별 gait mean curve long table과 CRF를 입력받아
FDA/fPCA/임상척도 연결/누수 방지 ML 분석만 수행한다.

필수 gait long table 컬럼:
    subject_id, feature, grid_pct, value
선택 컬럼:
    institution, phase, n_trials_total, n_trials_kept

실행:
    streamlit run app.py
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Ellipse
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import fdrcorrection


# =============================================================================
# App config
# =============================================================================

st.set_page_config(
    page_title="OpenCap Gait 분석 서비스",
    page_icon="🚶",
    layout="wide",
)

REQUIRED_GAIT_COLS = {"subject_id", "feature", "grid_pct", "value"}
INSTITUTIONS = ["UNI", "UUH", "JBH"]
DEFAULT_VALID_STATUS = {"O", "△", "A", "OK", "가능", "사용", "사용가능"}


# =============================================================================
# Utility
# =============================================================================


def safe_filename(name: str, max_len: int = 120) -> str:
    s = str(name)
    s = re.sub(r"[\\/:*?\"<>|]+", "_", s)
    s = re.sub(r"\s+", "_", s).strip("._ ")
    return (s or "unnamed")[:max_len]


def _clean_col_name(x) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).replace("\n", " ")).strip()


def _make_unique_columns(cols: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen: Dict[str, int] = {}
    for i, c in enumerate(cols):
        c = _clean_col_name(c) or f"col_{i}"
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out


def normalize_subject_id_value(value, institution: str = "") -> str:
    inst = str(institution).upper().strip()
    if pd.isna(value):
        return ""
    raw = str(value).strip()
    raw = re.sub(r"\.0$", "", raw)
    raw = raw.replace(" ", "")
    if inst:
        m = re.search(rf"({inst}\d+)", raw, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search(r"([A-Za-z]{2,5}\d+)", raw)
    if m:
        return m.group(1).upper()
    m = re.search(r"\d+", raw)
    if m and inst:
        return f"{inst}{int(m.group(0))}"
    return raw.upper()


@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def read_excel_all_cached(file_bytes: bytes) -> Dict[str, pd.DataFrame]:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=None, header=None)


@st.cache_data(show_spinner=False)
def read_excel_sheet_plain_cached(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


def read_single_table(uploaded_file) -> pd.DataFrame:
    """csv/xlsx 파일 하나를 DataFrame으로 읽는다."""
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    b = uploaded_file.getvalue()
    if name.endswith(".csv"):
        return read_csv_cached(b)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(b))
    raise ValueError("지원하지 않는 파일 형식입니다. CSV 또는 XLSX를 올려주세요.")


def read_preprocessed_gait_file(uploaded_file) -> pd.DataFrame:
    """
    전처리된 gait curve 파일을 읽는다.
    지원:
    - 단일 CSV/XLSX
    - ZIP 안의 CSV/XLSX 여러 개. 모두 concat.
    """
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    b = uploaded_file.getvalue()
    frames: List[pd.DataFrame] = []

    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(b)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                rel = info.filename.replace("\\", "/")
                fn = Path(rel).name.lower()
                if fn.startswith("~$"):
                    continue
                raw = zf.read(info)
                try:
                    if fn.endswith(".csv"):
                        df = pd.read_csv(io.BytesIO(raw))
                    elif fn.endswith(".xlsx") or fn.endswith(".xls"):
                        df = pd.read_excel(io.BytesIO(raw))
                    else:
                        continue
                    df["source_file"] = rel
                    frames.append(df)
                except Exception as e:
                    st.warning(f"ZIP 내부 파일 읽기 실패: {rel} ({e})")
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(b))
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(io.BytesIO(b))
    raise ValueError("전처리 gait 데이터는 CSV/XLSX/ZIP만 지원합니다.")


def standardize_gait_long(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """전처리 gait long table 컬럼을 표준화하고 오류를 반환한다."""
    errors: List[str] = []
    if df.empty:
        return df, ["전처리 gait 데이터가 비어 있습니다."]

    out = df.copy()
    out.columns = [_clean_col_name(c) for c in out.columns]

    # 흔한 alias 보정
    alias = {
        "id": "subject_id",
        "subject": "subject_id",
        "Subject": "subject_id",
        "SUBJECT_ID": "subject_id",
        "participant_id": "subject_id",
        "variable": "feature",
        "joint": "feature",
        "grid": "grid_pct",
        "percent": "grid_pct",
        "time_pct": "grid_pct",
        "x": "grid_pct",
        "y": "value",
        "curve_value": "value",
        "adjusted_value": "value",
        "mean_value": "value",
    }
    rename = {c: alias[c] for c in out.columns if c in alias and alias[c] not in out.columns}
    if rename:
        out = out.rename(columns=rename)

    missing = sorted(REQUIRED_GAIT_COLS - set(out.columns))
    if missing:
        errors.append("필수 컬럼 누락: " + ", ".join(missing))
        return out, errors

    out["subject_id"] = out["subject_id"].astype(str).str.strip()
    out["feature"] = out["feature"].astype(str).str.strip()
    out["grid_pct"] = pd.to_numeric(out["grid_pct"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["subject_id", "feature", "grid_pct", "value"])
    out = out[(out["subject_id"] != "") & (out["feature"] != "")]
    out = out.sort_values(["subject_id", "feature", "grid_pct"]).reset_index(drop=True)

    if out.empty:
        errors.append("필수 컬럼은 있지만 유효한 subject/feature/grid/value 행이 없습니다.")
    return out, errors


def choose_crf_sheet(file_bytes: bytes, institution: str) -> str:
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheets = xls.sheet_names
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


def read_institution_crf(file_obj, institution: str) -> pd.DataFrame:
    """기관별 CRF를 subject_id/피험자군 중심으로 표준화한다."""
    if file_obj is None:
        return pd.DataFrame()
    b = file_obj.getvalue()
    inst = institution.upper()

    if file_obj.name.lower().endswith(".csv"):
        raw = pd.read_csv(io.BytesIO(b), header=None)
    else:
        sheet = choose_crf_sheet(b, inst)
        raw = pd.read_excel(io.BytesIO(b), sheet_name=sheet, header=None)

    header_idx: Optional[int] = None
    for i in range(min(30, len(raw))):
        vals = [_clean_col_name(v) for v in raw.iloc[i].tolist()]
        joined = " ".join(vals)
        has_group = "피험자군" in joined or re.search(r"\bgroup\b|diagnosis|군|그룹", joined, flags=re.IGNORECASE)
        has_subject = ("피험자번호" in joined) or ("Session Name" in joined) or ("Sub Num" in joined) or re.search(r"subject|\bID\b", joined, flags=re.IGNORECASE)
        if has_group and has_subject:
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0

    cols = _make_unique_columns(raw.iloc[header_idx].tolist())
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = cols
    df = df.dropna(how="all").reset_index(drop=True)
    df["institution"] = inst

    subj_candidates = [
        c for c in df.columns
        if ("피험자번호" in c) or ("Session Name" in c) or ("Sub Num" in c) or re.search(r"subject|\bID\b", c, flags=re.IGNORECASE)
    ]
    subj_col = subj_candidates[0] if subj_candidates else df.columns[0]
    df["subject_id"] = df[subj_col].apply(lambda x: normalize_subject_id_value(x, inst))
    df = df[df["subject_id"].astype(str).str.len() > 0].copy()

    group_candidates = [c for c in df.columns if "피험자군" in c]
    if not group_candidates:
        group_candidates = [c for c in df.columns if re.search(r"group|diagnosis|군|그룹", c, flags=re.IGNORECASE)]
    if group_candidates:
        gcol = group_candidates[0]
        if gcol != "피험자군":
            df["피험자군"] = df[gcol]
    else:
        df["피험자군"] = np.nan

    alias_map = {
        "성별": "sex_auto",
        "만 나이": "age_auto",
        "나이": "age_auto",
        "키": "height_auto",
        "신장": "height_auto",
        "체중": "weight_auto",
        "주손": "dominant_hand_auto",
        "주발": "dominant_foot_auto",
        "호엔": "hy_stage_auto",
        "Hoehn": "hy_stage_auto",
        "HY": "hy_stage_auto",
        "UPDRS": "updrs_auto",
    }
    for pattern, alias in alias_map.items():
        cand = [c for c in df.columns if pattern in c]
        if cand and alias not in df.columns:
            df[alias] = df[cand[0]]

    return df.reset_index(drop=True)


def merge_crf_files(crf_files: Dict[str, object]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for inst, f in crf_files.items():
        if f is None:
            continue
        try:
            frames.append(read_institution_crf(f, inst))
        except Exception as e:
            st.error(f"{inst} CRF 읽기 실패: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # 같은 subject가 여러 번 있으면 첫 행 유지
    out = out.drop_duplicates(subset=["subject_id"], keep="first")
    return out.reset_index(drop=True)


def merge_preprocessed_gait_files(gait_files: Dict[str, object]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """기관별 전처리 gait curve 파일을 읽어 하나의 long table로 합친다.

    이 함수는 원본 MOT/TRC를 읽지 않는다. 1단계 로컬 전처리 결과
    CSV/XLSX/ZIP만 입력으로 받아 subject_id, feature, grid_pct, value 구조를 검증한다.
    """
    frames: List[pd.DataFrame] = []
    diag_rows: List[dict] = []
    errors: List[str] = []

    for inst, f in gait_files.items():
        if f is None:
            continue
        inst = inst.upper()
        try:
            raw = read_preprocessed_gait_file(f)
            standardized, gait_errors = standardize_gait_long(raw)
            if gait_errors:
                errors.extend([f"{inst} gait 데이터 오류: {e}" for e in gait_errors])
                diag_rows.append({
                    "institution": inst,
                    "file_name": getattr(f, "name", ""),
                    "status": "error",
                    "n_rows": int(raw.shape[0]) if isinstance(raw, pd.DataFrame) else 0,
                    "n_subjects": 0,
                    "n_features": 0,
                    "message": "; ".join(gait_errors),
                })
                continue

            # 기관별 업로드이므로 institution을 명시한다. 파일에 institution 컬럼이 있어도 업로드 슬롯 기준을 우선한다.
            standardized["institution"] = inst
            standardized["subject_id"] = standardized["subject_id"].apply(lambda x: normalize_subject_id_value(x, inst))
            standardized["source_institution"] = inst
            standardized["source_upload_file"] = getattr(f, "name", "")
            frames.append(standardized)
            diag_rows.append({
                "institution": inst,
                "file_name": getattr(f, "name", ""),
                "status": "ok",
                "n_rows": int(standardized.shape[0]),
                "n_subjects": int(standardized["subject_id"].nunique()),
                "n_features": int(standardized["feature"].nunique()),
                "grid_min": float(standardized["grid_pct"].min()) if len(standardized) else np.nan,
                "grid_max": float(standardized["grid_pct"].max()) if len(standardized) else np.nan,
                "message": "",
            })
        except Exception as e:
            errors.append(f"{inst} gait 데이터 읽기 실패: {e}")
            diag_rows.append({
                "institution": inst,
                "file_name": getattr(f, "name", ""),
                "status": "error",
                "n_rows": 0,
                "n_subjects": 0,
                "n_features": 0,
                "message": str(e),
            })

    if not frames:
        return pd.DataFrame(), pd.DataFrame(diag_rows), errors or ["기관별 전처리 gait 데이터를 최소 1개 업로드하세요."]

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["subject_id", "feature", "grid_pct"], keep="first")
    out = out.sort_values(["institution", "subject_id", "feature", "grid_pct"]).reset_index(drop=True)
    return out, pd.DataFrame(diag_rows), errors


def dataframe_to_zip_bytes(files: Dict[str, pd.DataFrame], extra_bytes: Optional[Dict[str, bytes]] = None) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, df in files.items():
            if df is None or df.empty:
                continue
            zf.writestr(name, df.to_csv(index=False).encode("utf-8-sig"))
        if extra_bytes:
            for name, b in extra_bytes.items():
                zf.writestr(name, b)
    mem.seek(0)
    return mem.read()


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# =============================================================================
# Curve matrix / adjustment / fPCA
# =============================================================================


def long_to_feature_matrices(long_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    matrices: Dict[str, pd.DataFrame] = {}
    if long_df.empty:
        return matrices
    for feat, sub in long_df.groupby("feature"):
        mat = sub.pivot_table(index="subject_id", columns="grid_pct", values="value", aggfunc="mean")
        mat = mat.sort_index(axis=1)
        matrices[str(feat)] = mat
    return matrices


def make_design_matrix(meta: pd.DataFrame, covariates: Sequence[str], fit_columns: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    if not covariates:
        X = pd.DataFrame(index=meta.index)
        if fit_columns is not None:
            for c in fit_columns:
                X[c] = 0.0
            return X[list(fit_columns)], list(fit_columns)
        return X, []

    parts: List[pd.DataFrame] = []
    for c in covariates:
        if c not in meta.columns:
            continue
        s = meta[c]
        numeric = pd.to_numeric(s, errors="coerce")
        if numeric.notna().mean() >= 0.8:
            med = numeric.median()
            if not np.isfinite(med):
                med = 0.0
            parts.append(pd.DataFrame({c: numeric.fillna(med).astype(float)}, index=meta.index))
        else:
            cat = s.astype("object").where(s.notna(), "Missing").astype(str)
            d = pd.get_dummies(cat, prefix=c, drop_first=True, dtype=float)
            d.index = meta.index
            parts.append(d)

    X = pd.concat(parts, axis=1) if parts else pd.DataFrame(index=meta.index)
    if fit_columns is not None:
        for c in fit_columns:
            if c not in X.columns:
                X[c] = 0.0
        X = X[list(fit_columns)]
        return X, list(fit_columns)
    return X, X.columns.tolist()


def adjust_matrix_all_subjects(mat: pd.DataFrame, meta: pd.DataFrame, subject_col: str, covariates: Sequence[str]) -> pd.DataFrame:
    meta2 = meta.copy()
    meta2[subject_col] = meta2[subject_col].astype(str)
    meta2 = meta2.set_index(subject_col)
    mat2 = mat.copy()
    mat2.index = mat2.index.astype(str)
    common = [sid for sid in mat2.index if sid in meta2.index]
    if not common:
        return pd.DataFrame()
    mat2 = mat2.loc[common]
    meta_sub = meta2.loc[common]
    X, _ = make_design_matrix(meta_sub, covariates)
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
            lm = LinearRegression()
            lm.fit(X.iloc[ok].to_numpy(), y[ok])
            pred = lm.predict(X.to_numpy())
            resid = y - pred
        adjusted[grid] = resid + np.nanmean(y[ok])
    return adjusted


def adjust_all_feature_matrices(matrices: Dict[str, pd.DataFrame], meta: pd.DataFrame, subject_col: str, covariates: Sequence[str]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for feat, mat in matrices.items():
        adj = adjust_matrix_all_subjects(mat, meta, subject_col, covariates)
        if not adj.empty:
            out[feat] = adj
    return out


def matrix_to_long(matrices: Dict[str, pd.DataFrame], value_col: str = "adjusted_value") -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
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
    mat_num = mat.apply(pd.to_numeric, errors="coerce")
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
        score_df[f"FPC{j + 1}"] = scores[:, j]

    grid = mat_num.columns.astype(float).to_numpy()
    loading_rows = []
    for j in range(n_comp):
        for gp, loading in zip(grid, pca.components_[j]):
            loading_rows.append({"component": f"FPC{j + 1}", "grid_pct": float(gp), "loading": float(loading)})
    loading_df = pd.DataFrame(loading_rows)
    evr_df = pd.DataFrame({
        "component": [f"FPC{j + 1}" for j in range(n_comp)],
        "explained_variance_ratio": pca.explained_variance_ratio_,
    })
    return score_df, loading_df, evr_df


def run_fpca_all_features(matrices: Dict[str, pd.DataFrame], n_components: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_rows: List[pd.DataFrame] = []
    loading_rows: List[pd.DataFrame] = []
    evr_rows: List[pd.DataFrame] = []
    wide_parts: List[pd.DataFrame] = []

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

    if not score_rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    scores_long = pd.concat(score_rows, ignore_index=True)
    loadings_long = pd.concat(loading_rows, ignore_index=True)
    evr = pd.concat(evr_rows, ignore_index=True)
    scores_wide = wide_parts[0]
    for part in wide_parts[1:]:
        scores_wide = scores_wide.merge(part, on="subject_id", how="outer")
    return scores_long, loadings_long, evr, scores_wide


# =============================================================================
# Tests
# =============================================================================


def hotelling_t2_2group(X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    y = np.asarray(y)
    X0, X1 = X[y == 0], X[y == 1]
    n0, n1 = X0.shape[0], X1.shape[0]
    p = X.shape[1]
    if n0 <= p or n1 <= p:
        return np.nan, np.nan, np.nan
    m0, m1 = X0.mean(axis=0), X1.mean(axis=0)
    S0, S1 = np.cov(X0, rowvar=False), np.cov(X1, rowvar=False)
    Sp = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)
    diff = m1 - m0
    T2 = (n0 * n1 / (n0 + n1)) * float(diff.T @ np.linalg.pinv(Sp) @ diff)
    F = ((n0 + n1 - p - 1) / ((n0 + n1 - 2) * p)) * T2
    pval = 1 - stats.f.cdf(F, p, n0 + n1 - p - 1)
    return float(T2), float(F), float(pval)


def permanova_2group_euclidean(X: np.ndarray, y: np.ndarray, n_perm: int = 1000, random_state: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    groups = np.unique(y)
    if len(groups) != 2:
        return np.nan, np.nan, np.nan
    n = X.shape[0]
    grand = X.mean(axis=0)

    def stat_for(labels: np.ndarray) -> Tuple[float, float]:
        ss_total = float(((X - grand) ** 2).sum())
        ss_between = 0.0
        for g in groups:
            Xg = X[labels == g]
            if len(Xg) == 0:
                continue
            mg = Xg.mean(axis=0)
            ss_between += Xg.shape[0] * float(((mg - grand) ** 2).sum())
        ss_within = ss_total - ss_between
        F = (ss_between / (len(groups) - 1)) / (ss_within / (n - len(groups))) if ss_within > 0 else np.inf
        R2 = ss_between / ss_total if ss_total > 0 else np.nan
        return float(F), float(R2)

    obs_F, obs_R2 = stat_for(y)
    cnt = 0
    for _ in range(n_perm):
        Fp, _ = stat_for(rng.permutation(y))
        if Fp >= obs_F:
            cnt += 1
    return obs_F, obs_R2, float((cnt + 1) / (n_perm + 1))


def energy_distance_test_2group(X: np.ndarray, y: np.ndarray, n_perm: int = 1000, random_state: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    X0, X1 = X[y == 0], X[y == 1]
    if X0.shape[0] < 2 or X1.shape[0] < 2:
        return np.nan, np.nan

    def energy(a: np.ndarray, b: np.ndarray) -> float:
        return float(2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean())

    obs = energy(X0, X1)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        stat_p = energy(X[yp == 0], X[yp == 1])
        if stat_p >= obs:
            cnt += 1
    return obs, float((cnt + 1) / (n_perm + 1))


def fpca_2d_tests(scores_long: pd.DataFrame, meta: pd.DataFrame, subject_col: str, group_col: str, control_label: str, disease_label: str, n_perm: int) -> pd.DataFrame:
    if scores_long.empty:
        return pd.DataFrame()
    meta2 = meta[[subject_col, group_col]].copy()
    meta2[subject_col] = meta2[subject_col].astype(str)
    rows: List[dict] = []
    for feat, sub in scores_long.groupby("feature"):
        if not {"FPC1", "FPC2"}.issubset(sub.columns):
            continue
        tmp = sub.merge(meta2, left_on="subject_id", right_on=subject_col, how="left")
        tmp = tmp[tmp[group_col].astype(str).isin([str(control_label), str(disease_label)])]
        tmp = tmp.dropna(subset=["FPC1", "FPC2", group_col])
        if tmp[group_col].nunique() != 2 or tmp.shape[0] < 6:
            continue
        y = (tmp[group_col].astype(str) == str(disease_label)).astype(int).to_numpy()
        X = tmp[["FPC1", "FPC2"]].to_numpy(dtype=float)
        t2, fstat, p_hot = hotelling_t2_2group(X, y)
        f_perm, r2_perm, p_perm = permanova_2group_euclidean(X, y, n_perm=n_perm)
        e_stat, p_energy = energy_distance_test_2group(X, y, n_perm=n_perm)
        rows.append({
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
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        for p_col in ["hotelling_p", "permanova_p", "energy_p"]:
            ok = out[p_col].notna()
            q = np.full(out.shape[0], np.nan)
            if ok.sum():
                _, q_vals = fdrcorrection(out.loc[ok, p_col].to_numpy())
                q[np.where(ok)[0]] = q_vals
            out[p_col.replace("_p", "_q_fdr")] = q
        out = out.sort_values(["permanova_q_fdr", "permanova_R2"], ascending=[True, False])
    return out


# =============================================================================
# Plotting
# =============================================================================


def bootstrap_mean_ci(arr: np.ndarray, n_boot: int = 400, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr).mean(axis=1) > 0.8]
    if arr.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])
    mean = np.nanmean(arr, axis=0)
    if arr.shape[0] < 3:
        return mean, mean, mean
    n = arr.shape[0]
    boots = [np.nanmean(arr[rng.integers(0, n, size=n)], axis=0) for _ in range(n_boot)]
    lo, hi = np.nanpercentile(np.vstack(boots), [2.5, 97.5], axis=0)
    return mean, lo, hi


def plot_fda_group_mean(mat: pd.DataFrame, meta: pd.DataFrame, subject_col: str, group_col: str, groups_to_plot: Sequence[str], title: str, stance_pct: float, show_significance: bool = True, alpha: float = 0.05) -> plt.Figure:
    meta2 = meta.copy()
    meta2[subject_col] = meta2[subject_col].astype(str)
    meta2 = meta2.set_index(subject_col)
    mat2 = mat.copy()
    mat2.index = mat2.index.astype(str)
    common = [sid for sid in mat2.index if sid in meta2.index]
    mat2 = mat2.loc[common]
    grid = mat2.columns.astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for g in groups_to_plot:
        sids = [sid for sid in common if str(meta2.loc[sid, group_col]) == str(g)]
        if not sids:
            continue
        arr = mat2.loc[sids].to_numpy(dtype=float)
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
            for col in mat2.columns:
                pvals.append(stats.ttest_ind(mat2.loc[s0, col], mat2.loc[s1, col], nan_policy="omit", equal_var=False).pvalue)
            pvals = np.asarray(pvals, dtype=float)
            ok = np.isfinite(pvals)
            sig = np.zeros_like(pvals, dtype=bool)
            if ok.sum():
                _, q = fdrcorrection(pvals[ok])
                sig[np.where(ok)[0]] = q < alpha
            if sig.any():
                ymin, ymax = ax.get_ylim()
                ax.scatter(grid[sig], np.full(sig.sum(), ymin + 0.04 * (ymax - ymin)), s=10, marker="|", label=f"FDR q<{alpha}")
    ax.axvline(stance_pct, linestyle="--", linewidth=1, alpha=0.55)
    ax.text(stance_pct, ax.get_ylim()[1], " stance/swing", va="top", fontsize=9)
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Adjusted curve value")
    ax.set_title(title)
    ax.legend(fontsize=9)
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


def plot_fpca_scatter(scores_long: pd.DataFrame, meta: pd.DataFrame, subject_col: str, group_col: str, feature: str, groups_to_plot: Sequence[str], title: str) -> plt.Figure:
    sub = scores_long[scores_long["feature"] == feature].copy()
    meta2 = meta[[subject_col, group_col]].copy()
    meta2[subject_col] = meta2[subject_col].astype(str)
    tmp = sub.merge(meta2, left_on="subject_id", right_on=subject_col, how="left")
    tmp = tmp[tmp[group_col].astype(str).isin([str(g) for g in groups_to_plot])]
    fig, ax = plt.subplots(figsize=(7.2, 5.7))
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


def plot_loading(loadings_long: pd.DataFrame, feature: str, stance_pct: float) -> plt.Figure:
    sub = loadings_long[loadings_long["feature"] == feature]
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    for comp, ss in sub.groupby("component"):
        ss = ss.sort_values("grid_pct")
        ax.plot(ss["grid_pct"], ss["loading"], label=comp)
    ax.axhline(0, linewidth=0.8, alpha=0.5)
    ax.axvline(stance_pct, linestyle="--", linewidth=1, alpha=0.55)
    ax.set_xlabel("Gait cycle (%)")
    ax.set_ylabel("Loading")
    ax.set_title(f"fPCA component loading: {feature}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_fpca_box(
    scores_long: pd.DataFrame,
    meta: pd.DataFrame,
    subject_col: str,
    group_col: str,
    feature: str,
    pc: str,
    groups: Sequence[str],
) -> plt.Figure:
    sub = scores_long[scores_long["feature"].astype(str) == str(feature)].copy()

    meta2 = meta[[subject_col, group_col]].copy()
    meta2[subject_col] = meta2[subject_col].astype(str).str.strip()
    meta2[group_col] = meta2[group_col].astype(str).str.strip()

    sub["subject_id"] = sub["subject_id"].astype(str).str.strip()

    tmp = sub.merge(
        meta2,
        left_on="subject_id",
        right_on=subject_col,
        how="left",
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.7))

    if pc not in tmp.columns:
        ax.text(
            0.5,
            0.5,
            f"{pc} 컬럼이 없습니다.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    plot_data = []
    plot_labels = []

    for g in groups:
        vals = pd.to_numeric(
            tmp.loc[tmp[group_col].astype(str) == str(g), pc],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)

        vals = vals[np.isfinite(vals)]

        # 값이 있는 그룹만 boxplot에 포함
        if len(vals) > 0:
            plot_data.append(vals)
            plot_labels.append(str(g))

    if len(plot_data) == 0:
        ax.text(
            0.5,
            0.5,
            "선택한 feature/PC에서 표시 가능한 그룹 데이터가 없습니다.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        return fig

    # matplotlib 3.9+에서는 labels 대신 tick_labels 권장
    try:
        ax.boxplot(plot_data, tick_labels=plot_labels, showfliers=False)
    except TypeError:
        # 구버전 matplotlib 호환
        ax.boxplot(plot_data, labels=plot_labels, showfliers=False)

    rng = np.random.default_rng(42)
    for i, vals in enumerate(plot_data, start=1):
        jitter = rng.normal(i, 0.035, size=len(vals))
        ax.scatter(jitter, vals, alpha=0.7, s=35)

    ax.set_title(f"{feature} - {pc} 그룹별 분포")
    ax.set_ylabel(f"{pc} score")
    ax.grid(axis="y", alpha=0.25)

    # 그룹 중 하나가 비어 있었으면 그림 안에 알림
    missing_groups = [str(g) for g in groups if str(g) not in plot_labels]
    if missing_groups:
        ax.text(
            0.01,
            0.01,
            "데이터 없음: " + ", ".join(missing_groups),
            transform=ax.transAxes,
            fontsize=9,
            va="bottom",
            ha="left",
            alpha=0.75,
        )

    fig.tight_layout()
    return fig


# =============================================================================
# Clinical & ML
# =============================================================================


def fpca_clinical_correlation(scores_wide: pd.DataFrame, meta: pd.DataFrame, subject_col: str, disease_mask: pd.Series, clinical_vars: Sequence[str]) -> pd.DataFrame:
    if scores_wide.empty or not clinical_vars:
        return pd.DataFrame()
    meta2 = meta.copy()
    meta2[subject_col] = meta2[subject_col].astype(str)
    meta2 = meta2.loc[disease_mask].copy()
    dat = scores_wide.merge(meta2, left_on="subject_id", right_on=subject_col, how="inner")
    score_cols = [c for c in scores_wide.columns if c != "subject_id"]
    rows: List[dict] = []
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
            rows.append({"clinical_variable": cv, "fpca_feature": sc, "n": int(ok.sum()), "spearman_rho": float(rho), "p_value": float(p)})
    out = pd.DataFrame(rows)
    if not out.empty:
        _, q = fdrcorrection(out["p_value"].fillna(1).to_numpy())
        out["q_value_fdr"] = q
        out = out.sort_values(["q_value_fdr", "p_value"])
    return out


def transform_feature_train_test_no_leakage(mat: pd.DataFrame, meta: pd.DataFrame, subject_col: str, covariates: Sequence[str], train_subjects: Sequence[str], test_subjects: Sequence[str], n_components: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    mat2 = mat.copy()
    mat2.index = mat2.index.astype(str)
    train_subjects = [str(s) for s in train_subjects if str(s) in mat2.index]
    test_subjects = [str(s) for s in test_subjects if str(s) in mat2.index]
    if len(train_subjects) < 4 or len(test_subjects) == 0:
        return pd.DataFrame(index=train_subjects), pd.DataFrame(index=test_subjects), []

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

    imp = SimpleImputer(strategy="mean")
    Xtr = imp.fit_transform(train_adj)
    Xte = imp.transform(test_adj)
    max_comp = min(n_components, Xtr.shape[0] - 1, Xtr.shape[1])
    if max_comp < 1:
        return pd.DataFrame(index=train_subjects), pd.DataFrame(index=test_subjects), []
    pca = PCA(n_components=max_comp, random_state=42)
    Ztr = pca.fit_transform(Xtr)
    Zte = pca.transform(Xte)
    cols = [f"FPC{i + 1}" for i in range(max_comp)]
    return pd.DataFrame(Ztr, index=train_subjects, columns=cols), pd.DataFrame(Zte, index=test_subjects, columns=cols), cols


def compute_binary_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    ok = np.isfinite(prob) & np.isfinite(y_true)
    y_true = y_true[ok].astype(int)
    prob = prob[ok]
    pred = (prob >= threshold).astype(int)
    auc_val = roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    return {"AUC": float(auc_val), "Accuracy": float(accuracy_score(y_true, pred)), "Sensitivity": float(sens), "Specificity": float(spec), "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


def run_no_leakage_ml(raw_matrices: Dict[str, pd.DataFrame], meta: pd.DataFrame, subject_col: str, group_col: str, control_label: str, disease_label: str, covariates: Sequence[str], selected_features: Sequence[str], n_components: int, n_splits: int, c_value: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta2 = meta.copy()
    meta2[subject_col] = meta2[subject_col].astype(str)
    meta2 = meta2[meta2[group_col].astype(str).isin([str(control_label), str(disease_label)])].copy()
    subjects = meta2[subject_col].astype(str).to_list()
    y = (meta2[group_col].astype(str) == str(disease_label)).astype(int).to_numpy()
    _, counts = np.unique(y, return_counts=True)
    max_splits = int(counts.min()) if len(counts) == 2 else 0
    if max_splits < 2:
        raise ValueError("두 그룹 중 한쪽 표본 수가 2명 미만이라 교차검증을 할 수 없습니다.")
    n_splits = max(2, min(int(n_splits), max_splits))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_prob = np.full(len(subjects), np.nan, dtype=float)
    fold_rows: List[dict] = []
    coef_rows: List[dict] = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(subjects, y), start=1):
        train_subjects = [subjects[i] for i in tr_idx]
        test_subjects = [subjects[i] for i in te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]
        train_parts: List[pd.DataFrame] = []
        test_parts: List[pd.DataFrame] = []
        for feat in selected_features:
            if feat not in raw_matrices:
                continue
            ztr, zte, cols = transform_feature_train_test_no_leakage(raw_matrices[feat], meta2, subject_col, covariates, train_subjects, test_subjects, n_components)
            if not cols:
                continue
            ztr = ztr.reindex(train_subjects)
            zte = zte.reindex(test_subjects)
            ztr.columns = [f"{feat}__{c}" for c in ztr.columns]
            zte.columns = [f"{feat}__{c}" for c in zte.columns]
            train_parts.append(ztr)
            test_parts.append(zte)
        if not train_parts:
            raise ValueError("ML에 사용할 fPCA feature를 만들 수 없습니다.")
        X_train = pd.concat(train_parts, axis=1)
        X_test = pd.concat(test_parts, axis=1)
        feature_names = X_train.columns.tolist()
        imp = SimpleImputer(strategy="mean")
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(imp.fit_transform(X_train))
        Xte = scaler.transform(imp.transform(X_test))
        clf = LogisticRegression(C=c_value, penalty="l2", solver="liblinear", class_weight="balanced", random_state=random_state)
        clf.fit(Xtr, y_train)
        prob = clf.predict_proba(Xte)[:, 1]
        oof_prob[te_idx] = prob
        metrics = compute_binary_metrics(y_test, prob)
        metrics.update({"fold": fold, "n_train": len(tr_idx), "n_test": len(te_idx)})
        fold_rows.append(metrics)
        for name, coef in zip(feature_names, clf.coef_.ravel()):
            coef_rows.append({"fold": fold, "feature": name, "coef_standardized": float(coef), "abs_coef_standardized": float(abs(coef))})

    oof_df = pd.DataFrame({"subject_id": subjects, "y_true": y, "oof_probability": oof_prob})
    oof_metrics = pd.DataFrame([compute_binary_metrics(y, oof_prob)])
    fold_df = pd.DataFrame(fold_rows)
    coef_df = pd.DataFrame(coef_rows)
    if not coef_df.empty:
        coef_summary = coef_df.groupby("feature", as_index=False).agg(
            coef_mean=("coef_standardized", "mean"),
            coef_sd=("coef_standardized", "std"),
            abs_coef_mean=("abs_coef_standardized", "mean"),
            selected_folds=("coef_standardized", "count"),
        ).sort_values("abs_coef_mean", ascending=False)
    else:
        coef_summary = pd.DataFrame()
    return fold_df, oof_metrics, oof_df, coef_summary


# =============================================================================
# Main UI
# =============================================================================

st.title("🚶 OpenCap Gait 분석 웹서비스")
st.caption("2~5번 분석 전용 버전: 1번 원본 MOT/TRC 전처리는 로컬에서 끝내고, 웹에는 전처리된 gait curve + 기관별 CRF만 업로드합니다.")

# session init
for key, default in {
    "loaded": False,
    "analyzed": False,
    "analysis_zip": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("입력 자료: 1단계 전처리 산출물 + CRF")
    st.caption("이 앱은 2~5번 분석 전용입니다. 원본 MOT/TRC는 업로드하지 않습니다.")

    gait_files = {}
    crf_files = {}
    for inst in INSTITUTIONS:
        with st.expander(f"{inst} 업로드", expanded=(inst in ["UNI", "UUH"])):
            gait_files[inst] = st.file_uploader(
                f"{inst} 전처리 gait curve CSV/XLSX/ZIP",
                type=["csv", "xlsx", "xls", "zip"],
                key=f"gait_{inst}",
                help="필수 컬럼: subject_id, feature, grid_pct, value",
            )
            crf_files[inst] = st.file_uploader(
                f"{inst} CRF",
                type=["xlsx", "xls", "csv"],
                key=f"crf_{inst}",
            )

    st.caption("전처리 gait 파일은 환자별 mean gait curve long table이어야 합니다: subject_id, feature, grid_pct, value")

    st.divider()
    st.header("2) 분석 파라미터")
    n_components = st.slider("fPC 개수", 1, 5, 2)
    n_perm = st.slider("2D 분포 검정 permutation", 100, 5000, 1000, step=100)
    stance_pct = st.slider("Stance/Swing 기준점 (%)", 40, 80, 60)
    alpha = st.number_input("유의수준 α", min_value=0.001, max_value=0.2, value=0.05, step=0.01)
    n_splits = st.slider("ML K-fold", 2, 10, 5)
    c_value = st.number_input("Logistic C", min_value=0.001, max_value=100.0, value=1.0)
    random_state = st.number_input("Random seed", value=42, step=1)

    st.divider()
    load_clicked = st.button("① 데이터 로드/매핑", type="primary", use_container_width=True)

if load_clicked:
    try:
        gait_long, gait_upload_diag, gait_errors = merge_preprocessed_gait_files(gait_files)
        crf = merge_crf_files(crf_files)
        if gait_errors:
            st.session_state["loaded"] = False
            st.session_state["gait_upload_diag"] = gait_upload_diag
            for e in gait_errors:
                st.error(e)
        elif crf.empty:
            st.session_state["loaded"] = False
            st.error("CRF가 비어 있습니다. 최소 1개 기관의 CRF를 업로드하세요.")
        else:
            st.session_state["gait_long"] = gait_long
            st.session_state["gait_upload_diag"] = gait_upload_diag
            st.session_state["crf"] = crf
            st.session_state["loaded"] = True
            st.session_state["analyzed"] = False
            st.session_state["analysis_zip"] = None
            st.success("기관별 전처리 gait 데이터와 CRF 로드/매핑 준비 완료")
    except Exception as e:
        st.session_state["loaded"] = False
        st.error(f"데이터 로드 실패: {e}")

if not st.session_state.get("loaded"):
    st.info("기관별 전처리 gait curve 데이터와 기관별 CRF를 업로드한 뒤, 사이드바의 `① 데이터 로드/매핑`을 눌러주세요.")
    with st.expander("전처리 gait 데이터 형식 예시", expanded=True):
        example = pd.DataFrame({
            "subject_id": ["UNI1", "UNI1", "UNI1", "UNI2", "UNI2"],
            "feature": ["hip_flexion_r", "hip_flexion_r", "hip_flexion_r", "hip_flexion_r", "hip_flexion_r"],
            "grid_pct": [0, 1, 2, 0, 1],
            "value": [10.2, 10.5, 11.0, 8.9, 9.1],
            "n_trials_kept": [3, 3, 3, 2, 2],
        })
        st.dataframe(example, use_container_width=True)
    st.stop()

# loaded data
raw_long: pd.DataFrame = st.session_state["gait_long"]
crf: pd.DataFrame = st.session_state["crf"]

# Mapping diagnostics
subject_col_default = "subject_id" if "subject_id" in crf.columns else crf.columns[0]
with st.sidebar:
    st.header("3) 컬럼/그룹 설정")
    subject_col = st.selectbox("CRF subject ID 컬럼", crf.columns.tolist(), index=crf.columns.tolist().index(subject_col_default))
    group_default = "피험자군" if "피험자군" in crf.columns else crf.columns.tolist()[0]
    group_col = st.selectbox("그룹 컬럼", crf.columns.tolist(), index=crf.columns.tolist().index(group_default))

    crf[group_col] = crf[group_col].astype(str)
    group_values = sorted([g for g in crf[group_col].dropna().astype(str).unique().tolist() if g and g != "nan"])
    if len(group_values) >= 2:
        control_guess = "Control" if "Control" in group_values else group_values[0]
        disease_guess = "Parkinson" if "Parkinson" in group_values else group_values[-1]
    else:
        control_guess = group_values[0] if group_values else "Control"
        disease_guess = group_values[-1] if group_values else "Parkinson"
    control_label = st.selectbox("정상군 label", group_values, index=group_values.index(control_guess) if control_guess in group_values else 0)
    disease_label = st.selectbox("질환군 label", group_values, index=group_values.index(disease_guess) if disease_guess in group_values else min(1, len(group_values)-1))

    candidate_covs = [c for c in ["age_auto", "sex_auto", "height_auto", "weight_auto", "dominant_hand_auto", "dominant_foot_auto"] if c in crf.columns]
    covariates = st.multiselect("공변량 보정 변수", crf.columns.tolist(), default=candidate_covs)

    all_features = sorted(raw_long["feature"].dropna().astype(str).unique().tolist())
    default_features = all_features[: min(12, len(all_features))]
    selected_features = st.multiselect("분석 feature", all_features, default=default_features)

    analyze_clicked = st.button("② 분석 시작", type="primary", use_container_width=True)

# Standardize subject col and merge
crf_work = crf.copy()
crf_work[subject_col] = crf_work[subject_col].astype(str).str.strip()
raw_long["subject_id"] = raw_long["subject_id"].astype(str).str.strip()
matched_subjects = sorted(set(raw_long["subject_id"]) & set(crf_work[subject_col].astype(str)))

# Header metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Gait subjects", raw_long["subject_id"].nunique())
c2.metric("CRF subjects", crf_work[subject_col].nunique())
c3.metric("Matched subjects", len(matched_subjects))
c4.metric("Features", raw_long["feature"].nunique())

if len(matched_subjects) == 0:
    st.error("gait 데이터 subject_id와 CRF subject ID가 매칭되지 않습니다.")
    st.stop()

if analyze_clicked:
    if not selected_features:
        st.error("분석할 feature를 하나 이상 선택하세요.")
    else:
        with st.spinner("FDA/fPCA 분석 중..."):
            long_sel = raw_long[raw_long["feature"].isin(selected_features)].copy()
            # matched subjects only
            long_sel = long_sel[long_sel["subject_id"].isin(matched_subjects)].copy()
            matrices_raw = long_to_feature_matrices(long_sel)
            adjusted = adjust_all_feature_matrices(matrices_raw, crf_work, subject_col, covariates)
            scores_long, loadings_long, evr, scores_wide = run_fpca_all_features(adjusted, n_components=n_components)
            tests_2d = fpca_2d_tests(scores_long, crf_work, subject_col, group_col, control_label, disease_label, n_perm=n_perm)
            adjusted_long = matrix_to_long(adjusted)
            st.session_state["matrices_raw"] = matrices_raw
            st.session_state["adjusted_matrices"] = adjusted
            st.session_state["adjusted_long"] = adjusted_long
            st.session_state["scores_long"] = scores_long
            st.session_state["scores_wide"] = scores_wide
            st.session_state["loadings_long"] = loadings_long
            st.session_state["evr"] = evr
            st.session_state["tests_2d"] = tests_2d
            st.session_state["analysis_config"] = pd.DataFrame([
                {"n_components": n_components, "n_perm": n_perm, "stance_pct": stance_pct, "alpha": alpha, "control_label": control_label, "disease_label": disease_label, "covariates": ";".join(covariates), "features": ";".join(selected_features)}
            ])
            st.session_state["analyzed"] = True
            st.session_state["analysis_zip"] = None
        st.success("분석 완료")

# tabs
tab_data, tab_fda, tab_fpca, tab_clinical, tab_ml, tab_download = st.tabs(["데이터/매핑", "FDA", "fPCA", "임상척도", "ML", "다운로드"])

with tab_data:
    st.subheader("데이터 구조 확인")
    gait_upload_diag = st.session_state.get("gait_upload_diag", pd.DataFrame())
    if not gait_upload_diag.empty:
        st.write("기관별 전처리 gait 업로드 진단")
        st.dataframe(gait_upload_diag, use_container_width=True)
    st.write("전처리 gait 데이터")
    st.dataframe(raw_long.head(200), use_container_width=True)
    st.write("CRF")
    st.dataframe(crf_work.head(100), use_container_width=True)
    map_df = pd.DataFrame({"subject_id": sorted(set(raw_long["subject_id"]) | set(crf_work[subject_col].astype(str)))})
    map_df["in_gait"] = map_df["subject_id"].isin(set(raw_long["subject_id"]))
    map_df["in_crf"] = map_df["subject_id"].isin(set(crf_work[subject_col].astype(str)))
    map_df["matched"] = map_df["in_gait"] & map_df["in_crf"]
    st.write("Subject 매핑 진단")
    st.dataframe(map_df, use_container_width=True)
    group_summary = crf_work[crf_work[subject_col].astype(str).isin(matched_subjects)].groupby(group_col).size().reset_index(name="n")
    st.write("매칭 subject 기준 그룹 분포")
    st.dataframe(group_summary, use_container_width=True)

if not st.session_state.get("analyzed"):
    st.warning("사이드바에서 파라미터를 정한 뒤 `② 분석 시작`을 눌러주세요.")
    st.stop()

matrices_raw: Dict[str, pd.DataFrame] = st.session_state["matrices_raw"]
adjusted: Dict[str, pd.DataFrame] = st.session_state["adjusted_matrices"]
adjusted_long: pd.DataFrame = st.session_state["adjusted_long"]
scores_long: pd.DataFrame = st.session_state["scores_long"]
scores_wide: pd.DataFrame = st.session_state["scores_wide"]
loadings_long: pd.DataFrame = st.session_state["loadings_long"]
evr: pd.DataFrame = st.session_state["evr"]
tests_2d: pd.DataFrame = st.session_state["tests_2d"]

with tab_fda:
    st.subheader("FDA mean curve")
    if not adjusted:
        st.error("보정 curve가 없습니다.")
    else:
        feat = st.selectbox("FDA feature 선택", sorted(adjusted.keys()), key="fda_feat")
        fig = plot_fda_group_mean(adjusted[feat], crf_work, subject_col, group_col, [control_label, disease_label], f"{feat}: {control_label} vs {disease_label}", stance_pct=stance_pct, show_significance=True, alpha=alpha)
        st.pyplot(fig)

with tab_fpca:
    st.subheader("fPCA 결과")
    if tests_2d.empty:
        st.info("2D 분포 검정 결과가 없습니다. 표본 수 또는 FPC 개수를 확인하세요.")
    else:
        st.write("FPC1-FPC2 그룹 분포 차이 검정")
        st.dataframe(tests_2d, use_container_width=True)
    if not scores_long.empty:
        feat_options = sorted(scores_long["feature"].unique().tolist())
        feat2 = st.selectbox("fPCA feature 선택", feat_options, key="fpca_feat")
        col1, col2 = st.columns(2)
        with col1:
            fig = plot_fpca_scatter(scores_long, crf_work, subject_col, group_col, feat2, [control_label, disease_label], f"{feat2}: FPC1-FPC2")
            st.pyplot(fig)
        with col2:
            fig = plot_loading(loadings_long, feat2, stance_pct=stance_pct)
            st.pyplot(fig)
        pc_cols = [c for c in scores_long.columns if c.startswith("FPC")]
        if pc_cols:
            pc = st.selectbox("Boxplot PC", pc_cols)
            fig = plot_fpca_box(scores_long, crf_work, subject_col, group_col, feat2, pc, [control_label, disease_label])
            st.pyplot(fig)
    if not evr.empty:
        st.write("설명분산")
        st.dataframe(evr, use_container_width=True)

with tab_clinical:
    st.subheader("질환군 내 임상척도 연결")
    disease_mask = crf_work[group_col].astype(str).eq(str(disease_label))
    numeric_candidates = []
    for c in crf_work.columns:
        if c in [subject_col, group_col, "institution"]:
            continue
        num = pd.to_numeric(crf_work[c], errors="coerce")
        if num.notna().sum() >= 5:
            numeric_candidates.append(c)
    default_clin = [c for c in ["updrs_auto", "hy_stage_auto"] if c in numeric_candidates]
    clinical_vars = st.multiselect("상관분석 임상척도", numeric_candidates, default=default_clin)
    if st.button("임상척도 상관분석 실행"):
        clin_corr = fpca_clinical_correlation(scores_wide, crf_work, subject_col, disease_mask, clinical_vars)
        st.session_state["clin_corr"] = clin_corr
    clin_corr = st.session_state.get("clin_corr", pd.DataFrame())
    if not clin_corr.empty:
        st.dataframe(clin_corr, use_container_width=True)
    else:
        st.info("질환군 내에서 UPDRS/HY 등 임상척도를 선택하고 실행하세요.")

with tab_ml:
    st.subheader("fPCA 기반 leakage-free 로지스틱 회귀")
    st.caption("각 fold 내부에서 공변량 보정과 PCA를 fit하고 test fold는 transform만 수행합니다.")
    ml_features = st.multiselect("ML feature", sorted(matrices_raw.keys()), default=sorted(matrices_raw.keys())[: min(20, len(matrices_raw))])
    if st.button("ML 실행"):
        try:
            with st.spinner("ML 교차검증 실행 중..."):
                fold_df, oof_metrics, oof_df, coef_df = run_no_leakage_ml(matrices_raw, crf_work, subject_col, group_col, control_label, disease_label, covariates, ml_features, n_components, n_splits, c_value, int(random_state))
                st.session_state["ml_fold"] = fold_df
                st.session_state["ml_oof_metrics"] = oof_metrics
                st.session_state["ml_oof"] = oof_df
                st.session_state["ml_coef"] = coef_df
            st.success("ML 완료")
        except Exception as e:
            st.error(f"ML 실패: {e}")
    if "ml_oof_metrics" in st.session_state:
        st.write("OOF 성능")
        st.dataframe(st.session_state["ml_oof_metrics"], use_container_width=True)
        st.write("Fold별 성능")
        st.dataframe(st.session_state["ml_fold"], use_container_width=True)
        oof = st.session_state["ml_oof"]
        ok = oof["oof_probability"].notna()
        if ok.sum() and oof.loc[ok, "y_true"].nunique() == 2:
            fpr, tpr, _ = roc_curve(oof.loc[ok, "y_true"], oof.loc[ok, "oof_probability"])
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(6.5, 5.2))
            ax.plot(fpr, tpr, label=f"OOF AUC={roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.6)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("OOF ROC")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
        st.write("계수 요약")
        st.dataframe(st.session_state["ml_coef"], use_container_width=True)

with tab_download:
    st.subheader("전체 분석 자료 다운로드")
    include_figures = st.checkbox("주요 PNG 그래프 포함", value=True)
    if st.button("전체 분석 ZIP 생성/갱신"):
        files = {
            "00_input/preprocessed_gait_long.csv": raw_long,
            "00_input/gait_upload_diagnostics.csv": st.session_state.get("gait_upload_diag", pd.DataFrame()),
            "00_input/crf_standardized.csv": crf_work,
            "00_input/subject_mapping.csv": map_df,
            "00_input/group_summary.csv": group_summary,
            "01_analysis/adjusted_curves_long.csv": adjusted_long,
            "02_fpca/fpca_scores_long.csv": scores_long,
            "02_fpca/fpca_scores_wide.csv": scores_wide,
            "02_fpca/fpca_loadings_long.csv": loadings_long,
            "02_fpca/fpca_explained_variance.csv": evr,
            "02_fpca/fpca_2d_group_tests.csv": tests_2d,
            "99_config/analysis_config.csv": st.session_state.get("analysis_config", pd.DataFrame()),
        }
        if "clin_corr" in st.session_state:
            files["03_clinical/fpca_clinical_correlations.csv"] = st.session_state["clin_corr"]
        if "ml_fold" in st.session_state:
            files["04_ml/ml_fold_metrics.csv"] = st.session_state["ml_fold"]
            files["04_ml/ml_oof_metrics.csv"] = st.session_state["ml_oof_metrics"]
            files["04_ml/ml_oof_predictions.csv"] = st.session_state["ml_oof"]
            files["04_ml/ml_logistic_coefficients.csv"] = st.session_state["ml_coef"]
        extra = {}
        if include_figures and adjusted:
            top_feats = sorted(list(adjusted.keys()))[: min(10, len(adjusted))]
            for feat in top_feats:
                try:
                    fig = plot_fda_group_mean(adjusted[feat], crf_work, subject_col, group_col, [control_label, disease_label], f"{feat}: FDA", stance_pct=stance_pct, show_significance=True, alpha=alpha)
                    extra[f"figures/fda_{safe_filename(feat)}.png"] = fig_to_png_bytes(fig)
                except Exception:
                    pass
                try:
                    fig = plot_loading(loadings_long, feat, stance_pct=stance_pct)
                    extra[f"figures/loading_{safe_filename(feat)}.png"] = fig_to_png_bytes(fig)
                except Exception:
                    pass
                try:
                    fig = plot_fpca_scatter(scores_long, crf_work, subject_col, group_col, feat, [control_label, disease_label], f"{feat}: FPC1-FPC2")
                    extra[f"figures/fpca_scatter_{safe_filename(feat)}.png"] = fig_to_png_bytes(fig)
                except Exception:
                    pass
        st.session_state["analysis_zip"] = dataframe_to_zip_bytes(files, extra)
        st.success("ZIP 생성 완료")

    if st.session_state.get("analysis_zip"):
        st.download_button(
            "전체 분석 자료 ZIP 다운로드",
            data=st.session_state["analysis_zip"],
            file_name="opencap_gait_analysis_results.zip",
            mime="application/zip",
            use_container_width=True,
        )

st.divider()
st.caption("주의: 이 앱은 원본 MOT/TRC 전처리를 수행하지 않습니다. 입력 gait 데이터는 환자별 mean curve가 생성된 전처리 결과여야 합니다.")
