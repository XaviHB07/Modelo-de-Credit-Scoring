from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config
from .utils import find_col, is_text, log, safe_name, top_levels

KEY = "key_value"
DATE_HINT = re.compile(r"(fecha|fec_|date|_dt$)")
COUNT_HINT = re.compile(r"^(ctd|cant|cnt|n|num|nro|numero|meses|anios|pred|flag)(_|$)")
MIN_LEVEL_COUNT = 50
PIVOT_MAX_LEVELS = 30
PIVOT_TOP = 6
MISSING_IND_RANGE = (0.02, 0.95)
MAX_MISSING = 0.95


def _split_types(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    dates, cats, nums = [], [], []
    for c in df.columns:
        s = df[c]
        if is_text(s) and DATE_HINT.search(c):
            dates.append(c)
        elif is_text(s):
            cats.append(c)
        elif pd.api.types.is_integer_dtype(s) and not COUNT_HINT.search(c):
            cats.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            nums.append(c)
    return dates, cats, nums


def _dates_to_months(df: pd.DataFrame, dates: list[str], ref_month: int) -> pd.DataFrame:
    out = df.copy()
    ref_idx = (ref_month // 100) * 12 + (ref_month % 100) - 1
    for c in dates:
        dt = pd.to_datetime(out[c], errors="coerce")
        out[c] = ref_idx - (dt.dt.year * 12 + dt.dt.month - 1)
    return out


def _mode_per_group(df: pd.DataFrame, key: str, col: str) -> pd.Series:
    cnt = df.dropna(subset=[col]).groupby([key, col], sort=False).size().rename("n").reset_index()
    cnt = cnt.sort_values([key, "n"], ascending=[True, False]).drop_duplicates(key)
    return cnt.set_index(key)[col]


def collapse_per_client(df: pd.DataFrame, key: str, state: dict, fit: bool,
                        pivot_numeric_by: tuple[str, ...] = ()) -> pd.DataFrame:
    if not df[key].duplicated().any():
        return df.set_index(key)

    cats, nums = state["cats"], state["nums"] + state["dates"]
    n_raw = df.groupby(key, sort=False).size().rename("n_registros")
    df = df.drop_duplicates()
    g = df.groupby(key, sort=False)
    parts = [n_raw.to_frame(), g.size().rename("n_registros_unicos").to_frame()]
    for c in nums:
        agg = g[c].agg(["mean", "min", "max", "std"])
        agg.columns = [f"{c}_{a}" for a in agg.columns]
        parts.append(agg)
        if df[c].isna().mean() > 0.05:
            parts.append(g[c].count().rename(f"{c}_n_con_dato").to_frame())
    for c in cats:
        parts.append(g[c].nunique().rename(f"{c}_nunique").to_frame())
        parts.append(_mode_per_group(df, key, c).rename(f"{c}_moda").to_frame())
    if fit:
        state["pivot_levels"] = {c: top_levels(df[c], PIVOT_TOP) for c in cats if df[c].nunique() <= PIVOT_MAX_LEVELS}
    for c, levels in state["pivot_levels"].items():
        sub = df[df[c].isin(levels)].assign(_one=1)
        cnt = sub.pivot_table(index=key, columns=c, values="_one", aggfunc="sum", fill_value=0)
        cnt = cnt.reindex(columns=levels, fill_value=0)
        cnt.columns = [f"{c}_n_{safe_name(l)}" for l in cnt.columns]
        parts.append(cnt)
    for c in pivot_numeric_by:
        if c not in cats:
            continue
        if fit:
            state.setdefault("pivot_num_levels", {})[c] = top_levels(df[c], config.TOP_K)
        levels = state["pivot_num_levels"][c]
        sub = df[df[c].isin(levels)]
        for n in nums[:1]:
            pv = sub.pivot_table(index=key, columns=c, values=n, aggfunc="mean")
            pv = pv.reindex(columns=levels)
            pv.columns = [f"{n}_{safe_name(c)}_{safe_name(l)}" for l in pv.columns]
            parts.append(pv)
    return pd.concat(parts, axis=1)


def build_static_features(df: pd.DataFrame, name: str, ref_month: int, state: dict | None = None,
                          corr_threshold: float | None = None, pca_components: int = 0,
                          pivot_numeric_by: tuple[str, ...] = ()) -> tuple[pd.DataFrame, dict]:
    fit = state is None
    key = find_col(df, "key", name, required=True)
    df = df.copy()
    if fit:
        d_, c_, n_ = _split_types(df.drop(columns=[key]))
        state = {"dates": d_, "cats": c_, "nums": n_}
        log.info("%s: %d numéricas, %d categóricas, %d fechas", name, len(n_), len(c_), len(d_))
    df = _dates_to_months(df, state["dates"], ref_month)
    for c in state["cats"]:
        df[c] = df[c].astype("string")

    X = collapse_per_client(df, key, state, fit, pivot_numeric_by)
    X.index.name = KEY
    X = X.reset_index()
    if X[KEY].duplicated().any():
        raise ValueError(f"{name}: la llave quedó duplicada tras agrupar")

    cat_cols = [c for c in X.columns if c != KEY and is_text(X[c])]
    num_cols = [c for c in X.columns if c != KEY and c not in cat_cols]

    if fit:
        state["cat_cols"] = cat_cols
        state["cat_levels"] = {}
        min_count = max(MIN_LEVEL_COUNT, int(0.0002 * len(X)))
        for c in cat_cols:
            freq = X[c].value_counts()
            state["cat_levels"][c] = freq[freq >= min_count].index.astype(str).tolist()
        miss = X[num_cols].isna().mean()
        nun = X[num_cols].nunique(dropna=True)
        keep = [c for c in num_cols if nun[c] > 1 and miss[c] <= MAX_MISSING]
        dropped = sorted(set(num_cols) - set(keep))
        if dropped:
            log.info("%s: se eliminan %d numéricas constantes o casi vacías", name, len(dropped))
        state["lo"] = X[keep].quantile(0.005).to_dict()
        state["hi"] = X[keep].quantile(0.995).to_dict()
        state["miss_ind"] = [c for c in keep if MISSING_IND_RANGE[0] <= miss[c] <= MISSING_IND_RANGE[1]]
        state["num_cols"] = keep

    for c in state["cat_cols"]:
        s = X[c].astype("object")
        s = s.where(s.isna(), s.astype(str))
        s = s.where(s.isin(state["cat_levels"][c]) | s.isna(), "otros").fillna("missing")
        X[c] = s.astype(str)

    for c in state["num_cols"]:
        X[c] = X[c].astype(float).clip(state["lo"][c], state["hi"][c])
    if state["miss_ind"]:
        ind = pd.DataFrame({f"{c}_ausente": X[c].isna().astype("int8") for c in state["miss_ind"]}, index=X.index)
        X = pd.concat([X, ind], axis=1)

    if fit and corr_threshold:
        filled = X[state["num_cols"]].fillna(X[state["num_cols"]].median())
        corr = filled.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        drop = [c for c in upper.columns if (upper[c] > corr_threshold).any()]
        state["corr_drop"] = drop
        log.info("%s: se eliminan %d numéricas con |corr| > %.2f", name, len(drop), corr_threshold)
    state.setdefault("corr_drop", [])
    X = X.drop(columns=[c for c in state["corr_drop"] if c in X.columns])
    kept_num = [c for c in state["num_cols"] if c not in state["corr_drop"]]

    if pca_components:
        if fit:
            med = X[kept_num].median()
            sc = StandardScaler().fit(X[kept_num].fillna(med))
            pca = PCA(n_components=min(pca_components, len(kept_num)), random_state=config.SEED)
            pca.fit(sc.transform(X[kept_num].fillna(med)))
            state.update(pca_med=med, pca_scaler=sc, pca=pca)
            log.info("%s: PCA %d comp. explican %.1f%% de la varianza", name, pca.n_components_,
                     100 * pca.explained_variance_ratio_.sum())
        Z = state["pca"].transform(state["pca_scaler"].transform(X[kept_num].fillna(state["pca_med"])))
        X = pd.concat([X, pd.DataFrame(Z.astype("float32"), columns=[f"pc{i + 1}" for i in range(Z.shape[1])],
                                       index=X.index)], axis=1)

    X = X.rename(columns={c: f"{name}_{c}" for c in X.columns if c != KEY})
    X = X.assign(**{f"{name}_tiene_registro": 1})
    if fit:
        state["columns"] = list(X.columns)
    else:
        X = X.reindex(columns=state["columns"])
    return X, state
