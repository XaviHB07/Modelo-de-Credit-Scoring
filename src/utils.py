from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score, roc_curve

from . import config


def get_logger(name: str = "credit_scoring") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger()


def read_csv_smart(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Copia los CSV de la competencia en {config.RAW_DIR} "
            "(o define la variable de entorno IB_RAW_DIR)."
        )
    try:
        df = pd.read_csv(path, low_memory=False, **kwargs)
    except UnicodeDecodeError:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1", **kwargs)
    df.columns = [normalize_name(c) for c in df.columns]
    return df


def normalize_name(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"\s+", "_", name)
    return name


def raw_file(key: str) -> Path | None:
    base = config.RAW_DIR / config.FILES[key]
    for cand in (base, base.with_suffix(".parquet")):
        if cand.exists():
            return cand
    return None


def read_table_smart(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        df.columns = [normalize_name(c) for c in df.columns]
        return df
    return read_csv_smart(path)


def load_raw(key: str, required: bool = True) -> pd.DataFrame | None:
    path = raw_file(key)
    if path is None:
        if not required:
            return None
        raise FileNotFoundError(
            f"No se encontró {config.FILES[key]} (ni su versión .parquet) en {config.RAW_DIR}. "
            "Copia los archivos de la competencia ahí o define la variable de entorno IB_RAW_DIR."
        )
    return read_table_smart(path)


def save_frame(df: pd.DataFrame, name: str) -> Path:
    path = config.PROC_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_frame(name: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = config.PROC_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Falta {path}. Ejecuta primero el script/notebook de esa fuente.")
    if columns is None:
        return pd.read_parquet(path)
    import pyarrow.parquet as pq
    available = pq.read_schema(path).names
    want = {"key_value", *columns}
    want |= {c for c in available if c.endswith(("_tiene_historial", "_tiene_registro"))}
    return pd.read_parquet(path, columns=[c for c in available if c in want])


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = out[col].astype("float32")
    for col in out.select_dtypes(include=["int64"]).columns:
        if out[col].abs().max() < 2**31 - 1:
            out[col] = out[col].astype("int32")
    return out


def find_col(df: pd.DataFrame, role: str, source: str | None = None, required: bool = False) -> str | None:
    if source and role in config.COLUMN_OVERRIDES.get(source, {}):
        col = config.COLUMN_OVERRIDES[source][role]
        if col not in df.columns:
            raise KeyError(f"Override {source}.{role}='{col}' no existe en el archivo.")
        return col
    for alias in config.ALIASES.get(role, []):
        if alias in df.columns:
            return alias
    if required:
        raise KeyError(
            f"No pude identificar la columna '{role}'. Columnas disponibles: {list(df.columns)}. "
            f"Agrega el nombre real a ALIASES['{role}'] o a COLUMN_OVERRIDES en src/config.py."
        )
    return None


def is_yyyymm(series: pd.Series) -> bool:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return False
    return bool(((s >= 190001) & (s <= 210012) & ((s % 100).between(1, 12))).mean() > 0.99)


def yyyymm_to_index(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return (s // 100) * 12 + (s % 100) - 1


def months_before_reference(period: pd.Series, ref_month: int) -> pd.Series:
    if is_yyyymm(period):
        return yyyymm_to_index(pd.Series([ref_month])).iloc[0] - yyyymm_to_index(period)
    p = pd.to_numeric(period, errors="coerce")
    codes = pd.Series(pd.factorize(p, sort=True)[0], index=p.index).where(p.notna())
    log.warning("Periodo no YYYYMM: se ordena y se asume que el último periodo es el mes previo al desembolso.")
    return (codes.max() + 1) - codes


def is_text(s: pd.Series) -> bool:
    return bool(pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
                or isinstance(s.dtype, pd.CategoricalDtype))


def categorical_columns(df: pd.DataFrame, exclude: Iterable[str] = (), max_card: int | None = None) -> list[str]:
    max_card = max_card or config.MAX_CARDINALITY_CATEGORICAL
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if is_text(df[c]):
            cols.append(c)
        elif pd.api.types.is_integer_dtype(df[c]) and df[c].nunique(dropna=True) <= max_card:
            cols.append(c)
    return cols


def top_levels(series: pd.Series, k: int) -> list:
    return series.value_counts(dropna=True).head(k).index.tolist()


def safe_name(x) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "", str(x).lower()) or "na"


def ks_statistic(y_true: Sequence, y_score: Sequence) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def gini(y_true: Sequence, y_score: Sequence) -> float:
    return 2 * roc_auc_score(y_true, y_score) - 1


def psi(expected: Sequence, actual: Sequence, bins: int = 10) -> float:
    expected = pd.Series(expected).dropna().to_numpy(dtype=float)
    actual = pd.Series(actual).dropna().to_numpy(dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return np.nan
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3:
        return 0.0
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected)
    a = np.histogram(actual, cuts)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def decile_table(y_true: Sequence, y_score: Sequence, n: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score)})
    df["decil"] = pd.qcut(df["p"].rank(method="first", ascending=False), n, labels=range(1, n + 1))
    t = df.groupby("decil", observed=True).agg(
        clientes=("y", "size"), defaults=("y", "sum"), tasa_default=("y", "mean"), score_medio=("p", "mean")
    )
    base = df["y"].mean()
    t["lift"] = t["tasa_default"] / base
    t["captura_acum"] = t["defaults"].cumsum() / df["y"].sum()
    return t.reset_index()


def summarize_metrics(y_true: Sequence, y_score: Sequence) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss

    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "gini": float(gini(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 0, 1))),
    }


def load_target() -> pd.Series:
    d = load_raw("y_train")
    key, tgt = find_col(d, "key", required=True), find_col(d, "target", required=True)
    return d.set_index(key)[tgt]


def read_column(path: str | Path, name: str) -> pd.Series:
    path = Path(path)
    if path.suffix == ".parquet":
        real = next(c for c in pq.ParquetFile(path).schema_arrow.names if normalize_name(c) == name)
        return pd.read_parquet(path, columns=[real])[real]
    return pd.read_csv(path, usecols=lambda c: normalize_name(c) == name).iloc[:, 0]


def sample_rows(path: str | Path, groups: int = 4) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        pf = pq.ParquetFile(path)
        step = max(1, pf.num_row_groups // groups)
        df = pf.read_row_groups(list(range(0, pf.num_row_groups, step))[:groups]).to_pandas()
    else:
        df = pd.read_csv(path, nrows=2_000_000)
    df.columns = [normalize_name(c) for c in df.columns]
    return df
