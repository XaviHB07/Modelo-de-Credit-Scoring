from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .utils import is_text, load_frame


def overview(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    t = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulos_%": (df.isna().mean() * 100).round(2),
        "unicos": df.nunique(dropna=True),
        "ejemplo": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan),
    })
    print(f"{name}: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    return t


def information_value(x: pd.Series, y: pd.Series, bins: int = 10) -> float:
    d = pd.DataFrame({"y": y.values})
    if is_text(x):
        d["b"] = x.astype(str).values
    else:
        xs = pd.Series(x.values)
        try:
            d["b"] = pd.qcut(xs, bins, duplicates="drop").astype(str)
        except ValueError:
            d["b"] = xs.astype(str)
        d.loc[xs.isna().values, "b"] = "missing"
    g = d.groupby("b")["y"].agg(["sum", "count"])
    bad, good = g["sum"] + 0.5, (g["count"] - g["sum"]) + 0.5
    woe = np.log((good / good.sum()) / (bad / bad.sum()))
    return float((((good / good.sum()) - (bad / bad.sum())) * woe).sum())


def univariate_table(X: pd.DataFrame, y: pd.Series, top: int | None = None) -> pd.DataFrame:
    rows = []
    yv = y.loc[X.index] if hasattr(y, "loc") else y
    for c in X.columns:
        s = X[c]
        if s.nunique(dropna=True) < 2:
            continue
        if is_text(s):
            auc = np.nan
        else:
            mask = s.notna()
            auc = roc_auc_score(yv[mask], s[mask]) if yv[mask].nunique() == 2 else np.nan
            auc = max(auc, 1 - auc)
        rows.append(dict(variable=c, auc_univariado=auc, iv=information_value(s, yv),
                         nulos_pct=round(100 * s.isna().mean(), 1)))
    t = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    return t.head(top) if top else t


def iv_label(iv: float) -> str:
    return ("inútil" if iv < 0.02 else "débil" if iv < 0.1 else "media" if iv < 0.3
            else "fuerte" if iv < 0.5 else "muy fuerte")


def default_rate_by_bin(x: pd.Series, y: pd.Series, bins: int = 10, ax=None, title: str | None = None):
    d = pd.DataFrame({"x": x.values, "y": y.values})
    if is_text(x):
        d["b"] = d["x"].astype(str)
    else:
        try:
            d["b"] = pd.qcut(d["x"], bins, duplicates="drop")
        except ValueError:
            d["b"] = d["x"]
        d["b"] = d["b"].astype(str).where(d["x"].notna(), "missing")
    g = d.groupby("b", sort=False)["y"].agg(["mean", "size"])
    if not is_text(x):
        g = g.loc[sorted(g.index, key=lambda s: (s == "missing", _left_edge(s)))]
    ax = ax or plt.gca()
    ax.bar(range(len(g)), g["mean"] * 100, color="#3b6ea5")
    ax.axhline(d["y"].mean() * 100, color="crimson", ls="--", lw=1, label="tasa global")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([str(i)[:14] for i in g.index], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("% default")
    ax.set_title(title or str(x.name))
    ax.legend(fontsize=7)
    return g


def _left_edge(label: str) -> float:
    try:
        return float(label.strip("([").split(",")[0])
    except ValueError:
        return 0.0


def missing_plot(df: pd.DataFrame, top: int = 25, ax=None):
    m = (df.isna().mean() * 100).sort_values(ascending=False).head(top)
    ax = ax or plt.gca()
    m[::-1].plot.barh(ax=ax, color="#8c564b")
    ax.set_xlabel("% nulos")
    ax.set_title("Variables con más valores nulos")
    return m


def source_univariate(name: str, y: pd.Series, n: int = 80_000) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = load_frame(f"features_{name}_train").set_index("key_value")
    idx = X.index.intersection(y.index)
    sample = X.loc[idx].sample(min(n, len(idx)), random_state=0)
    table = univariate_table(sample, y.loc[sample.index])
    table["lectura_iv"] = table["iv"].map(iv_label)
    return X, table
