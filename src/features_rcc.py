from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from . import config
from .utils import (find_col, is_text, is_yyyymm, log, months_before_reference, safe_name,
                    top_levels, categorical_columns)

PREFIX = "rcc_"
KEY = "key_value"

FAMILY_RULES = [
    ("linea_total_tc", ("linea total tc",)),
    ("linea_no_utilizada", ("no utilizada",)),
    ("garantia", ("garantia",)),
    ("contable", ("rendimientos", "provision", "resto rd", "resto ri", "no desembolsad", "derivados")),
    ("problema", ("castig", "refinanc")),
    ("tarjeta", ("tarjeta", "avance")),
    ("hipotecario", ("hipotec", "inmobiliar", "mivivienda")),
    ("vehicular", ("vehic",)),
    ("empresarial", ("micro empresa", "comercial", "descuento", "factoring", "leasing", "sobregiro", "fianza",
                     "cartas", "pequena", "mype", "corporat")),
    ("consumo", ("personal", "convenio", "otros creditos", "consumo", "efectivo")),
]
DEBT_FAMILIES = {"tarjeta", "hipotecario", "vehicular", "empresarial", "consumo", "problema", "otros"}


def _strip(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return f" {text.lower()} "


def product_family(description: str) -> str:
    text = _strip(description)
    for family, keys in FAMILY_RULES:
        if any(k in text for k in keys):
            return family
    return "otros"


def detect_schema(df: pd.DataFrame) -> dict:
    schema = {
        "key": find_col(df, "key", "rcc", required=True),
        "period": find_col(df, "period", "rcc"),
        "amount": find_col(df, "amount", "rcc"),
        "institution": find_col(df, "institution", "rcc"),
        "product": find_col(df, "product", "rcc"),
        "clasif": find_col(df, "clasif", "rcc"),
        "mora": find_col(df, "mora", "rcc"),
    }
    if schema["period"] is None:
        for c in df.columns:
            if c != schema["key"] and is_yyyymm(df[c]):
                schema["period"] = c
                break
    if schema["period"] is None:
        raise KeyError(f"RCC: no encuentro la columna de periodo. Columnas: {list(df.columns)}")

    assigned = {v for v in schema.values() if v}
    if schema["amount"] is None:
        numeric = [c for c in df.select_dtypes("number").columns if c not in assigned]
        if not numeric:
            raise KeyError(f"RCC: no encuentro una columna de monto. Columnas: {list(df.columns)}")
        schema["amount"] = max(numeric, key=lambda c: df[c].nunique())
        assigned.add(schema["amount"])

    rest = [c for c in df.columns if c not in assigned]
    cats = set(categorical_columns(df[rest]))
    schema["extra_dims"] = [c for c in rest if c in cats]
    schema["extra_measures"] = [c for c in rest if c not in cats and pd.api.types.is_numeric_dtype(df[c])]
    return schema


def _as_str_category(s: pd.Series) -> pd.Series:
    if pd.api.types.is_float_dtype(s):
        v = s.dropna()
        if len(v) and (v % 1 == 0).all():
            s = s.astype("Int64")
    c = s.astype("category")
    return c.cat.rename_categories([str(x) for x in c.cat.categories]).reset_index(drop=True)


def _family_map(df: pd.DataFrame, product_col: str | None, productos: pd.DataFrame | None) -> dict:
    if product_col is None:
        return {}
    values = pd.Series(df[product_col].dropna().unique()).pipe(lambda v: v.astype("int64") if pd.api.types.is_float_dtype(v) and (v % 1 == 0).all() else v).to_numpy()
    desc_map: dict = {v: str(v) for v in values}
    if productos is not None and len(productos):
        code_col = next((c for c in productos.columns
                         if set(map(str, productos[c].dropna().unique())) & set(map(str, values))), None)
        if code_col is not None:
            text_cols = [c for c in productos.columns if c != code_col and is_text(productos[c])]
            if text_cols:
                lookup = productos.set_index(productos[code_col].astype(str))[text_cols].astype(str).agg(" ".join, axis=1)
                desc_map = {v: lookup.get(str(v), str(v)) for v in values}
    return {v: product_family(d) for v, d in desc_map.items()}


def _window_stats(monthly: pd.DataFrame, w: int, div: float) -> pd.DataFrame:
    g = monthly[monthly["mb"] <= w].groupby("key")["amt"]
    out = pd.DataFrame({
        f"{PREFIX}deuda_prom_{w}m": g.sum() / div,
        f"{PREFIX}deuda_max_{w}m": g.max(),
        f"{PREFIX}deuda_std_{w}m": g.std(),
        f"{PREFIX}meses_activos_{w}m": g.size(),
    })
    return out


def _ratio(a: pd.Series, b: pd.Series, clip: tuple[float, float] = (0, 10)) -> pd.Series:
    r = a / b.replace(0, np.nan)
    return r.clip(*clip)


def build_rcc_features(df: pd.DataFrame, ref_month: int, state: dict | None = None,
                       productos: pd.DataFrame | None = None, amount_shift: float | None = None) -> tuple[pd.DataFrame, dict]:
    fit = state is None
    schema = detect_schema(df) if fit else state["schema"]
    if fit:
        log.info("RCC esquema detectado: %s", {k: v for k, v in schema.items() if v})

    d = pd.DataFrame({"key": df[schema["key"]].to_numpy()})
    d["mb"] = months_before_reference(df[schema["period"]], ref_month).values
    amt = pd.to_numeric(df[schema["amount"]], errors="coerce")
    if fit:
        shift = amount_shift if amount_shift is not None else (-float(np.nanmin(amt)) if (amt < 0).mean() > 0.02 else 0.0)
        schema["amount_shift"] = max(shift, 0.0)
        if schema["amount_shift"] > 0:
            log.info("RCC: monto estandarizado; se desplaza en %.6f para recuperar una escala no negativa.", schema["amount_shift"])
    d["amt"] = (amt + schema["amount_shift"]).fillna(0).clip(lower=0).to_numpy()
    for role in ("institution", "product"):
        if schema[role]:
            d[role[:4]] = _as_str_category(df[schema[role]])
    for role in ("clasif", "mora"):
        if schema[role]:
            d[role[:4]] = pd.to_numeric(df[schema[role]], errors="coerce").values
    for c in schema["extra_dims"]:
        d[f"dim__{c}"] = _as_str_category(df[c])
    for c in schema["extra_measures"]:
        d[f"mea__{c}"] = pd.to_numeric(df[c], errors="coerce").values

    n_future = int((d["mb"] < 1).sum())
    if n_future:
        log.warning("RCC: %d registros (%.2f%%) son del mes de referencia o posteriores y se descartan.",
                    n_future, 100 * n_future / len(d))
    d = d[(d["mb"] >= 1) & (d["mb"] <= max(config.WINDOWS))].copy()
    d["mb"] = d["mb"].astype(int)

    if fit:
        state = {"schema": schema, "n_months": int(d["mb"].max())}
        state["top_inst"] = top_levels(d["inst"], config.TOP_K) if "inst" in d else []
        state["top_prod"] = top_levels(d["prod"], config.TOP_K) if "prod" in d else []
        fmap = _family_map(df, schema["product"], productos)
        state["family_map"] = {str(k): v for k, v in fmap.items()}
        state["families"] = sorted(set(fmap.values()))
        state["use_debt_filter"] = len(set(fmap.values()) - {"otros"}) >= 3
        state["mora_days"] = bool("mora" in d and d["mora"].max() > 100)
        state["bad_thr"] = {r: float(np.nanmin(d[r])) for r in ("clas", "mora") if r in d and d[r].notna().any()}
        state["top_dims"] = {c: top_levels(d[f"dim__{c}"], 8) for c in schema["extra_dims"]}

    if "prod" in d:
        d["fam"] = d["prod"].map(state["family_map"]).fillna("otros")
    dd = d[d["fam"].isin(DEBT_FAMILIES)].copy() if ("fam" in d and state["use_debt_filter"]) else d

    feats: list[pd.DataFrame] = []
    W = config.WINDOWS

    def ew(w: int) -> int:
        return min(w, state["n_months"])

    monthly = dd.groupby(["key", "mb"], as_index=False)["amt"].sum()
    for w in W:
        feats.append(_window_stats(monthly, w, ew(w)))
    recent = monthly.sort_values("mb").groupby("key").first()
    feats.append(pd.DataFrame({
        f"{PREFIX}deuda_ultimo_mes": recent["amt"],
        f"{PREFIX}meses_desde_ultimo_registro": recent["mb"],
        f"{PREFIX}antiguedad_historial": monthly.groupby("key")["mb"].max(),
    }))

    f0 = pd.concat(feats, axis=1)
    trend = pd.DataFrame(index=f0.index)
    trend[f"{PREFIX}tend_3m_vs_6m"] = _ratio(f0[f"{PREFIX}deuda_prom_3m"], f0[f"{PREFIX}deuda_prom_6m"])
    trend[f"{PREFIX}tend_3m_vs_12m"] = _ratio(f0[f"{PREFIX}deuda_prom_3m"], f0[f"{PREFIX}deuda_prom_12m"])
    trend[f"{PREFIX}tend_6m_vs_12m"] = _ratio(f0[f"{PREFIX}deuda_prom_6m"], f0[f"{PREFIX}deuda_prom_12m"])
    feats.append(trend)
    mn = monthly.groupby("key")["amt"].agg(["mean", "std"])
    feats.append(pd.DataFrame({f"{PREFIX}coef_variacion_12m": (mn["std"] / mn["mean"].replace(0, np.nan)).clip(0, 10)}))

    for role, tag in (("inst", "inst"), ("prod", "prod")):
        if role not in d:
            continue
        for w in W:
            feats.append(dd[dd["mb"] <= w].groupby("key")[role].nunique().rename(f"{PREFIX}n_{tag}_{w}m").to_frame())
        oldest = dd.groupby(["key", role])["mb"].max().reset_index()
        new = oldest[oldest["mb"] <= 3].groupby("key").size().rename(f"{PREFIX}n_{tag}_nuevas_3m")
        feats.append(new.to_frame())
        feats.append(dd.groupby("key").size().rename(f"{PREFIX}n_registros").to_frame() if role == "inst" else pd.DataFrame())

    if "inst" in d:
        byinst = dd[dd["mb"] <= 6].groupby(["key", "inst"])["amt"].sum().reset_index()
        tot = byinst.groupby("key")["amt"].transform("sum").replace(0, np.nan)
        byinst["share"] = byinst["amt"] / tot
        feats.append(pd.DataFrame({
            f"{PREFIX}share_inst_principal_6m": byinst.groupby("key")["share"].max(),
            f"{PREFIX}hhi_inst_6m": (byinst["share"] ** 2).groupby(byinst["key"]).sum(),
        }))
    if "inst" in d:
        for w in (3, 12):
            sub = dd[(dd["inst"].isin(state["top_inst"])) & (dd["mb"] <= w)]
            pv = sub.pivot_table(index="key", columns="inst", values="amt", aggfunc="sum", fill_value=0) / ew(w)
            pv = pv.reindex(columns=state["top_inst"], fill_value=0)
            pv.columns = [f"{PREFIX}inst_{safe_name(c)}_prom_{w}m" for c in pv.columns]
            feats.append(pv)

    if "fam" in d:
        fam_avg = {}
        for w in W:
            sub = d[d["mb"] <= w]
            pv = sub.pivot_table(index="key", columns="fam", values="amt", aggfunc="sum", fill_value=0) / ew(w)
            pv = pv.reindex(columns=state["families"], fill_value=0)
            fam_avg[w] = pv
            out = pv.copy()
            out.columns = [f"{PREFIX}fam_{c}_prom_{w}m" for c in out.columns]
            feats.append(out)
        act = d[d["amt"] > 0].groupby(["key", "fam"])["mb"].nunique().unstack(fill_value=0)
        act = act.reindex(columns=state["families"], fill_value=0)
        act.columns = [f"{PREFIX}fam_{c}_meses_con_saldo" for c in act.columns]
        feats.append(act)
        if {"linea_total_tc", "tarjeta"} <= set(state["families"]):
            for w in (3, 12):
                feats.append(pd.DataFrame({
                    f"{PREFIX}utilizacion_tc_{w}m": _ratio(fam_avg[w]["tarjeta"], fam_avg[w]["linea_total_tc"], (0, 5))
                }))
        if True:
            tot12 = fam_avg[12][[f for f in state["families"] if f in DEBT_FAMILIES]].sum(axis=1).replace(0, np.nan)
            for fam in state["families"]:
                if fam in DEBT_FAMILIES:
                    feats.append(pd.DataFrame({f"{PREFIX}peso_{fam}_12m": fam_avg[12][fam] / tot12}))
        sub = d[d["prod"].isin(state["top_prod"]) & (d["mb"] <= 12)]
        pv = (sub.pivot_table(index="key", columns="prod", values="amt", aggfunc="sum", fill_value=0) / ew(12))
        pv = pv.reindex(columns=state["top_prod"], fill_value=0)
        pv.columns = [f"{PREFIX}prod_{safe_name(c)}_prom_12m" for c in pv.columns]
        feats.append(pv)

    for col, tag in (("clas", "clasif"), ("mora", "mora")):
        if col not in dd or col not in state["bad_thr"]:
            continue
        thr = state["bad_thr"][col]
        dd[f"bad_{col}"] = (dd[col] > thr).astype(int)
        for w in W:
            sub = dd[dd["mb"] <= w]
            g = sub.groupby("key")
            feats.append(pd.DataFrame({
                f"{PREFIX}{tag}_max_{w}m": g[col].max(),
                f"{PREFIX}{tag}_prom_{w}m": g[col].mean(),
                f"{PREFIX}{tag}_registros_malos_{w}m": g[f"bad_{col}"].sum(),
                f"{PREFIX}{tag}_meses_malos_{w}m": sub[sub[f"bad_{col}"] == 1].groupby("key")["mb"].nunique(),
                f"{PREFIX}{tag}_saldo_malo_{w}m": sub[sub[f"bad_{col}"] == 1].groupby("key")["amt"].sum() / ew(w),
            }))
        if col == "mora" and state["mora_days"]:
            for w in W:
                sub = dd[dd["mb"] <= w]
                for days in (30, 90):
                    feats.append(sub[sub[col] > days].groupby("key")["mb"].nunique()
                                 .rename(f"{PREFIX}mora_gt{days}d_meses_{w}m").to_frame())
        bad = dd[dd[f"bad_{col}"] == 1]
        since = bad.groupby("key")["mb"].min().rename(f"{PREFIX}{tag}_meses_desde_ultimo_malo")
        feats.append(since.to_frame())
        last_month = dd[dd["mb"] == dd.groupby("key")["mb"].transform("min")]
        feats.append(last_month.groupby("key")[col].max().rename(f"{PREFIX}{tag}_max_ultimo_mes").to_frame())

    for c in schema["extra_dims"]:
        sub = d[d["mb"] <= 12]
        feats.append(sub.groupby("key")[f"dim__{c}"].nunique().rename(f"{PREFIX}n_{safe_name(c)}_12m").to_frame())
        lv = state["top_dims"][c]
        cnt = sub[sub[f"dim__{c}"].isin(lv)].pivot_table(index="key", columns=f"dim__{c}", values="amt",
                                                         aggfunc="size", fill_value=0)
        cnt = cnt.reindex(columns=lv, fill_value=0)
        cnt.columns = [f"{PREFIX}{safe_name(c)}_{safe_name(x)}_n12m" for x in cnt.columns]
        feats.append(cnt)
    for c in schema["extra_measures"]:
        for w in (3, 12):
            g = d[d["mb"] <= w].groupby("key")[f"mea__{c}"]
            feats.append(pd.DataFrame({
                f"{PREFIX}{safe_name(c)}_prom_{w}m": g.mean(),
                f"{PREFIX}{safe_name(c)}_max_{w}m": g.max(),
            }))

    out = pd.concat([f for f in feats if len(f.columns)], axis=1)
    out = out.loc[:, ~out.columns.duplicated()]

    zero_like = ("meses_activos", "meses_malos", "d_meses_",  "registros_malos", "saldo_malo", "meses_con_saldo", "_n12m")
    for c in out.columns:
        if c.startswith(f"{PREFIX}n_") or any(t in c for t in zero_like):
            out[c] = out[c].fillna(0)
    for c in out.columns:
        if c.endswith("meses_desde_ultimo_malo"):
            out[c] = out[c].fillna(max(config.WINDOWS) + 1)

    out = out.rename_axis(KEY).reset_index()
    out[f"{PREFIX}tiene_historial"] = 1
    if fit:
        state["columns"] = list(out.columns)
    else:
        out = out.reindex(columns=state["columns"])
    return out.astype({c: "float32" for c in out.columns if c != KEY and out[c].dtype == "float64"}), state
