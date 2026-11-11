from __future__ import annotations

import gc
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config, utils
from .features_rcc import build_rcc_features, detect_schema
from .features_static import build_static_features
from .utils import downcast, load_raw, log, normalize_name, save_frame

STATIC_OPTIONS = {
    "sunat": dict(corr_threshold=None, pca_components=0, pivot_numeric_by=("ciiu",)),
    "se": dict(corr_threshold=0.95, pca_components=0),
    "censo": dict(corr_threshold=0.95, pca_components=5),
}


N_PARTITIONS = 24
BATCH_ROWS = 1_000_000


def _iter_batches(path: Path):
    if path.suffix == ".parquet":
        for b in pq.ParquetFile(path).iter_batches(batch_size=BATCH_ROWS):
            df = b.to_pandas()
            df.columns = [normalize_name(c) for c in df.columns]
            yield df
    else:
        for df in pd.read_csv(path, chunksize=BATCH_ROWS, low_memory=False):
            df.columns = [normalize_name(c) for c in df.columns]
            yield df


def partition_rcc(path: Path, ref_month: int, workdir: Path) -> tuple[list[Path], float]:
    workdir.mkdir(parents=True, exist_ok=True)
    key_col = amt_col = per_col = None
    min_amt, total, kept, future = np.inf, 0, 0, 0
    for i, df in enumerate(_iter_batches(path)):
        if key_col is None:
            sch = detect_schema(df)
            key_col, amt_col, per_col = sch["key"], sch["amount"], sch["period"]
            if not utils.is_yyyymm(df[per_col]):
                raise ValueError("El procesamiento por bloques requiere periodos YYYYMM en el RCC.")
        total += len(df)
        min_amt = min(min_amt, float(df[amt_col].min()))
        mb = utils.months_before_reference(df[per_col], ref_month)
        future += int((mb < 1).sum())
        df = df[(mb >= 1) & (mb <= max(config.WINDOWS))]
        kept += len(df)
        part = (df[key_col].to_numpy() % N_PARTITIONS)
        for p in np.unique(part):
            df[part == p].to_parquet(workdir / f"p{p:03d}_{i:04d}.parquet", index=False)
    log.info("RCC %s: %s filas leídas · %s dentro de la ventana · %s del mes de desembolso o posteriores descartadas",
             path.name, f"{total:,}", f"{kept:,}", f"{future:,}")
    return sorted(workdir.glob("p*.parquet")), min_amt


def _build_partitions(files: list[Path], ref_month: int, state, productos, amount_shift):
    by_part: dict[str, list[Path]] = {}
    for f in files:
        by_part.setdefault(f.name[:4], []).append(f)
    out = []
    for name in sorted(by_part):
        df = pd.concat([pd.read_parquet(f) for f in by_part[name]], ignore_index=True)
        feats, state = build_rcc_features(df, ref_month, state, productos, amount_shift)
        out.append(feats)
        del df
    return pd.concat(out, ignore_index=True).sort_values("key_value").reset_index(drop=True), state


def run_rcc() -> dict:
    productos = load_raw("productos", required=False)
    tmp = config.PROC_DIR / "_rcc_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    results, state = {}, None
    for split in ("train", "test"):
        path = utils.raw_file(f"rcc_{split}")
        if path is None:
            raise FileNotFoundError(f"Falta rcc_{split}.csv/.parquet en {config.RAW_DIR}")
        files, min_amt = partition_rcc(path, config.REF_MONTH[split], tmp / split)
        if split == "train":
            shift = max(-min_amt, 0.0)
            first = sorted({f.name[:4] for f in files})[0]
            _, state = build_rcc_features(
                pd.concat([pd.read_parquet(f) for f in files if f.name.startswith(first)], ignore_index=True),
                config.REF_MONTH["train"], None, productos, shift)
        feats, _ = _build_partitions(files, config.REF_MONTH[split], state, productos, state["schema"]["amount_shift"])
        save_frame(downcast(feats), f"features_rcc_{split}")
        results[split] = feats.shape
        shutil.rmtree(tmp / split, ignore_errors=True)
    joblib.dump(state, config.MODELS_DIR / "state_rcc.joblib")
    shutil.rmtree(tmp, ignore_errors=True)
    log.info("RCC listo: train %s | test %s", results["train"], results["test"])
    return results


def run_static(name: str) -> dict:
    opts = STATIC_OPTIONS[name]
    tr = downcast(load_raw(f"{name}_train"))
    f_tr, state = build_static_features(tr, name, config.REF_MONTH["train"], None, **opts)
    del tr
    gc.collect()
    te = downcast(load_raw(f"{name}_test"))
    f_te, _ = build_static_features(te, name, config.REF_MONTH["test"], state, **opts)
    del te
    save_frame(downcast(f_tr), f"features_{name}_train")
    save_frame(downcast(f_te), f"features_{name}_test")
    joblib.dump(state, config.MODELS_DIR / f"state_{name}.joblib")
    log.info("%s listo: train %s | test %s", name.upper(), f_tr.shape, f_te.shape)
    return {"train": f_tr.shape, "test": f_te.shape}


def run_all() -> None:
    run_rcc()
    for name in ("sunat", "se", "censo"):
        run_static(name)
