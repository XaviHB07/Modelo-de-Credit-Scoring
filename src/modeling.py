from __future__ import annotations

import gc
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, TargetEncoder

from . import config
from .utils import is_text, load_frame, load_raw, log, find_col, psi, summarize_metrics

KEY = "key_value"
LOW_CARD = 12


def _target_and_template() -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    y_df = load_raw("y_train")
    ykey, ycol = find_col(y_df, "key", "y", required=True), find_col(y_df, "target", "y", required=True)
    y_df = y_df.rename(columns={ykey: KEY, ycol: "target"})[[KEY, "target"]]
    sub = load_raw("sample_submission")
    skey = find_col(sub, "key", "sub", required=True)
    pred_col = [c for c in sub.columns if c != skey][0]
    return y_df, sub.rename(columns={skey: KEY}), pred_col, skey


def load_split(split: str, sources: tuple[str, ...] = ("rcc", "sunat", "se", "censo"),
               columns: list[str] | None = None, keys: pd.Index | None = None) -> pd.DataFrame:
    y_df, sub, _, _ = _target_and_template()
    base = (y_df if split == "train" else sub)[[KEY]]
    if keys is not None:
        base = base[base[KEY].isin(keys)]
    X, flags = base.copy(), []
    for src in sources:
        f = load_frame(f"features_{src}_{split}", columns)
        if keys is not None:
            f = f[f[KEY].isin(keys)]
        X = X.merge(f, on=KEY, how="left")
        flags += [c for c in f.columns if c.endswith(("_tiene_historial", "_tiene_registro"))]
        del f
    X[flags] = X[flags].fillna(0)
    X["n_fuentes_con_info"] = X[flags].sum(axis=1)
    X = X.set_index(KEY)
    return X if columns is None else X[[c for c in columns if c in X.columns]]


def load_dataset(sources: tuple[str, ...] = ("rcc", "sunat", "se", "censo"), load_test: bool = True,
                 max_rows: int = 0) -> dict:
    y_df, sub, pred_col, skey = _target_and_template()
    out = {"submission_template": sub, "pred_col": pred_col, "key_name": skey}
    y_full = y_df.set_index(KEY)["target"].astype(int)
    out["y_full"] = y_full
    keys = None
    if max_rows and len(y_full) > max_rows:
        _, ys = stratified_sample(y_full.to_frame(), y_full, max_rows)
        keys = ys.index
    out["X_train"] = load_split("train", sources, keys=keys)
    out["y"] = y_full.loc[out["X_train"].index]
    if load_test:
        out["X_test"] = load_split("test", sources)
    log.info("Dataset: train %s%s | tasa de default %.2f%%", out["X_train"].shape,
             f" | test {out['X_test'].shape}" if load_test else "", 100 * out["y"].mean())
    return out


def unsupervised_prune(X_train: pd.DataFrame, X_test: pd.DataFrame, max_missing: float = 0.98,
                       corr_threshold: float = 0.98) -> tuple[list[str], pd.DataFrame]:
    report = []
    keep = list(X_train.columns)
    for c in list(keep):
        s = X_train[c]
        if s.isna().mean() > max_missing:
            report.append((c, "casi_vacia")); keep.remove(c)
        elif s.nunique(dropna=True) <= 1:
            report.append((c, "constante")); keep.remove(c)
    num = [c for c in keep if not is_text(X_train[c])]
    sample = X_train[num] if len(X_train) <= 20_000 else X_train[num].sample(20_000, random_state=config.SEED)
    corr = sample.corr(method="spearman").abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    for c in upper.columns:
        mates = upper.index[upper[c] > corr_threshold]
        if len(mates):
            report.append((c, f"corr>{corr_threshold} con {mates[0]}")); keep.remove(c)
    return keep, pd.DataFrame(report, columns=["variable", "motivo"])


class Prep(BaseEstimator, TransformerMixin):

    def __init__(self, kind: str = "tree", seed: int = config.SEED):
        self.kind, self.seed = kind, seed

    def _build(self, X: pd.DataFrame) -> pd.DataFrame:
        cat = [c for c in X.columns if is_text(X[c])]
        self.low_ = [c for c in cat if X[c].nunique() <= LOW_CARD]
        self.high_ = [c for c in cat if c not in self.low_]
        self.num_ = [c for c in X.columns if c not in cat]
        num_step = "passthrough" if self.kind == "tree" else Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("qt", QuantileTransformer(n_quantiles=min(200, len(X)), output_distribution="normal",
                                       random_state=self.seed)),
        ])
        steps = [("num", num_step, self.num_)]
        if self.low_:
            steps.append(("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False),
                          self.low_))
        if self.high_:
            steps.append(("te", TargetEncoder(target_type="binary", cv=5, random_state=self.seed), self.high_))
        self.ct_ = ColumnTransformer(steps, verbose_feature_names_out=False).set_output(transform="pandas")
        return X.astype({c: "object" for c in cat})

    def fit(self, X: pd.DataFrame, y=None):
        Xc = self._build(X)
        self.ct_.fit(Xc, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        Xc = self._build(X)
        return self._finish(self.ct_.fit_transform(Xc, y))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cat = self.low_ + self.high_
        return self._finish(self.ct_.transform(X.astype({c: "object" for c in cat})))

    @staticmethod
    def _finish(out: pd.DataFrame) -> pd.DataFrame:
        names, seen = [], {}
        for c in out.columns:
            n = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
            seen[n] = seen.get(n, -1) + 1
            names.append(n if seen[n] == 0 else f"{n}_{seen[n]}")
        out.columns = names
        return out.astype("float32")


@dataclass
class PreparedFolds:
    X: pd.DataFrame
    y: pd.Series
    kind: str
    n_splits: int = config.N_SPLITS
    n_repeats: int = config.N_REPEATS
    seed: int = config.SEED
    memory_limit: int = 60_000
    folds: list = field(default_factory=list)

    def __post_init__(self):
        self.on_disk = len(self.X) > self.memory_limit
        self.cache = None
        if self.on_disk:
            self.cache = Path(tempfile.mkdtemp(prefix="folds_", dir=config.PROC_DIR))
        cv = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.seed)
        for i, (tr, va) in enumerate(cv.split(self.X, self.y)):
            prep = Prep(self.kind, self.seed)
            Xtr = prep.fit_transform(self.X.iloc[tr], self.y.iloc[tr])
            Xva = prep.transform(self.X.iloc[va])
            fold = dict(rep=i // self.n_splits, tr=tr, va=va, cols=list(Xtr.columns),
                        ytr=self.y.iloc[tr].to_numpy(), yva=self.y.iloc[va].to_numpy())
            if self.on_disk:
                np.save(self.cache / f"tr_{i}.npy", Xtr.to_numpy())
                np.save(self.cache / f"va_{i}.npy", Xva.to_numpy())
                fold["i"] = i
            else:
                fold["Xtr"], fold["Xva"] = Xtr, Xva
            self.folds.append(fold)
            del Xtr, Xva, prep
            gc.collect()

    def load(self, f: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.on_disk:
            return f["Xtr"], f["Xva"]
        tr = pd.DataFrame(np.load(self.cache / f"tr_{f['i']}.npy"), columns=f["cols"], copy=False)
        va = pd.DataFrame(np.load(self.cache / f"va_{f['i']}.npy"), columns=f["cols"], copy=False)
        return tr, va

    def close(self) -> None:
        if self.cache is not None:
            shutil.rmtree(self.cache, ignore_errors=True)

    def evaluate(self, estimator, top_k: int | None = None) -> dict:
        oof = np.zeros((self.n_repeats, len(self.X)))
        aucs = []
        for f in self.folds:
            Xtr, Xva = self.load(f)
            if top_k:
                cols = select_top_features(Xtr, f["ytr"], top_k)
                Xtr, Xva = Xtr[cols], Xva[cols]
            m = clone(estimator).fit(Xtr, f["ytr"])
            p = m.predict_proba(Xva)[:, 1]
            oof[f["rep"], f["va"]] = p
            aucs.append(roc_auc_score(f["yva"], p))
            del Xtr, Xva, m
            gc.collect()
        return {"auc_folds": np.array(aucs), "oof": oof.mean(axis=0), "oof_by_rep": oof}


def select_top_features(X: pd.DataFrame, y, k: int) -> list[str]:
    sel = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=30,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.5, importance_type="gain",
                             random_state=config.SEED, verbose=-1, n_jobs=-1).fit(X, y)
    imp = pd.Series(sel.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(k).index.tolist()


def make_logreg(C: float = 0.05) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="lbfgs")


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=50, max_features="sqrt",
                                  max_samples=0.5, class_weight="balanced_subsample", n_jobs=-1,
                                  random_state=config.SEED)


LGB_DEFAULT = dict(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.5, reg_lambda=5.0, reg_alpha=0.0,
                   random_state=config.SEED, verbose=-1, n_jobs=-1)


def make_lgbm(**params) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**{**LGB_DEFAULT, **params})


def tune_lgbm(prepared: PreparedFolds, n_trials: int = config.OPTUNA_TRIALS, timeout: int | None = None) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pos = float(prepared.y.mean())
    max_spw = max(1.0, np.sqrt((1 - pos) / pos))
    folds = prepared.folds[: prepared.n_splits]

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 150, 700),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves=trial.suggest_int("num_leaves", 4, 48),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.9),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-2, 50, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, max_spw),
        )
        scores = []
        for f in folds:
            Xtr, Xva = prepared.load(f)
            m = make_lgbm(**params).fit(Xtr, f["ytr"])
            scores.append(roc_auc_score(f["yva"], m.predict_proba(Xva)[:, 1]))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    log.info("Optuna: mejor AUC CV %.4f en %d trials", study.best_value, len(study.trials))
    return {"params": study.best_params, "best_value": study.best_value, "study": study}


class PlattCalibrator:

    def fit(self, p: np.ndarray, y: np.ndarray):
        z = self._logit(p)[:, None]
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return self

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self.lr_.predict_proba(self._logit(p)[:, None])[:, 1]


def expected_calibration_error(y, p, bins: int = 10) -> float:
    df = pd.DataFrame({"y": y, "p": p})
    df["b"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("b").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] * (g["y"] - g["p"]).abs()).sum() / len(df))


def threshold_table(y, p, cost_fn_over_fp: float = config.COST_FN_OVER_FP, n: int = 50) -> pd.DataFrame:
    y, p = np.asarray(y), np.asarray(p)
    rows = []
    for t in np.quantile(p, np.linspace(0.02, 0.98, n)):
        approve = p <= t
        fn = int(((y == 1) & approve).sum())
        fp = int(((y == 0) & ~approve).sum())
        rows.append(dict(corte=t, aprobacion=approve.mean(), mora_aprobados=y[approve].mean() if approve.any() else np.nan,
                         malos_aprobados=fn, buenos_rechazados=fp, costo=(cost_fn_over_fp * fn + fp) / len(y)))
    return pd.DataFrame(rows)


def adversarial_validation(X_train: pd.DataFrame, X_test: pd.DataFrame, seed: int = config.SEED,
                           max_rows: int = 30_000, n_splits: int = 3) -> dict:
    if len(X_train) > max_rows:
        X_train = X_train.sample(max_rows, random_state=seed)
    if len(X_test) > max_rows:
        X_test = X_test.sample(max_rows, random_state=seed)
    A = pd.concat([X_train, X_test], axis=0)
    label = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]
    oof = np.zeros(len(A))
    imps = []
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(A, label):
        p = Prep("tree", seed)
        m = make_lgbm(n_estimators=200, num_leaves=15, colsample_bytree=0.6, importance_type="gain").fit(
            p.fit_transform(A.iloc[tr], label[tr]), label[tr])
        oof[va] = m.predict_proba(p.transform(A.iloc[va]))[:, 1]
        imps.append(pd.Series(m.feature_importances_, index=m.feature_name_))
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    return {"auc": float(roc_auc_score(label, oof)), "importance": imp / imp.sum()}


def psi_table(X_train: pd.DataFrame, X_test: pd.DataFrame, top: int | None = None) -> pd.DataFrame:
    rows = []
    for c in X_train.columns:
        if is_text(X_train[c]):
            a = X_train[c].value_counts(normalize=True)
            b = X_test[c].value_counts(normalize=True).reindex(a.index, fill_value=0)
            a, b = a.clip(1e-6), b.clip(1e-6)
            rows.append((c, float(((b - a) * np.log(b / a)).sum())))
        else:
            rows.append((c, psi(X_train[c], X_test[c])))
    t = pd.DataFrame(rows, columns=["variable", "psi"]).sort_values("psi", ascending=False)
    return t.head(top) if top else t


def holdout_split(X: pd.DataFrame, y: pd.Series, size: float = config.HOLDOUT_SIZE, seed: int = config.SEED):
    return train_test_split(X, y, test_size=size, stratify=y, random_state=seed)


def bootstrap_ci(y, p, metric=roc_auc_score, n: int = 500, seed: int = config.SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metric(y[idx], p[idx]))
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def drift_prune(X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float = 0.65, max_iter: int = 6,
                drop_per_iter: int = 3, initial: dict | None = None) -> tuple[list[str], pd.DataFrame]:
    keep = list(X_train.columns)
    history = []
    for it in range(max_iter + 1):
        adv = initial if (it == 0 and initial is not None) else adversarial_validation(X_train[keep], X_test[keep])
        history.append(dict(iteracion=it, auc_adversarial=adv["auc"], n_variables=len(keep)))
        if adv["auc"] <= threshold or it == max_iter:
            break
        drop = adv["importance"].index[:drop_per_iter]
        origin = [next((c for c in keep if d == c or d.startswith(c + "_")), None) for d in drop]
        keep = [c for c in keep if c not in set(filter(None, origin))]
    return keep, pd.DataFrame(history)


SOURCE_PREFIXES = ("rcc", "sunat", "se", "censo")


def feature_source(name: str) -> str:
    for p in SOURCE_PREFIXES:
        if name.startswith(p + "_"):
            return p
    return "otras"


def lgbm_contributions(model: lgb.LGBMClassifier, X: pd.DataFrame) -> pd.DataFrame:
    contrib = model.booster_.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend_stats(p_lgbm: np.ndarray, p_lr: np.ndarray) -> dict:
    return {k: (float(_logit(v).mean()), float(_logit(v).std())) for k, v in (("lgbm", p_lgbm), ("lr", p_lr))}


def blend_scores(p_lgbm: np.ndarray, p_lr: np.ndarray, w: float, stats: dict) -> np.ndarray:
    z_l = (_logit(p_lgbm) - stats["lgbm"][0]) / stats["lgbm"][1]
    z_r = (_logit(p_lr) - stats["lr"][0]) / stats["lr"][1]
    return w * z_l + (1 - w) * z_r


def fit_models(X: pd.DataFrame, y: pd.Series, lgb_params: dict, logreg_C: float = 0.05, n_seeds: int = 5,
               logreg_max_rows: int = 150_000) -> dict:
    pt = Prep("tree")
    Xt = pt.fit_transform(X, y)
    models = [make_lgbm(**{**lgb_params, "random_state": config.SEED + seed}).fit(Xt, y) for seed in range(n_seeds)]
    del Xt
    gc.collect()
    Xl, yl = stratified_sample(X, y, logreg_max_rows)
    pl = Prep("linear")
    lr = make_logreg(logreg_C).fit(pl.fit_transform(Xl, yl), yl)
    del Xl
    gc.collect()
    return dict(prep_tree=pt, prep_linear=pl, lgbm_models=models, logreg=lr)


def predict_models(fit: dict, X_new: pd.DataFrame, chunk: int = 100_000) -> dict:
    p_lgb, p_lr = np.zeros(len(X_new)), np.zeros(len(X_new))
    for a in range(0, len(X_new), chunk):
        part = X_new.iloc[a:a + chunk]
        p_lgb[a:a + chunk] = np.mean([m.predict_proba(fit["prep_tree"].transform(part))[:, 1]
                                      for m in fit["lgbm_models"]], axis=0)
        p_lr[a:a + chunk] = fit["logreg"].predict_proba(fit["prep_linear"].transform(part))[:, 1]
    return dict(p_lgbm=p_lgb, p_logreg=p_lr)


def fit_final(X: pd.DataFrame, y: pd.Series, X_new: pd.DataFrame, lgb_params: dict, **kw) -> dict:
    fit = fit_models(X, y, lgb_params, **kw)
    return {**fit, **predict_models(fit, X_new)}


def stratified_sample(X: pd.DataFrame, y: pd.Series, n: int, seed: int = config.SEED):
    if len(X) <= n:
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys
