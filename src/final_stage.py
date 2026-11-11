from __future__ import annotations

import os

import joblib
import numpy as np

from . import config
from . import modeling as M
from .utils import log


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main() -> None:
    cfg = joblib.load(config.MODELS_DIR / "final_config.joblib")
    feats, sources = cfg["features"], tuple(cfg["sources"])
    y_df, sub, pred_col, key_name = M._target_and_template()
    y = y_df.set_index(M.KEY)["target"].astype(int)

    cap = int(os.environ.get("IB_FINAL_MAX_ROWS", 0))
    keys = None
    if cap and cap < len(y):
        _, ys = M.stratified_sample(y.to_frame(), y, cap)
        keys = ys.index
    X = M.load_split("train", sources, columns=feats, keys=keys)
    log.info("Reentrenando con %s clientes de train (de %s): %s", f"{len(X):,}", f"{len(y):,}", X.shape)
    fit = M.fit_models(X, y.loc[X.index], cfg["params"], logreg_C=cfg["logreg_C"], n_seeds=cfg["n_seeds"])
    del X

    X_test = M.load_split("test", sources, columns=feats)
    pred = M.predict_models(fit, X_test)
    s_test = M.blend_scores(pred["p_lgbm"], pred["p_logreg"], cfg["w_lgbm"], cfg["blend_stats"])
    p_test = cfg["calibrator"].predict(sigmoid(s_test))

    out = sub.copy()
    assert (out[M.KEY].values == X_test.index.values).all()
    out[pred_col] = p_test
    out.rename(columns={M.KEY: key_name}).to_csv(config.REPORTS_DIR / "submission.csv", index=False)
    np.savez(config.MODELS_DIR / "predictions_test.npz", key=X_test.index.values, s_test=s_test, p_test=p_test,
             p_lgbm=pred["p_lgbm"], p_logreg=pred["p_logreg"])
    joblib.dump({"prep_tree": fit["prep_tree"], "prep_linear": fit["prep_linear"], "lgbm_models": fit["lgbm_models"],
                 "logreg": fit["logreg"], **{k: cfg[k] for k in ("features", "params", "w_lgbm", "blend_stats",
                                                                  "calibrator")}},
                config.MODELS_DIR / "model_final.joblib")
    log.info("Listo: %s filas en %s", len(out), config.REPORTS_DIR / "submission.csv")


if __name__ == "__main__":
    main()
