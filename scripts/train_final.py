import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib

from src import config, modeling as M
from src.utils import log, summarize_metrics


def main(skip_tuning: bool = False) -> dict:
    d = M.load_dataset()
    keep, _ = M.unsupervised_prune(d["X_train"], d["X_test"])
    X, X_test, y = d["X_train"][keep], d["X_test"][keep], d["y"]
    X_dev, X_ho, y_dev, y_ho = M.holdout_split(X, y)

    params_file = config.REPORTS_DIR / "best_params.json"
    if params_file.exists():
        params = json.loads(params_file.read_text())
    elif skip_tuning:
        params = {}
    else:
        params = M.tune_lgbm(M.PreparedFolds(X_dev, y_dev, "tree"))["params"]
        params_file.write_text(json.dumps(params, indent=2))

    ho = M.fit_final(X_dev, y_dev, X_ho, params, n_seeds=3)
    res = {"holdout_lgbm": summarize_metrics(y_ho, ho["p_lgbm"]),
           "holdout_logreg": summarize_metrics(y_ho, ho["p_logreg"])}
    res["holdout_lgbm"]["auc_ic95"] = list(M.bootstrap_ci(y_ho, ho["p_lgbm"]))
    log.info("Holdout LightGBM: %s", res["holdout_lgbm"])

    fin = M.fit_final(X, y, X_test, params, n_seeds=5)
    sub = d["submission_template"].copy()
    sub[d["pred_col"]] = fin["p_lgbm"]
    out = config.REPORTS_DIR / "submission.csv"
    sub.rename(columns={M.KEY: d["key_name"]}).to_csv(out, index=False)
    joblib.dump({k: fin[k] for k in ("prep_tree", "lgbm_models")}, config.MODELS_DIR / "model_lgbm.joblib")
    (config.REPORTS_DIR / "metrics_cli.json").write_text(json.dumps(res, indent=2))
    log.info("Submission escrita en %s", out)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tuning", action="store_true")
    main(ap.parse_args().skip_tuning)
