from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = Path(os.environ.get("IB_RAW_DIR", ROOT / "data" / "raw"))
PROC_DIR = Path(os.environ.get("IB_PROC_DIR", ROOT / "data" / "processed"))
REPORTS_DIR = Path(os.environ.get("IB_REPORTS_DIR", ROOT / "reports"))
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = Path(os.environ.get("IB_MODELS_DIR", ROOT / "models"))

for _d in (PROC_DIR, REPORTS_DIR, FIG_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

REF_MONTH = {"train": 201802, "test": 201902}

WINDOWS = (3, 6, 12)
TOP_K = 12
MAX_CARDINALITY_CATEGORICAL = 60

FILES = {
    "y_train": "y_train.csv",
    "sample_submission": "sample_submission.csv",
    "productos": "productos.csv",
    "rcc_train": "rcc_train.csv",
    "rcc_test": "rcc_test.csv",
    "sunat_train": "sunat_train.csv",
    "sunat_test": "sunat_test.csv",
    "se_train": "se_train.csv",
    "se_test": "se_test.csv",
    "censo_train": "censo_train.csv",
    "censo_test": "censo_test.csv",
}

ALIASES = {
    "key": ["key_value", "id_persona", "id_cliente", "cliente_id", "id"],
    "target": ["target", "codtarget", "default", "y", "label"],
    "period": ["codmes", "cod_mes", "periodo", "mes", "fecha_mes"],
    "amount": [
        "mto_saldo", "saldo", "saldo_deuda", "monto", "mto_deuda",
        "riesgo_directo", "deuda", "importe", "mto",
    ],
    "institution": [
        "cod_banco", "cod_instit_financiera", "cod_institucion",
        "institucion", "entidad", "banco", "cod_entidad",
    ],
    "product": [
        "producto", "cod_producto", "tipo_producto", "tipo_credito",
        "cod_tipo_credito", "id_producto",
    ],
    "clasif": [
        "clasif", "cod_clasificacion_deudor", "clasificacion",
        "calificacion", "cod_clasificacion", "clasif_deudor",
    ],
    "mora": ["rango_mora", "dias_mora", "mora", "rango_dias_mora", "cond_mora", "condicion"],
    "product_desc": [
        "descripcion", "desc_producto", "nombre_producto", "producto_desc",
        "nombre", "producto", "des_producto",
    ],
}

COLUMN_OVERRIDES: dict[str, dict[str, str]] = {}

N_SPLITS = 5
N_REPEATS = 2
HOLDOUT_SIZE = 0.20
OPTUNA_TRIALS = 40
COST_FN_OVER_FP = 5.0
