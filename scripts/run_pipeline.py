import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-only", action="store_true")
    ap.add_argument("--skip-tuning", action="store_true", help="usa hiperparámetros por defecto en el entrenamiento final")
    a = ap.parse_args()
    pipeline.run_all()
    if not a.features_only:
        from train_final import main as train_main
        train_main(skip_tuning=a.skip_tuning)
