import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run_rcc, run_static

if __name__ == "__main__":
    run_static('se')
