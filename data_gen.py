from pathlib import Path

from model import generate_synthetic_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "generated"

if __name__ == "__main__":
    generate_synthetic_dataset(DATA_DIR)
    print(f"Generated synthetic dataset in {DATA_DIR}")
