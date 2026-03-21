from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

AZT1D_DIR = RAW_DIR / "AZT1D" / "AZT1D 2025" / "CGM Records"
HUPA_DIR = RAW_DIR / "HUPA" / "HUPA-UCM Diabetes Dataset" / "Preprocessed"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
