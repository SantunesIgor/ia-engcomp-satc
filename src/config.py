from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "Airlines.csv"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "modelo_atraso_voos.pkl"

METADATA_PATH = MODELS_DIR / "metadados_interface.json"

RESULTS_PATH = REPORTS_DIR / "resultados_modelos.csv"

CONFUSION_MATRIX_PATH = REPORTS_DIR / "matriz_confusao_melhor_modelo.png"

TARGET_COLUMN = "Delay"

COLUMNS_TO_REMOVE = ["id", "Flight"]

EXPECTED_COLUMNS = [
    "id",
    "Airline",
    "Flight",
    "AirportFrom",
    "AirportTo",
    "DayOfWeek",
    "Time",
    "Length",
    "Delay",
]

CATEGORICAL_FEATURES = ["Airline", "AirportFrom", "AirportTo", "DayOfWeek", "PeriodOfDay"]

NUMERIC_FEATURES = ["Time", "Length", "DepartureHour"]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

DAY_NAMES = {
    1: "Segunda-feira",
    2: "Terca-feira",
    3: "Quarta-feira",
    4: "Quinta-feira",
    5: "Sexta-feira",
    6: "Sabado",
    7: "Domingo",
}

DAY_NAME_TO_NUMBER = {name: number for number, name in DAY_NAMES.items()}
