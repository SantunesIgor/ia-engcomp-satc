"""
CONFIGURAÇÕES DO PROJETO

Este arquivo guarda valores fixos que sao usados em varias partes do sistema.
"""

"""
IMPORTS

pathlib.Path
    Ferramenta usada para montar caminhos de arquivos e pastas.
"""

"""
CONSTANTES
As dizem alguns valores sobre as colunas do dataset e os caminhos de onde estão os arquivos de entrada e saída.
    - raiz do projeto;
    - caminho do dataset;
    - pasta de modelos;
    - pasta de relatórios;
    - caminho do modelo salvo;
    - caminho dos metadados;
    - coluna alvo;
    - colunas removidas;
    - colunas esperadas no CSV;
    - features categóricas;
    - features numéricas;
    - nomes dos dias da semana.
"""
from pathlib import Path

# Caminhos do projeto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "Airlines.csv"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODELS_DIR / "modelo_atraso_voos.pkl"

METADATA_PATH = MODELS_DIR / "metadados_interface.json"

RESULTS_PATH = REPORTS_DIR / "resultados_modelos.csv"

CONFUSION_MATRIX_PATH = REPORTS_DIR / "matriz_confusao_melhor_modelo.png"

# Colunas do dataset
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

# Definição das features do modelo
CATEGORICAL_FEATURES = [
    "Airline",
    "AirportFrom",
    "AirportTo",
    "DayOfWeek",
    "PeriodOfDay",
]

NUMERIC_FEATURES = ["Time", "Length", "DepartureHour"]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Tratamento dos dias da semana pelo app.py
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
