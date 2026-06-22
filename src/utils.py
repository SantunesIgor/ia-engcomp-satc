import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

try:
    from src.config import EXPECTED_COLUMNS, TARGET_COLUMN
    from src.display_names import (
        ALLOWED_AIRLINE_CODES,
        ALLOWED_AIRPORT_CODES,
        airline_label,
        airport_label,
        normalize_code,
    )
except ImportError:
    from config import EXPECTED_COLUMNS, TARGET_COLUMN
    from display_names import (
        ALLOWED_AIRLINE_CODES,
        ALLOWED_AIRPORT_CODES,
        airline_label,
        airport_label,
        normalize_code,
    )


def validate_dataset_columns(dataframe):
    missing_columns = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            "O arquivo nao possui todas as colunas esperadas. "
            f"Colunas ausentes: {missing_text}"
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError("A coluna alvo nao foi encontrada no arquivo.")


def convert_time_to_hour(time_value):
    """Converte minutos desde meia-noite para hora inteira."""
    try:
        minutes = int(float(time_value))
    except (TypeError, ValueError):
        return 0

    minutes = max(0, min(minutes, 1439))
    return minutes // 60


def classify_period_of_day(hour):
    if 0 <= hour <= 5:
        return "Madrugada"
    if 6 <= hour <= 11:
        return "Manha"
    if 12 <= hour <= 17:
        return "Tarde"
    return "Noite"


def add_time_features(dataframe):
    dataframe = dataframe.copy()

    dataframe["DepartureHour"] = dataframe["Time"].apply(convert_time_to_hour)


    dataframe["PeriodOfDay"] = dataframe["DepartureHour"].apply(classify_period_of_day)
    return dataframe


def _apply_filter(dataframe, mask, report, key):
    before = len(dataframe)
    dataframe = dataframe.loc[mask].copy()

    report[key] = int(before - len(dataframe))
    return dataframe


def clean_dataset(dataframe, columns_to_remove):
    dataframe = dataframe.copy()
    report = {"registros_originais": int(len(dataframe))}

    for column in dataframe.select_dtypes(include="object").columns:
        dataframe[column] = dataframe[column].astype(str).str.strip()

    for column in ["Airline", "AirportFrom", "AirportTo"]:
        dataframe[column] = dataframe[column].apply(normalize_code)

    dataframe[TARGET_COLUMN] = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")
    dataframe = _apply_filter(
        dataframe,
        dataframe[TARGET_COLUMN].isin([0, 1]),
        report,
        "removidos_delay_invalido",
    )
    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(int)

    for column in ["Time", "Length", "DayOfWeek"]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    dataframe = _apply_filter(
        dataframe,
        dataframe["Airline"].isin(ALLOWED_AIRLINE_CODES),
        report,
        "removidos_companhia_aerea_nao_identificada",
    )

    dataframe = _apply_filter(
        dataframe,
        dataframe["AirportFrom"].isin(ALLOWED_AIRPORT_CODES),
        report,
        "removidos_aeroporto_origem_nao_identificado",
    )

    dataframe = _apply_filter(
        dataframe,
        dataframe["AirportTo"].isin(ALLOWED_AIRPORT_CODES),
        report,
        "removidos_aeroporto_destino_nao_identificado",
    )

    dataframe = _apply_filter(
        dataframe,
        dataframe["AirportFrom"] != dataframe["AirportTo"],
        report,
        "removidos_origem_igual_destino",
    )

    dataframe = _apply_filter(
        dataframe,
        dataframe["DayOfWeek"].between(1, 7),
        report,
        "removidos_dia_semana_invalido",
    )

    dataframe = _apply_filter(
        dataframe,
        dataframe["Time"].between(0, 1439),
        report,
        "removidos_horario_invalido",
    )

    dataframe = _apply_filter(
        dataframe,
        dataframe["Length"].between(20, 800),
        report,
        "removidos_duracao_invalida",
    )

    dataframe["DayOfWeek"] = dataframe["DayOfWeek"].astype(int)
    dataframe["Time"] = dataframe["Time"].astype(int)
    dataframe["Length"] = dataframe["Length"].astype(int)

    for column in columns_to_remove:
        if column in dataframe.columns:
            dataframe = dataframe.drop(columns=column)

    dataframe = add_time_features(dataframe)

    report["registros_apos_limpeza"] = int(len(dataframe))
    report["registros_removidos_total"] = int(report["registros_originais"] - len(dataframe))
    report["companhias_aereas_mantidas"] = sorted_unique_values(dataframe["Airline"])
    report["aeroportos_mantidos"] = sorted(
        set(sorted_unique_values(dataframe["AirportFrom"])) | set(sorted_unique_values(dataframe["AirportTo"]))
    )

    dataframe.attrs["cleaning_report"] = report
    return dataframe


def probability_to_risk_level(probability):
    if probability < 0.40:
        return "Baixo"
    if probability < 0.60:
        return "Moderado"
    return "Alto"


def practical_recommendation(risk_level):
    if risk_level == "Baixo":
        return "Mantenha o planejamento normal, mas acompanhe os canais oficiais da companhia aerea."
    if risk_level == "Moderado":
        return "Considere sair com alguma antecedencia e evite conexoes muito apertadas."
    return "Planeje uma margem maior de tempo, acompanhe o voo com frequencia e avise pessoas envolvidas no deslocamento."


def build_explanation(input_data, probability, risk_level):
    hour = int(input_data.get("DepartureHour", 0))
    minute = int(input_data.get("Time", 0)) % 60
    period = input_data.get("PeriodOfDay", "periodo informado")
    origin = airport_label(input_data.get("AirportFrom", "origem informada"))
    destination = airport_label(input_data.get("AirportTo", "destino informado"))
    airline = airline_label(input_data.get("Airline", "companhia informada"))
    length = int(input_data.get("Length", 0))

    return (
        f"O sistema estimou risco {risk_level.lower()} porque o modelo encontrou, nos dados historicos, "
        f"padroes associados a companhia aerea {airline}, a rota {origin} -> {destination}, "
        f"ao horario de saida por volta de {hour:02d}:{minute:02d} ({period.lower()}) e a duracao prevista de {length} minutos. "
        f"A probabilidade calculada foi de {probability:.1%}. Essa previsao estima risco e nao garante que o voo ira atrasar."
    )


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def sorted_unique_values(series):
    return sorted(series.dropna().unique().tolist())
