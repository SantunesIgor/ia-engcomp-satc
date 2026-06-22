from typing import Dict

import joblib
import pandas as pd

# Este arquivo faz a previsao depois que o modelo ja foi treinado.
# Ele e usado pela interface Streamlit em app.py.

try:
    # FEATURE_COLUMNS garante que a previsao use as mesmas colunas do treinamento.
    # MODEL_PATH aponta para o arquivo .pkl do modelo.
    # METADATA_PATH aponta para o JSON com informacoes da interface.
    from src.config import FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH
    from src.utils import (
        build_explanation,
        classify_period_of_day,
        load_json,
        practical_recommendation,
        probability_to_risk_level,
    )
except ImportError:
    # Fallback para casos em que o arquivo seja executado diretamente.
    from config import FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH
    from utils import (
        build_explanation,
        classify_period_of_day,
        load_json,
        practical_recommendation,
        probability_to_risk_level,
    )


def load_model_and_metadata():
    """Carrega o modelo treinado e os metadados da interface."""
    # Se estes arquivos nao existem, significa que o treinamento ainda nao foi feito.
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Modelo nao encontrado. Execute primeiro: python src/train_model.py"
        )

    # joblib.load abre o modelo salvo em .pkl.
    # Esse modelo ja inclui o pre-processamento e o algoritmo escolhido.
    model = joblib.load(MODEL_PATH)

    # Os metadados guardam listas de companhias, aeroportos, metricas e melhor modelo.
    metadata = load_json(METADATA_PATH)
    return model, metadata


def prepare_input(
    airline: str,
    airport_from: str,
    airport_to: str,
    day_of_week: int,
    departure_hour: int,
    departure_minute: int,
    length: int,
) -> Dict:
    """Transforma os campos da interface no formato que o modelo espera."""
    # O usuario informa hora e minuto separados pela interface.
    # O dataset original usa Time como minutos desde meia-noite.
    # Exemplo: 18:30 vira 18 * 60 + 30 = 1110.
    time_in_minutes = int(departure_hour) * 60 + int(departure_minute)

    # Cria a mesma variavel PeriodOfDay usada no treinamento.
    # Exemplo: 18 vira "Noite".
    period_of_day = classify_period_of_day(int(departure_hour))

    # Este dicionario representa um unico voo novo.
    # As chaves precisam bater com FEATURE_COLUMNS.
    return {
        "Airline": airline,
        "AirportFrom": airport_from,
        "AirportTo": airport_to,
        "DayOfWeek": int(day_of_week),
        "Time": time_in_minutes,
        "Length": int(length),
        "DepartureHour": int(departure_hour),
        "PeriodOfDay": period_of_day,
    }


def predict_delay(input_data: Dict) -> Dict:
    """Calcula a previsao de atraso e monta o resultado para a interface."""
    # Carrega o modelo treinado e informacoes salvas no treinamento.
    model, metadata = load_model_and_metadata()

    # Transforma o dicionario de um voo em DataFrame.
    # O scikit-learn espera receber dados nesse formato.
    # [FEATURE_COLUMNS] tambem garante a ordem correta das colunas.
    input_dataframe = pd.DataFrame([input_data])[FEATURE_COLUMNS]

    # predict_proba devolve a probabilidade de cada classe.
    # [0][1] significa: primeira linha, probabilidade da classe 1.
    # Classe 1 = voo com atraso.
    if hasattr(model, "predict_proba"):
        probability_delay = float(model.predict_proba(input_dataframe)[0][1])
    else:
        # Este fallback existe para modelos que nao oferecem predict_proba.
        probability_delay = float(model.predict(input_dataframe)[0])

    # Transforma a probabilidade em classe.
    # Se a chance for 50% ou mais, considera classe 1: atraso.
    predicted_class = int(probability_delay >= 0.50)

    # Transforma a probabilidade em texto: Baixo, Moderado ou Alto.
    risk_level = probability_to_risk_level(probability_delay)

    # Retorna tudo que a interface precisa mostrar.
    return {
        "classe_prevista": predicted_class,
        "probabilidade_atraso": probability_delay,
        "nivel_risco": risk_level,
        "mensagem": "Risco de atraso identificado." if predicted_class == 1 else "Risco de atraso menor identificado.",
        "recomendacao": practical_recommendation(risk_level),
        "explicacao": build_explanation(input_data, probability_delay, risk_level),
        "melhor_modelo": metadata.get("melhor_modelo", "modelo treinado"),
    }
