"""
PREVISÃO DE ATRASO

Este arquivo e usado depois que o modelo ja foi treinado.

A função dele e pegar o modelo salvo no disco e usar esse modelo para fazer uma previsão para um voo informado pela interface.
    - app.py recebe os dados do usuário;
    - predict.py transforma esses dados no formato certo;
    - o modelo treinado calcula a probabilidade de atraso;
    - predict.py devolve um resultado pronto para a interface mostrar.
"""

"""
IMPORTS

joblib
    Carrega o modelo treinado.

pandas
    Cria um DataFrame com os dados de um único voo.
"""
import joblib
import pandas as pd

try:
    from src.config import FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH
    from src.utils import (
        build_explanation,
        classify_period_of_day,
        load_json,
        practical_recommendation,
        probability_to_risk_level,
    )
except ImportError:
    from config import FEATURE_COLUMNS, METADATA_PATH, MODEL_PATH
    from utils import (
        build_explanation,
        classify_period_of_day,
        load_json,
        practical_recommendation,
        probability_to_risk_level,
    )


def load_model_and_metadata():
    """
    CARREGAMENTO DO MODELO E DOS METADADOS

    Esta função abre dois arquivos gerados pelo treinamento:
        - modelo_atraso_voos.pkl - Contem o Pipeline treinado.
        - metadados_interface.json - Contem informações auxiliares para a interface.
    """
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Modelo nao encontrado. Treinamento ainda não realizado"
        )

    model = joblib.load(MODEL_PATH)
    metadata = load_json(METADATA_PATH)

    return model, metadata


def prepare_input(
    airline,
    airport_from,
    airport_to,
    day_of_week,
    departure_hour,
    departure_minute,
    length,
):
    """
    PREPARAÇÃO DA ENTRADA DO USUÁRIO
    --------------------------------

    O usuário preenche campos na interface a função transforma o input em um dicionario representando um único voo.
    """
    # O usuário informa hora e minuto separados pela interface e o original usa Time como minutos desde meia-noite.
    time_in_minutes = int(departure_hour) * 60 + int(departure_minute)

    # Cria a mesma variavel PeriodOfDay usada no treinamento.
    period_of_day = classify_period_of_day(int(departure_hour))

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


def predict_delay(input_data):
    """
    PREVISÃO DO ATRASO

    Esta função recebe os dados ja preparados de um voo e pergunta ao modelo qual e a probabilidade deste voo pertencer a classe 1
        - classe 0 significa menor risco/sem atraso
        - classe 1 significa atraso

    Depois a probabilidade e transformada em nível de risco.

    A função devolve um dicionario com tudo que o app.py precisa exibir.
    """
    model, metadata = load_model_and_metadata()

    # Transforma o dicionario de um voo em DataFrame.
    input_dataframe = pd.DataFrame([input_data])[FEATURE_COLUMNS]

    # Devolve a probabilidade de cada classe.
    probability_delay = float(model.predict_proba(input_dataframe)[0][1])

    # Transforma a probabilidade em classe.
    predicted_class = int(probability_delay >= 0.50)

    # Transforma a probabilidade em texto: Baixo, Moderado ou Alto.
    risk_level = probability_to_risk_level(probability_delay)

    # Retorna tudo que a interface precisa mostrar.
    return {
        "classe_prevista": predicted_class,
        "probabilidade_atraso": probability_delay,
        "nivel_risco": risk_level,
        "mensagem": (
            "Risco de atraso identificado."
            if predicted_class == 1
            else "Risco de atraso menor identificado."
        ),
        "recomendacao": practical_recommendation(risk_level),
        "explicacao": build_explanation(input_data, probability_delay, risk_level),
        "melhor_modelo": metadata.get("melhor_modelo", "modelo treinado"),
    }
