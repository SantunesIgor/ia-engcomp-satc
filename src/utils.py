"""
FUNÇÕES UTILITÁRIAS

Este arquivo guarda funções auxiliares usadas em varias partes do projeto.

1. validar se o dataset tem as colunas certas;
2. limpar e filtrar os dados;
3. criar novas variáveis de horário;
4. transformar probabilidade em texto para o usuário;
5. salvar e carregar arquivos JSON.
"""

"""
IMPORTS

json
    Biblioteca usada para trabalhar com arquivos JSON.

pandas
    Biblioteca usada para manipular tabelas de dados.

src.config
    Fornece EXPECTED_COLUMNS e TARGET_COLUMN.

src.display_names
    Fornece listas de códigos permitidos e funções para normalizar/mostrar nomes.
"""
import json
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
    """
    VALIDAÇÃO DAS COLUNAS DO DATASET

    Esta função verifica se todas as colunas esperadas existem.
    Se alguma coluna estiver faltando, a função interrompe o programa com ValueError e mostra quais colunas estão ausentes.
    """
    missing_columns = [
        column for column in EXPECTED_COLUMNS if column not in dataframe.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(
            "O arquivo nao possui todas as colunas esperadas. "
            f"Colunas ausentes: {missing_text}"
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError("A coluna alvo nao foi encontrada no arquivo.")


def convert_time_to_hour(time_value):
    """
    CONVERSÃO DE MINUTOS PARA HORA

    No dataset, a coluna Time representa o horário do voo em minutos desde meia-noite.
    Para criar uma feature mais simples, transformamos esse valor em hora inteira.
    """
    try:
        minutes = int(float(time_value))
    except (TypeError, ValueError):
        return 0

    minutes = max(0, min(minutes, 1439))
    return minutes // 60


def classify_period_of_day(hour):
    """
    CLASSIFICAÇÃO DO PERÍODO DO DIA

    Esta função transforma uma hora numérica em uma categoria textual.
        - 0 a 5: Madrugada;
        - 6 a 11: Manha;
        - 12 a 17: Tarde;
        - 18 a 23: Noite.

    """
    if 0 <= hour <= 5:
        return "Madrugada"
    if 6 <= hour <= 11:
        return "Manha"
    if 12 <= hour <= 17:
        return "Tarde"
    return "Noite"


def add_time_features(dataframe):
    """
    CRIAÇÃO DE FEATURES DE HORÁRIO

    Esta função recebe um DataFrame e adiciona novas colunas relacionadas ao horário do voo.
        - DepartureHour;
        - PeriodOfDay.
    """
    dataframe = dataframe.copy()

    dataframe["DepartureHour"] = dataframe["Time"].apply(convert_time_to_hour)

    dataframe["PeriodOfDay"] = dataframe["DepartureHour"].apply(classify_period_of_day)

    return dataframe


def _apply_filter(dataframe, mask, report, key):
    """
    APLICAÇÃO DE FILTRO COM RELATÓRIO

    Ela aplica um filtro no DataFrame e registra quantas linhas foram removidas.
    O parâmetro mask e uma serie de valores True (mantém a linha) ou False (remove a linha).
    O parâmetro report e um dicionario onde salvamos os números da limpeza.
    """
    before = len(dataframe)

    dataframe = dataframe.loc[mask].copy()

    report[key] = int(before - len(dataframe))

    return dataframe


def clean_dataset(dataframe, columns_to_remove):
    """
    LIMPEZA DO DATASET
    Ela recebe o dataset bruto e devolve uma versão limpa, padronizada e pronta para o treinamento.
    """
    dataframe = dataframe.copy()

    report = {"registros_originais": int(len(dataframe))}

    # Remove espaços extras de textos.
    for column in dataframe.select_dtypes(include="object").columns:
        dataframe[column] = dataframe[column].astype(str).str.strip()

    # Normaliza códigos em maiúsculas
    for column in ["Airline", "AirportFrom", "AirportTo"]:
        dataframe[column] = dataframe[column].apply(normalize_code)

    # Converte a coluna alvo para numérico e classifica NaN.
    dataframe[TARGET_COLUMN] = pd.to_numeric(dataframe[TARGET_COLUMN], errors="coerce")

    # Aplica o filtro para retirar as colunas em que a variável alvo é NaN.
    dataframe = _apply_filter(
        dataframe,
        dataframe[TARGET_COLUMN].isin([0, 1]),
        report,
        "removidos_delay_invalido",
    )

    # Transforma a coluna alvo em inteiro.
    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(int)

    # Converte todas as colunas numéricas e classifica NaN.
    for column in ["Time", "Length", "DayOfWeek"]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    # Mantém apenas companhias que a interface sabe nomear.
    dataframe = _apply_filter(
        dataframe,
        dataframe["Airline"].isin(ALLOWED_AIRLINE_CODES),
        report,
        "removidos_companhia_aerea_nao_identificada",
    )

    # Mantém apenas aeroportos de origem que a interface sabe nomear.
    dataframe = _apply_filter(
        dataframe,
        dataframe["AirportFrom"].isin(ALLOWED_AIRPORT_CODES),
        report,
        "removidos_aeroporto_origem_nao_identificado",
    )

    # Mantém apenas aeroportos de destino que a interface sabe nomear.
    dataframe = _apply_filter(
        dataframe,
        dataframe["AirportTo"].isin(ALLOWED_AIRPORT_CODES),
        report,
        "removidos_aeroporto_destino_nao_identificado",
    )

    # Remove rotas impossíveis em que origem e destino sao iguais.
    dataframe = _apply_filter(
        dataframe,
        dataframe["AirportFrom"] != dataframe["AirportTo"],
        report,
        "removidos_origem_igual_destino",
    )

    # Mantém apenas dias da semana no intervalo usado pelo dataset: 1 a 7.
    dataframe = _apply_filter(
        dataframe,
        dataframe["DayOfWeek"].between(1, 7),
        report,
        "removidos_dia_semana_invalido",
    )

    # Mantém apenas horários validos em minutos desde meia-noite.
    dataframe = _apply_filter(
        dataframe,
        dataframe["Time"].between(0, 1439),
        report,
        "removidos_horario_invalido",
    )

    # Mantém apenas durações plausíveis para voos, em minutos.
    dataframe = _apply_filter(
        dataframe,
        dataframe["Length"].between(20, 800),
        report,
        "removidos_duracao_invalida",
    )

    # Converte dia da semana, horário e duração para inteiro apos remover inválidos.
    dataframe["DayOfWeek"] = dataframe["DayOfWeek"].astype(int)
    dataframe["Time"] = dataframe["Time"].astype(int)
    dataframe["Length"] = dataframe["Length"].astype(int)

    # Remove colunas que nao serão usadas como features.
    for column in columns_to_remove:
        if column in dataframe.columns:
            dataframe = dataframe.drop(columns=column)

    # Adiciona as features derivadas de horario.
    dataframe = add_time_features(dataframe)

    # Registra o total final de linhas apos todos os filtros.
    report["registros_apos_limpeza"] = int(len(dataframe))

    # Registra o total removido do inicio ao fim.
    report["registros_removidos_total"] = int(
        report["registros_originais"] - len(dataframe)
    )

    # Registra quais companhias continuaram no dataset limpo.
    report["companhias_aereas_mantidas"] = sorted_unique_values(dataframe["Airline"])

    # Registra quais aeroportos continuaram, considerando origem e destino.
    report["aeroportos_mantidos"] = sorted(
        set(sorted_unique_values(dataframe["AirportFrom"]))
        | set(sorted_unique_values(dataframe["AirportTo"]))
    )

    # Armazena o relatório dentro dos atributos do DataFrame para o treinamento recuperar depois.
    dataframe.attrs["cleaning_report"] = report

    return dataframe


def probability_to_risk_level(probability):
    """
    CONVERSÃO DE PROBABILIDADE EM NÍVEL DE RISCO

    O modelo devolve uma probabilidade numérica e transformamos a probabilidade em:
        - Baixo;
        - Moderado;
        - Alto.
    """
    if probability < 0.40:
        return "Baixo"

    if probability < 0.60:
        return "Moderado"

    return "Alto"


def practical_recommendation(risk_level):
    """
    RECOMENDAÇÃO PRATICA

    Esta função transforma o nível de risco em uma orientação simples para o usuário.
    """
    if risk_level == "Baixo":
        return "Mantenha o planejamento normal, mas acompanhe os canais oficiais da companhia aérea."

    if risk_level == "Moderado":
        return (
            "Considere sair com alguma antecedência e evite conexões muito apertadas."
        )

    return "Planeje uma margem maior de tempo, acompanhe o voo com frequência e avise pessoas envolvidas no deslocamento."


def build_explanation(input_data, probability, risk_level):
    """
    EXPLICAÇÃO DA PREVISÃO

    Esta função cria uma explicação textual simples, dizendo quais informações foram consideradas na previsão.
    """
    hour = int(input_data.get("DepartureHour", 0))

    minute = int(input_data.get("Time", 0)) % 60

    period = input_data.get("PeriodOfDay", "periodo informado")

    origin = airport_label(input_data.get("AirportFrom", "origem informada"))

    destination = airport_label(input_data.get("AirportTo", "destino informado"))

    airline = airline_label(input_data.get("Airline", "companhia informada"))

    length = int(input_data.get("Length", 0))

    return (
        f"O sistema estimou risco {risk_level.lower()} porque o modelo encontrou, nos dados históricos, "
        f"padrões associados a companhia aérea {airline}, a rota {origin} -> {destination}, "
        f"ao horário de saída por volta de {hour:02d}:{minute:02d} ({period.lower()}) e a duração prevista de {length} minutos. "
        f"A probabilidade calculada foi de {probability:.1%}. Essa previsão estima risco e nao garante que o voo ira atrasar."
    )


def save_json(data, path):
    """
    SALVAMENTO DE JSON

    No projeto, usamos JSON para salvar metadados da interface:
        - melhor modelo;
        - métricas;
        - companhias disponíveis;
        - aeroportos disponíveis;
        - relatório de limpeza.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_json(path):
    """
    LEITURA DE JSON

    Esta função abre um arquivo JSON e transforma o conteúdo em objetos Python.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def sorted_unique_values(series):
    """
    VALORES ÚNICOS ORDENADOS

    Esta função recebe uma coluna do pandas e devolve uma lista ordenada com os valores únicos.
    Ela remove valores ausentes antes de montar a lista.
    """
    return sorted(series.dropna().unique().tolist())
