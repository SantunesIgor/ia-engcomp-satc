"""
IMPORTS

re
    Biblioteca usada para expressoes regulares, usado para validar textos como "08:30 JSON.

datetime
    Importa datetime para converter uma string HH:MM em um objeto de horario.

Streamlit
    Biblioteca que cria a interface web do projeto.

src.display_names
    Fornece listas de códigos permitidos e funções para normalizar/mostrar nomes.
"""

import re
from datetime import datetime
import streamlit as st

from src.config import DAY_NAME_TO_NUMBER
from src.display_names import (
    ALLOWED_AIRLINE_CODES,
    ALLOWED_AIRPORT_CODES,
    airline_label,
    airport_label,
)
from src.predict import load_model_and_metadata, predict_delay, prepare_input

st.set_page_config(
    page_title="Meu voo vai atrasar?",
    layout="centered",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.1rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }
        .subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-bottom: 1rem;
        }
        .risk-box {
            padding: 1.2rem;
            border-radius: 0.9rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(0,0,0,0.08);
        }
        .risk-low {
            background-color: #dcfce7;
            color: #14532d;
        }
        .risk-medium {
            background-color: #fef9c3;
            color: #713f12;
        }
        .risk-high {
            background-color: #fee2e2;
            color: #7f1d1d;
        }
        .small-note {
            color: #6b7280;
            font-size: 0.92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def sorted_by_label(codes, label_function):
    """
    ORDENAÇÃO DAS OPÇÕES DA INTERFACE

    Esta função recebe códigos e devolve esses códigos ordenados pelo nome que aparece para o usuario.
    """
    # Normaliza cada código recebido, remove vazios e ordena pela label exibida ao usuario.
    return sorted(
        {str(code).strip().upper() for code in codes if str(code).strip()},
        key=label_function,
    )


def parse_time_strict(time_text):
    """
    VALIDAÇÃO DO HORÁRIO

    Esta função valida o horário digitado pelo usuário no formato HH:MM
    """
    clean_time = str(time_text).strip()

    if not re.fullmatch(r"\d{2}:\d{2}", clean_time):
        raise ValueError(
            "Informe o horário no formato HH:MM, usando dois dígitos para hora e dois dígitos para minuto. Exemplo: 08:30."
        )

    # Separa horas e minutos
    hour_text, minute_text = clean_time.split(":")
    hour = int(hour_text)
    minute = int(minute_text)

    # Rejeita horas fora de 00-23 ou minutos fora de 00-59.
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("Informe um horário válido entre 00:00 e 23:59.")

    # Retorna um objeto time
    return datetime.strptime(clean_time, "%H:%M").time()


def parse_duration_strict(duration_text):
    """
    VALIDAÇÃO DA DURAÇÃO DO VOO

    Esta função valida a duração prevista do voo.
    O usuário precisa informar apenas números, representando minutos.

    Também limita a duração entre 20 e 800 minutos.
    """
    clean_duration = str(duration_text).strip()

    if not re.fullmatch(r"\d+", clean_duration):
        raise ValueError(
            "Informe a duração prevista do voo usando apenas números. Exemplo: 120."
        )

    # Converte a duração textual para inteiro.
    duration = int(clean_duration)

    # Mantém somente durações plausíveis.
    if duration < 20 or duration > 800:
        raise ValueError("Informe uma duração prevista entre 20 e 800 minutos.")

    # Retorna a duração validada em minutos.
    return duration


# Início da parte gráfica.
st.markdown(
    '<div class="main-title">Meu voo vai atrasar?</div>', unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Assistente inteligente para estimar o risco de atraso em voos com base em padrões históricos.</div>',
    unsafe_allow_html=True,
)

# Tenta carregar o modelo treinado e os metadados salvos pelo script de treinamento.
try:
    _, metadata = load_model_and_metadata()
except FileNotFoundError:
    st.error("O modelo ainda não foi treinado.")
    st.stop()

st.divider()
st.subheader("Informe os dados do voo")

# Monta as opções de companhias, origem e destino usando apenas códigos permitidos e presentes nos metadados.
airline_options = sorted_by_label(
    [
        code
        for code in metadata["companhias_aereas"]
        if str(code).strip().upper() in ALLOWED_AIRLINE_CODES
    ],
    airline_label,
)
origin_options = sorted_by_label(
    [
        code
        for code in metadata["aeroportos_origem"]
        if str(code).strip().upper() in ALLOWED_AIRPORT_CODES
    ],
    airport_label,
)
destination_all_options = sorted_by_label(
    [
        code
        for code in metadata["aeroportos_destino"]
        if str(code).strip().upper() in ALLOWED_AIRPORT_CODES
    ],
    airport_label,
)

# Se alguma lista ficou vazia, nao ha dados suficientes para previsão pela interface.
if not airline_options or not origin_options or not destination_all_options:
    st.error("Nao ha opções suficientes apos a limpeza dos dados.")
    st.stop()

# Campo de seleção da companhia aérea.
airline = st.selectbox(
    "Companhia aérea",
    airline_options,
    format_func=airline_label,
)

# Cria duas colunas para origem e destino.
col_origin, col_destination = st.columns(2)

with col_origin:
    # Campo de seleção do aeroporto de origem.
    airport_from = st.selectbox(
        "Aeroporto de origem",
        origin_options,
        format_func=airport_label,
    )

# Remove da lista de destinos o aeroporto que foi escolhido como origem.
destination_options = [code for code in destination_all_options if code != airport_from]

with col_destination:
    # Caso a limpeza deixe nenhuma opção de destino diferente da origem, mostra erro.
    if not destination_options:
        st.error(
            "Nao ha aeroporto de destino disponível diferente da origem escolhida."
        )
        airport_to = None
    else:
        # Campo de seleção do aeroporto de destino.
        airport_to = st.selectbox(
            "Aeroporto de destino",
            destination_options,
            format_func=airport_label,
        )

# Campo de seleção do dia da semana.
day_name = st.selectbox("Dia da semana", list(DAY_NAME_TO_NUMBER.keys()))

# Cria duas colunas para horário e duração.
col_time, col_length = st.columns(2)

with col_time:
    # Campo textual para obrigar o usuário a informar o formato HH:MM.
    departure_time_text = st.text_input(
        "Horario previsto de saida",
        value="08:00",
        placeholder="08:00",
    )

with col_length:
    # Campo textual para validar manualmente se a duração contem apenas números.
    length_text = st.text_input(
        "Duracao prevista do voo em minutos",
        value="120",
        placeholder="120",
    )

# Botão principal que dispara o calculo.
submitted = st.button(
    "Calcular risco de atraso", use_container_width=True, type="primary"
)

if submitted:
    # Proteções contra inputs errados.
    if airport_to is None:
        st.stop()

    if airport_from == airport_to:
        st.error(
            "O aeroporto de origem e o aeroporto de destino não podem ser iguais."
            "Escolha aeroportos diferentes para calcular o risco."
        )
        st.stop()

    try:
        departure_time = parse_time_strict(departure_time_text)
        length = parse_duration_strict(length_text)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    # Converte os dados do formulário para o dicionário de features esperado pelo modelo.
    input_data = prepare_input(
        airline=airline,
        airport_from=airport_from,
        airport_to=airport_to,
        day_of_week=DAY_NAME_TO_NUMBER[day_name],
        departure_hour=departure_time.hour,
        departure_minute=departure_time.minute,
        length=length,
    )

    # Executa a previsão e recebe classe, probabilidade, risco e textos explicativos.
    result = predict_delay(input_data)

    probability = result["probabilidade_atraso"]
    risk_level = result["nivel_risco"]

    # Troca as classes CSS para um visual mais vinculado com o risco.
    css_class = "risk-low"
    if risk_level == "Moderado":
        css_class = "risk-medium"
    elif risk_level == "Alto":
        css_class = "risk-high"

    # Mostra o resultado principal em uma caixa destacada com cor baseada no risco.
    st.markdown(
        f"""
        <div class="risk-box {css_class}">
            <h3>Risco de atraso: {risk_level}</h3>
            <p><strong>Probabilidade estimada de atraso:</strong> {probability:.1%}</p>
            <p><strong>Mensagem:</strong> {result['mensagem']}</p>
            <p><strong>Recomendação pratica:</strong> {result['recomendacao']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Explicação da decisão")
    st.write(result["explicacao"])
    st.caption(f"Modelo utilizado: {result['melhor_modelo']}")
