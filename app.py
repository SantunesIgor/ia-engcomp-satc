import re
from datetime import datetime

import streamlit as st

from src.config import DAY_NAME_TO_NUMBER
from src.display_names import ALLOWED_AIRLINE_CODES, ALLOWED_AIRPORT_CODES, airline_label, airport_label
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
    return sorted(
        {str(code).strip().upper() for code in codes if str(code).strip()},
        key=label_function,
    )

def parse_time_strict(time_text):
    clean_time = str(time_text).strip()

    if not re.fullmatch(r"\d{2}:\d{2}", clean_time):
        raise ValueError(
            "Informe o horario no formato HH:MM, usando dois digitos para hora e dois digitos para minuto. Exemplo: 08:30."
        )

    hour_text, minute_text = clean_time.split(":")
    hour = int(hour_text)
    minute = int(minute_text)

    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("Informe um horario valido entre 00:00 e 23:59.")

    return datetime.strptime(clean_time, "%H:%M").time()

def parse_duration_strict(duration_text):
    clean_duration = str(duration_text).strip()

    if not re.fullmatch(r"\d+", clean_duration):
        raise ValueError("Informe a duracao prevista do voo usando apenas numeros. Exemplo: 120.")

    duration = int(clean_duration)

    if duration < 20 or duration > 800:
        raise ValueError("Informe uma duracao prevista entre 20 e 800 minutos.")

    return duration

st.markdown('<div class="main-title">Meu voo vai atrasar?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Assistente inteligente para estimar o risco de atraso em voos com base em padroes historicos.</div>',
    unsafe_allow_html=True,
)

try:
    _, metadata = load_model_and_metadata()
except FileNotFoundError:
    st.error(
        "O modelo ainda nao foi treinado."
    )
    st.stop()

st.divider()
st.subheader("Informe os dados do voo")


airline_options = sorted_by_label(
    [code for code in metadata["companhias_aereas"] if str(code).strip().upper() in ALLOWED_AIRLINE_CODES],
    airline_label,
)
origin_options = sorted_by_label(
    [code for code in metadata["aeroportos_origem"] if str(code).strip().upper() in ALLOWED_AIRPORT_CODES],
    airport_label,
)
destination_all_options = sorted_by_label(
    [code for code in metadata["aeroportos_destino"] if str(code).strip().upper() in ALLOWED_AIRPORT_CODES],
    airport_label,
)

if not airline_options or not origin_options or not destination_all_options:
    st.error(
        "Nao ha opcoes suficientes apos a limpeza dos dados."
    )
    st.stop()

airline = st.selectbox(
    "Companhia aerea",
    airline_options,
    format_func=airline_label,
)

col_origin, col_destination = st.columns(2)

with col_origin:
    airport_from = st.selectbox(
        "Aeroporto de origem",
        origin_options,
        format_func=airport_label,
    )

destination_options = [code for code in destination_all_options if code != airport_from]

with col_destination:
    if not destination_options:
        st.error("Nao ha aeroporto de destino disponivel diferente da origem escolhida.")
        airport_to = None
    else:
        airport_to = st.selectbox(
            "Aeroporto de destino",
            destination_options,
            format_func=airport_label,
        )

day_name = st.selectbox("Dia da semana", list(DAY_NAME_TO_NUMBER.keys()))

col_time, col_length = st.columns(2)

with col_time:
    departure_time_text = st.text_input(
        "Horario previsto de saida",
        value="08:00",
        placeholder="08:00",
    )

with col_length:
    length_text = st.text_input(
        "Duracao prevista do voo em minutos",
        value="120",
        placeholder="120",
    )

submitted = st.button("Calcular risco de atraso", use_container_width=True, type="primary")

if submitted:
    if airport_to is None:
        st.stop()

    if airport_from == airport_to:
        st.error(
            "O aeroporto de origem e o aeroporto de destino nao podem ser iguais."
            "Escolha aeroportos diferentes para calcular o risco."
        )
        st.stop()

    try:
        departure_time = parse_time_strict(departure_time_text)
        length = parse_duration_strict(length_text)
    except ValueError as error:
        st.error(str(error))
        st.stop()

    input_data = prepare_input(
        airline=airline,
        airport_from=airport_from,
        airport_to=airport_to,
        day_of_week=DAY_NAME_TO_NUMBER[day_name],
        departure_hour=departure_time.hour,
        departure_minute=departure_time.minute,
        length=length,
    )

    result = predict_delay(input_data)
    probability = result["probabilidade_atraso"]
    risk_level = result["nivel_risco"]

    css_class = "risk-low"
    if risk_level == "Moderado":
        css_class = "risk-medium"
    elif risk_level == "Alto":
        css_class = "risk-high"

    st.markdown(
        f"""
        <div class="risk-box {css_class}">
            <h3>Risco de atraso: {risk_level}</h3>
            <p><strong>Probabilidade estimada de atraso:</strong> {probability:.1%}</p>
            <p><strong>Mensagem:</strong> {result['mensagem']}</p>
            <p><strong>Recomendacao pratica:</strong> {result['recomendacao']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Explicacao da decisao")
    st.write(result["explicacao"])
    st.caption(f"Modelo utilizado: {result['melhor_modelo']}")

st.divider()

with st.expander("Como interpretar o resultado"):
    st.write(
        "A probabilidade mostra a estimativa do modelo para o voo informado. "
        "Risco baixo indica menor chance historica de atraso. Risco moderado indica atencao. "
        "Risco alto indica que os padroes historicos do voo informado se parecem mais com casos de atraso."
    )

with st.expander("Limitacoes do sistema"):
    st.write(
        "O sistema usa apenas os dados disponiveis no dataset historico. "
        "Ele nao considera clima em tempo real, manutencao da aeronave, decisoes operacionais recentes, "
        "trafego aereo no momento, greves, fechamento de aeroportos ou informacoes oficiais atualizadas. "
        "Por isso, o resultado deve ser usado como apoio ao planejamento, nao como garantia."
    )
