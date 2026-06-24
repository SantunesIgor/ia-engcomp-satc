"""
TREINAMENTO DO MODELO

Este arquivo e responsável por ensinar o computador a reconhecer padrões de atraso em voos.

Para executar:
    python src/train_model.py

o programa le o arquivo data/Airlines.csv, limpa os dados, separa uma parte para treino e outra para teste, treina alguns modelos
de Machine Learning e escolhe o melhor deles.
"""

"""
IMPORTS

warnings
    Biblioteca usada para controlar avisos do Python.
    Alguns avisos aparecem durante o treinamento, mas nem sempre indicam erro.
    Neste projeto, escondemos avisos que nao impedem o funcionamento para deixar
    o terminal mais limpo.

joblib
    Biblioteca usada para salvar e carregar objetos Python.
    Aqui, ela salva o modelo treinado em models/ .
    Isso evita treinar tudo novamente sempre que a interface for usada.

matplotlib.pyplot
    Biblioteca usada para gerar gráficos e imagens.
    Neste projeto, ela salva a matriz de confusão, que mostra acertos e erros do modelo.

pandas
    Biblioteca usada para trabalhar com dados CSV em formato de tabela.

IMPORTS DO SCIKIT-LEARN

ColumnTransformer
    Aplica transformações diferentes em grupos diferentes de colunas.
    No projeto, colunas de texto passam por OneHotEncoder e colunas numéricas passam por StandardScaler.

RandomForestClassifier
    Modelo de classificação formado por varias arvores de decisão.

LogisticRegression
    Modelo de classificação mais simples e linear.

DecisionTreeClassifier
    Modelo baseado em uma única arvore de decisão.

Métricas
    As métricas medem se o modelo esta indo bem.

    accuracy_score:
        Mede a porcentagem geral de acertos.

    precision_score:
        Mede, entre os voos previstos como atrasados, quantos realmente atrasaram.

    recall_score:
        Mede, entre os voos que realmente atrasaram, quantos o modelo encontrou.

    f1_score:
        Equilibra precisão e recall. Foi a métrica usada para escolher o melhor modelo.

    confusion_matrix:
        Mostra acertos e erros em forma de tabela.

    ConfusionMatrixDisplay:
        Transforma a matriz de confusão em gráfico/imagem.

train_test_split
    Separa os dados em treino e teste.

Pipeline
    Junta varias etapas em uma sequencia única.

OneHotEncoder
    Transforma categorias em números.

StandardScaler
    Padroniza colunas numéricas para uma escala mais equilibrada.
    Isso ajuda principalmente modelos como Regressão Logística.
"""

"""
FLUXO

1. Carrega o CSV.
2. Valida se as colunas existem.
3. Limpa dados invalidos.
4. Cria novas features de horario.
5. Separa entrada X e resposta y.
6. Divide em treino e teste.
7. Treina tres modelos.
8. Calcula metricas.
9. Escolhe o melhor pelo F1-score.
10. Salva modelo, metadados, metricas e matriz de confusao.
"""
import warnings
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from src.config import (
        CATEGORICAL_FEATURES,
        COLUMNS_TO_REMOVE,
        CONFUSION_MATRIX_PATH,
        DATA_PATH,
        FEATURE_COLUMNS,
        METADATA_PATH,
        MODEL_PATH,
        MODELS_DIR,
        NUMERIC_FEATURES,
        REPORTS_DIR,
        RESULTS_PATH,
        TARGET_COLUMN,
    )
    from src.display_names import ALLOWED_AIRLINE_CODES, ALLOWED_AIRPORT_CODES
    from src.utils import (
        clean_dataset,
        save_json,
        sorted_unique_values,
        validate_dataset_columns,
    )
except ImportError:
    from config import (
        CATEGORICAL_FEATURES,
        COLUMNS_TO_REMOVE,
        CONFUSION_MATRIX_PATH,
        DATA_PATH,
        FEATURE_COLUMNS,
        METADATA_PATH,
        MODEL_PATH,
        MODELS_DIR,
        NUMERIC_FEATURES,
        REPORTS_DIR,
        RESULTS_PATH,
        TARGET_COLUMN,
    )
    from display_names import ALLOWED_AIRLINE_CODES, ALLOWED_AIRPORT_CODES
    from utils import (
        clean_dataset,
        save_json,
        sorted_unique_values,
        validate_dataset_columns,
    )

# Evita que avisos de compatibilidade.
warnings.filterwarnings("ignore", category=UserWarning)


def build_pipeline(model):
    """
    CONSTRUÇÃO DO PIPELINE

    Neste projeto, o pipeline representa o caminho que os dados percorrem ate chegar na previsão:
        -> dados originais
        -> pré-processamento das colunas categóricas
        -> pré-processamento das colunas numéricas
        -> modelo de Machine Learning
        -> previsão final

    Os mesmos tratamentos aplicados no treinamento são aplicados quando a interface fizer uma nova previsão.
    """
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )

    numeric_transformer = StandardScaler()

    # Define quais transformações serão aplicadas em quais colunas.
    preprocessor = ColumnTransformer(
        transformers=[
            ("variaveis_categoricas", categorical_transformer, CATEGORICAL_FEATURES),
            ("variaveis_numericas", numeric_transformer, NUMERIC_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("pre_processamento", preprocessor),
            ("modelo", model),
        ]
    )


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    """
    AVALIAÇÃO DE UM MODELO

    Esta função recebe um algoritmo de Machine Learning, treina esse algoritmo e mede o desempenho dele.
    """
    pipeline = build_pipeline(model)

    pipeline.fit(x_train, y_train)

    predictions = pipeline.predict(x_test)

    metrics = {
        "modelo": name,
        "acuracia": accuracy_score(y_test, predictions),
        "precisao": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
    }
    return pipeline, metrics, predictions


def main():
    """
    FUNÇÃO PRINCIPAL DO TREINAMENTO

    - abre os dados;
    - limpa os dados;
    - separa treino e teste;
    - treina os modelos;
    - escolhe o melhor;
    - salva tudo que a interface vai precisar.
    """
    print("Iniciando treinamento do modelo de previsão de risco de atraso em voos...")

    # Confere se os diretórios e o dataset realmente existem no projeto.
    if not DATA_PATH.exists():
        raise FileNotFoundError("Arquivo nao encontrado")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(DATA_PATH)

    # Confere se as colunas esperadas estão presentes.
    validate_dataset_columns(dataframe)

    # Mostra informações básicas para auditoria do treinamento.
    print("Informações básicas do dataset:")
    print(f"Total de registros: {len(dataframe)}")
    print(f"Total de colunas: {len(dataframe.columns)}")
    print("Valores da variável alvo:")
    print(dataframe[TARGET_COLUMN].value_counts())

    # Limpa dados inválidos, remove colunas desnecessárias e cria features de horário.
    dataframe = clean_dataset(dataframe, COLUMNS_TO_REMOVE)

    # Recupera o relatório de limpeza guardado em dataframe.attrs.
    cleaning_report = dataframe.attrs.get("cleaning_report", {})

    # Mostra no terminal o impacto da limpeza dos dados.
    print("\nResumo da limpeza do dataset:")
    print(
        f"Registros originais: {cleaning_report.get('registros_originais', 'nao informado')}"
    )
    print(
        f"Registros apos limpeza: {cleaning_report.get('registros_apos_limpeza', len(dataframe))}"
    )
    print(
        f"Registros removidos: {cleaning_report.get('registros_removidos_total', 'nao informado')}"
    )
    print(
        f"Companhias aéreas mantidas: {', '.join(cleaning_report.get('companhias_aereas_mantidas', []))}"
    )

    # Impede treinamento com uma amostra pequena demais apos a filtragem.
    if len(dataframe) < 300:
        raise ValueError(
            "Depois da limpeza, o dataset ficou com menos de 300 registros. "
        )

    # Carrega as features e a variável alvo em x e y, respectivamente.
    x = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    # Separa 80% para treino e 20% para teste, mantendo proporção de atrasos com stratify.
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Dicionario com os algoritmos que serão comparados.
    models = {
        "Regressao Logistica": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Arvore de Decisao": DecisionTreeClassifier(
            max_depth=12, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }

    # Guarda os pipelines treinados por nome as métricas de cada modelo e as previsões de teste.
    trained_models = {}
    metrics_list = []
    predictions_by_model = {}

    # Treina e avalia cada algoritmo do dicionario.
    for name, model in models.items():
        print(f"\nTreinando modelo: {name}")
        pipeline, metrics, predictions = evaluate_model(
            name, model, x_train, x_test, y_train, y_test
        )
        trained_models[name] = pipeline
        metrics_list.append(metrics)
        predictions_by_model[name] = predictions
        print(metrics)

    # Transforma as métricas em tabela e ordena pelo F1-score e salva.
    results_dataframe = pd.DataFrame(metrics_list).sort_values(
        by="f1_score", ascending=False
    )
    results_dataframe.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    # O melhor modelo e o primeiro da tabela ordenada por F1-score.
    best_model_name = results_dataframe.iloc[0]["modelo"]

    # Salva o pipeline completo, incluindo pre-processamento e algoritmo.
    best_model = trained_models[best_model_name]
    joblib.dump(best_model, MODEL_PATH)

    # Metadados usados pela interface e pela explicação do projeto.
    metadata = {
        "melhor_modelo": best_model_name,
        "metricas": results_dataframe.to_dict(orient="records"),
        "companhias_aereas": sorted_unique_values(dataframe["Airline"]),
        "aeroportos_origem": sorted_unique_values(dataframe["AirportFrom"]),
        "aeroportos_destino": sorted_unique_values(dataframe["AirportTo"]),
        "dias_semana_disponiveis": sorted_unique_values(dataframe["DayOfWeek"]),
        "quantidade_registros": int(len(dataframe)),
        "colunas_usadas_no_modelo": FEATURE_COLUMNS,
        "colunas_removidas": COLUMNS_TO_REMOVE,
        "companhias_aereas_autorizadas": sorted(ALLOWED_AIRLINE_CODES),
        "aeroportos_autorizados": sorted(ALLOWED_AIRPORT_CODES),
        "relatorio_limpeza_dataset": cleaning_report,
    }
    save_json(metadata, METADATA_PATH)

    # Calcula a matriz de confusao.
    best_predictions = predictions_by_model[best_model_name]
    matrix = confusion_matrix(y_test, best_predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Sem atraso", "Com atraso"],
    )

    display.plot(values_format="d")
    plt.title(f"Matriz de confusao - {best_model_name}")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    # Mensagens finais.
    print("\nTreinamento concluido.")
    print(f"Melhor modelo: {best_model_name}")
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Metadados salvos em: {METADATA_PATH}")
    print(f"Resultados salvos em: {RESULTS_PATH}")
    print(f"Matriz de confusao salva em: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
