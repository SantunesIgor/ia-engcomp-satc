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
    from src.utils import clean_dataset, save_json, sorted_unique_values, validate_dataset_columns
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
    from utils import clean_dataset, save_json, sorted_unique_values, validate_dataset_columns

warnings.filterwarnings("ignore", category=UserWarning)


def create_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_pipeline(model):
    categorical_transformer = create_one_hot_encoder()

    numeric_transformer = StandardScaler()

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
    print("Iniciando treinamento do modelo de previsao de risco de atraso em voos...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Arquivo nao encontrado"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(DATA_PATH)

    validate_dataset_columns(dataframe)

    print("Informacoes basicas do dataset:")
    print(f"Total de registros: {len(dataframe)}")
    print(f"Total de colunas: {len(dataframe.columns)}")
    print("Valores da variavel alvo:")
    print(dataframe[TARGET_COLUMN].value_counts())

    dataframe = clean_dataset(dataframe, COLUMNS_TO_REMOVE)
    cleaning_report = dataframe.attrs.get("cleaning_report", {})

    print("\nResumo da limpeza do dataset:")
    print(f"Registros originais: {cleaning_report.get('registros_originais', 'nao informado')}")
    print(f"Registros apos limpeza: {cleaning_report.get('registros_apos_limpeza', len(dataframe))}")
    print(f"Registros removidos: {cleaning_report.get('registros_removidos_total', 'nao informado')}")
    print(f"Companhias aereas mantidas: {', '.join(cleaning_report.get('companhias_aereas_mantidas', []))}")

    if len(dataframe) < 300:
        raise ValueError(
            "Depois da limpeza, o dataset ficou com menos de 300 registros. "
        )

    x = dataframe[FEATURE_COLUMNS]
    y = dataframe[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Regressao Logistica": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Arvore de Decisao": DecisionTreeClassifier(max_depth=12, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }

    trained_models = {}
    metrics_list = []
    predictions_by_model = {}

    for name, model in models.items():
        print(f"\nTreinando modelo: {name}")
        pipeline, metrics, predictions = evaluate_model(name, model, x_train, x_test, y_train, y_test)
        trained_models[name] = pipeline
        metrics_list.append(metrics)
        predictions_by_model[name] = predictions
        print(metrics)

    results_dataframe = pd.DataFrame(metrics_list).sort_values(by="f1_score", ascending=False)

    results_dataframe.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    best_model_name = results_dataframe.iloc[0]["modelo"]
    best_model = trained_models[best_model_name]
    best_predictions = predictions_by_model[best_model_name]

    joblib.dump(best_model, MODEL_PATH)

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

    print("\nTreinamento concluido.")
    print(f"Melhor modelo: {best_model_name}")
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Metadados salvos em: {METADATA_PATH}")
    print(f"Resultados salvos em: {RESULTS_PATH}")
    print(f"Matriz de confusao salva em: {CONFUSION_MATRIX_PATH}")

if __name__ == "__main__":
    main()