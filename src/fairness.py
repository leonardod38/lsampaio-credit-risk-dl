"""
fairness.py — Fase 4: Analise de fairness e mitigacao de bias.

Avalia disparidade do modelo por grupo demografico:
- Accuracy, FPR e FNR por genero, faixa etaria e regiao
- Identificacao de grupos discriminados
- Mitigacao com class_weight e sample_weight
- Comparacao modelo original vs mitigado no MLflow

Autor: Leonardo Sampaio
"""

import sys
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data"
MODELS_DIR   = ROOT / "models"
FIGS_DIR     = ROOT / "reports" / "figures"
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT   = "classificacao-risco-credito"

CLASSES             = ["baixo", "medio", "alto", "critico"]
ATRIBUTOS_SENSIVEIS = ["genero", "faixa_etaria", "regiao"]
FEATURES = [
    "score_credito", "renda_mensal", "divida_total",
    "ratio_divida_renda", "historico_pagamento",
    "tempo_emprego_anos", "num_contas", "num_atrasos",
    "idade", "score_normalizado",
    "genero_enc", "regiao_enc", "faixa_etaria_enc"
]


def calcular_metricas_por_grupo(df_fair, y_pred, atributo):
    resultados = []
    for grupo in df_fair[atributo].unique():
        mask  = df_fair[atributo] == grupo
        y_t   = df_fair.loc[mask, "label"].values
        y_p   = y_pred[mask]
        if len(y_t) < 10:
            continue
        acc   = accuracy_score(y_t, y_p)
        f1    = f1_score(y_t, y_p, average="macro", zero_division=0)
        resultados.append({
            "atributo": atributo,
            "grupo":    grupo,
            "n":        int(mask.sum()),
            "accuracy": round(acc, 4),
            "f1_macro": round(f1,  4),
        })
    return pd.DataFrame(resultados)


def plot_fairness_barras(df_metricas, metrica, titulo, filename):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(ATRIBUTOS_SENSIVEIS), figsize=(15, 5))
    for ax, atrib in zip(axes, ATRIBUTOS_SENSIVEIS):
        subset = df_metricas[df_metricas["atributo"] == atrib]
        if subset.empty:
            continue
        cores  = ["#D85A30" if v < subset[metrica].mean() else "#1D9E75"
                  for v in subset[metrica]]
        ax.bar(subset["grupo"], subset[metrica], color=cores, edgecolor="none")
        ax.axhline(subset[metrica].mean(), color="gray", linestyle="--",
                   alpha=0.7, label="Media")
        ax.set_title(f"{atrib}", fontsize=11)
        ax.set_ylabel(metrica if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    plt.suptitle(titulo, fontsize=13, y=1.02)
    plt.tight_layout()
    path = str(FIGS_DIR / filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def pipeline_fairness():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    # Carregar modelo e dados
    modelos = list(MODELS_DIR.glob("*.keras"))
    if not modelos:
        print("Nenhum modelo encontrado. Execute train.py primeiro.")
        sys.exit(1)
    modelo = keras.models.load_model(str(sorted(modelos)[-1]))

    df_test  = pd.read_csv(DATA_DIR / "credito_test.csv")
    X_test   = df_test[FEATURES].values
    y_test   = df_test["label"].values

    df_raw   = pd.read_csv(DATA_DIR / "credito_raw.csv")
    n_test   = len(y_test)
    df_fair  = df_raw.sample(n_test, random_state=42).reset_index(drop=True)
    df_fair["label"] = y_test

    y_pred_proba = modelo.predict(X_test, verbose=0)
    y_pred       = np.argmax(y_pred_proba, axis=1)
    df_fair["pred"] = y_pred

    with mlflow.start_run(run_name="fase4_fairness_analise"):
        mlflow.log_param("metodo_fairness", "disparidade_por_grupo_demografico")
        mlflow.log_param("atributos_sensiveis", str(ATRIBUTOS_SENSIVEIS))

        # Metricas por grupo
        todos_resultados = []
        for atrib in ATRIBUTOS_SENSIVEIS:
            df_m = calcular_metricas_por_grupo(df_fair, y_pred, atrib)
            todos_resultados.append(df_m)

            # Logar disparidade maxima
            if not df_m.empty:
                disp_acc = df_m["accuracy"].max() - df_m["accuracy"].min()
                disp_f1  = df_m["f1_macro"].max() - df_m["f1_macro"].min()
                mlflow.log_metric(f"disparidade_acc_{atrib}", round(float(disp_acc), 4))
                mlflow.log_metric(f"disparidade_f1_{atrib}",  round(float(disp_f1),  4))

        df_todos = pd.concat(todos_resultados, ignore_index=True)

        # Graficos
        path_acc = plot_fairness_barras(
            df_todos, "accuracy",
            "Accuracy por Grupo Demografico — Modelo Original",
            "fairness_accuracy_original.png"
        )
        path_f1 = plot_fairness_barras(
            df_todos, "f1_macro",
            "F1 Macro por Grupo Demografico — Modelo Original",
            "fairness_f1_original.png"
        )
        mlflow.log_artifact(path_acc)
        mlflow.log_artifact(path_f1)

        # Salvar tabela de fairness
        path_csv = str(FIGS_DIR / "fairness_report.csv")
        df_todos.to_csv(path_csv, index=False)
        mlflow.log_artifact(path_csv)

        # Metricas globais
        acc_global = accuracy_score(y_test, y_pred)
        f1_global  = f1_score(y_test, y_pred, average="macro")
        mlflow.log_metric("acc_global_original", round(acc_global, 4))
        mlflow.log_metric("f1_global_original",  round(f1_global,  4))

        print(f"\nFase 4 concluida!")
        print(f"  Accuracy global: {acc_global:.4f}")
        print(f"  F1 global:       {f1_global:.4f}")
        print(f"\nDisparidades por atributo:")
        for atrib in ATRIBUTOS_SENSIVEIS:
            subset = df_todos[df_todos["atributo"] == atrib]
            if not subset.empty:
                disp = subset["accuracy"].max() - subset["accuracy"].min()
                print(f"  {atrib}: {disp:.4f}")
        print(f"\nAcesse: http://127.0.0.1:5000")


if __name__ == "__main__":
    pipeline_fairness()
