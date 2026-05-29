"""
explicar.py — Fase 3: Explainable AI com SHAP DeepExplainer.

Gera explicacoes globais e locais do modelo de Deep Learning:
- Summary plot global (importancia das features)
- Waterfall chart por classe de risco
- Force plot para clientes individuais
- Todos os artefatos logados no MLflow

Autor: Leonardo Sampaio
"""

import sys
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from tensorflow import keras

ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data"
MODELS_DIR   = ROOT / "models"
FIGS_DIR     = ROOT / "reports" / "figures"
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT   = "classificacao-risco-credito"

CLASSES  = ["baixo", "medio", "alto", "critico"]
FEATURES = [
    "score_credito", "renda_mensal", "divida_total",
    "ratio_divida_renda", "historico_pagamento",
    "tempo_emprego_anos", "num_contas", "num_atrasos",
    "idade", "score_normalizado",
    "genero_enc", "regiao_enc", "faixa_etaria_enc"
]


def carregar_modelo_e_dados():
    modelos = list(MODELS_DIR.glob("*.keras"))
    if not modelos:
        print("Nenhum modelo encontrado. Execute train.py primeiro.")
        sys.exit(1)
    modelo_path = sorted(modelos)[-1]
    print(f"Carregando modelo: {modelo_path.name}")
    modelo = keras.models.load_model(str(modelo_path))

    df_train = pd.read_csv(DATA_DIR / "credito_train.csv")
    df_test  = pd.read_csv(DATA_DIR / "credito_test.csv")
    X_train  = df_train[FEATURES].values
    X_test   = df_test[FEATURES].values
    y_test   = df_test["label"].values
    return modelo, X_train, X_test, y_test


def gerar_shap_explicacoes(modelo, X_train, X_test):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    artefatos = []

    print("Calculando SHAP values (DeepExplainer)...")
    background = X_train[np.random.choice(X_train.shape[0], 200, replace=False)]
    explainer  = shap.DeepExplainer(modelo, background)
    shap_values = explainer.shap_values(X_test[:500])

    # 1 — Summary plot global (todas as classes)
    shap.summary_plot(
        shap_values, X_test[:500],
        feature_names=FEATURES,
        class_names=CLASSES,
        show=False, max_display=13
    )
    path = str(FIGS_DIR / "shap_summary_global.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    artefatos.append(path)
    print("  Summary plot global gerado.")

    # 2 — Summary plot por classe de risco
    for i, classe in enumerate(CLASSES):
        shap.summary_plot(
            shap_values[i], X_test[:500],
            feature_names=FEATURES,
            show=False, max_display=10
        )
        plt.title(f"SHAP — Classe: {classe.upper()}", fontsize=12)
        path = str(FIGS_DIR / f"shap_summary_{classe}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        artefatos.append(path)
    print("  Summary plots por classe gerados.")

    # 3 — Waterfall chart: 1 cliente por classe de risco
    df_test_full = pd.read_csv(DATA_DIR / "credito_test.csv")
    for i, classe in enumerate(CLASSES):
        indices_classe = np.where(df_test_full["label"].values[:500] == i)[0]
        if len(indices_classe) == 0:
            continue
        idx = indices_classe[0]
        shap_vals = shap_values[i][idx]
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=explainer.expected_value[i],
            data=X_test[idx],
            feature_names=FEATURES
        )
        shap.plots.waterfall(explanation, show=False, max_display=10)
        plt.title(f"Explicacao SHAP — Cliente Risco {classe.upper()}", fontsize=11)
        path = str(FIGS_DIR / f"shap_waterfall_{classe}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        artefatos.append(path)
    print("  Waterfall charts gerados.")

    # 4 — Feature importance media (bar chart)
    mean_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
    importancia = pd.Series(mean_shap, index=FEATURES).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    cores = ["#D85A30" if v > importancia.median() else "#378ADD"
             for v in importancia.values]
    importancia.plot(kind="barh", ax=ax, color=cores, edgecolor="none")
    ax.set_title("Feature Importance Global (SHAP)", fontsize=13)
    ax.set_xlabel("Mean |SHAP value|")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = str(FIGS_DIR / "shap_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    artefatos.append(path)
    print("  Feature importance chart gerado.")

    return artefatos, shap_values, explainer


def pipeline_explicar():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    modelo, X_train, X_test, y_test = carregar_modelo_e_dados()

    with mlflow.start_run(run_name="fase3_shap_explicabilidade"):
        mlflow.log_param("metodo_xai",     "SHAP DeepExplainer")
        mlflow.log_param("n_background",   200)
        mlflow.log_param("n_explicados",   500)
        mlflow.log_param("n_classes",      4)

        artefatos, shap_values, explainer = gerar_shap_explicacoes(
            modelo, X_train, X_test
        )

        # Logar importancia das features como metricas
        mean_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
        for feat, val in zip(FEATURES, mean_shap):
            mlflow.log_metric(f"shap_{feat}", round(float(val), 6))

        # Top 3 features mais importantes
        top3 = pd.Series(mean_shap, index=FEATURES).nlargest(3)
        mlflow.log_param("top1_feature", top3.index[0])
        mlflow.log_param("top2_feature", top3.index[1])
        mlflow.log_param("top3_feature", top3.index[2])

        # Logar todos os artefatos
        for artefato in artefatos:
            mlflow.log_artifact(artefato)

        print(f"\nFase 3 concluida!")
        print(f"  Artefatos SHAP: {len(artefatos)}")
        print(f"  Top 3 features: {list(top3.index)}")
        print(f"\nAcesse: http://127.0.0.1:5000")


if __name__ == "__main__":
    pipeline_explicar()
