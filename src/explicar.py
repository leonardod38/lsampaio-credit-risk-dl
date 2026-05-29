"""
explicar.py — Fase 3: Explainable AI com SHAP GradientExplainer (PyTorch).

Gera explicacoes globais e locais do modelo:
- Summary plot global
- Waterfall chart por classe de risco
- Feature importance bar chart
- Todos os artefatos logados no MLflow

Autor: Leonardo Sampaio
"""

import sys
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
from pathlib import Path

ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data"
MODELS_DIR   = ROOT / "models"
FIGS_DIR     = ROOT / "reports" / "figures"
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT   = "classificacao-risco-credito"
CLASSES      = ["baixo", "medio", "alto", "critico"]
FEATURES = [
    "score_credito", "renda_mensal", "divida_total",
    "ratio_divida_renda", "historico_pagamento",
    "tempo_emprego_anos", "num_contas", "num_atrasos",
    "idade", "score_normalizado",
    "genero_enc", "regiao_enc", "faixa_etaria_enc"
]


def carregar_modelo_pytorch():
    from train import CreditRiskNet
    pts = list(MODELS_DIR.glob("*.pt"))
    if not pts:
        print("Nenhum modelo .pt encontrado. Execute train.py primeiro.")
        sys.exit(1)
    pt_path = sorted(pts)[-1]
    print(f"Carregando: {pt_path.name}")
    n_feat = len(FEATURES)
    modelo = CreditRiskNet(n_feat)
    modelo.load_state_dict(torch.load(str(pt_path), weights_only=True))
    modelo.eval()
    return modelo


def gerar_explicacoes(modelo):
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    artefatos = []

    df_train = pd.read_csv(DATA_DIR / "credito_train.csv")
    df_test  = pd.read_csv(DATA_DIR / "credito_test.csv")
    X_train  = torch.FloatTensor(df_train[FEATURES].values)
    X_test   = torch.FloatTensor(df_test[FEATURES].values[:500])

    print("Calculando SHAP values (GradientExplainer)...")
    bg_idx   = np.random.choice(len(X_train), 200, replace=False)
    background = X_train[bg_idx]
    explainer  = shap.GradientExplainer(modelo, background)
    shap_values = explainer.shap_values(X_test)

    X_np = X_test.numpy()

    # 1 — Summary plot global
    shap.summary_plot(shap_values, X_np, feature_names=FEATURES,
                      class_names=CLASSES, show=False, max_display=13)
    path = str(FIGS_DIR / "shap_summary_global.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    artefatos.append(path)
    print("  Summary global gerado.")

    # 2 — Summary por classe
    for i, classe in enumerate(CLASSES):
        shap.summary_plot(shap_values[i], X_np, feature_names=FEATURES,
                          show=False, max_display=10)
        plt.title(f"SHAP — Classe: {classe.upper()}", fontsize=12)
        path = str(FIGS_DIR / f"shap_summary_{classe}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        artefatos.append(path)
    print("  Summary por classe gerados.")

    # 3 — Waterfall por classe
    y_test = pd.read_csv(DATA_DIR / "credito_test.csv")["label"].values[:500]
    for i, classe in enumerate(CLASSES):
        idxs = np.where(y_test == i)[0]
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        sv  = shap_values[i][idx]
        exp = shap.Explanation(
            values=sv,
            base_values=float(explainer.expected_value[i]),
            data=X_np[idx],
            feature_names=FEATURES
        )
        shap.plots.waterfall(exp, show=False, max_display=10)
        plt.title(f"Waterfall — Risco {classe.upper()}", fontsize=11)
        path = str(FIGS_DIR / f"shap_waterfall_{classe}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        artefatos.append(path)
    print("  Waterfall charts gerados.")

    # 4 — Feature importance bar chart
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
    plt.savefig(path, dpi=150); plt.close()
    artefatos.append(path)
    print("  Feature importance gerado.")

    return artefatos, shap_values


def pipeline_explicar():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    modelo = carregar_modelo_pytorch()

    with mlflow.start_run(run_name="fase3_shap_xai"):
        mlflow.log_param("metodo_xai",   "SHAP GradientExplainer")
        mlflow.log_param("framework",    "PyTorch")
        mlflow.log_param("n_background", 200)
        mlflow.log_param("n_explicados", 500)

        artefatos, shap_values = gerar_explicacoes(modelo)

        mean_shap = np.mean([np.abs(sv).mean(0) for sv in shap_values], axis=0)
        for feat, val in zip(FEATURES, mean_shap):
            mlflow.log_metric(f"shap_{feat}", round(float(val), 6))

        top3 = pd.Series(mean_shap, index=FEATURES).nlargest(3)
        mlflow.log_param("top1_feature", top3.index[0])
        mlflow.log_param("top2_feature", top3.index[1])
        mlflow.log_param("top3_feature", top3.index[2])

        for artefato in artefatos:
            mlflow.log_artifact(artefato)

        print(f"\nFase 3 concluida! Artefatos: {len(artefatos)}")
        print(f"Top 3 features: {list(top3.index)}")
        print(f"\nAcesse: http://127.0.0.1:5000")


if __name__ == "__main__":
    pipeline_explicar()
