"""
feature_engineering.py — Fase 1: Preprocessing, SMOTE e EDA.

Prepara o dataset para o modelo de Deep Learning:
- Encoding de variáveis categóricas
- Tratamento de outliers com IQR
- SMOTE para balanceamento de classes
- StandardScaler nas features numéricas
- Gráficos EDA logados no MLflow

Autor: Leonardo Sampaio
"""

import sys
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mlflow_utils import configure_mlflow
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGS_DIR = ROOT / "reports" / "figures"

import os as _os
TRACKING_URI = _os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT   = "classificacao-risco-credito"

FEATURES_NUMERICAS = [
    "score_credito", "renda_mensal", "divida_total",
    "ratio_divida_renda", "historico_pagamento",
    "tempo_emprego_anos", "num_contas", "num_atrasos",
    "idade", "score_normalizado"
]

FEATURES_CATEGORICAS = ["genero", "regiao", "faixa_etaria"]
TARGET = "label"


def tratar_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURES_NUMERICAS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df


def gerar_graficos_eda(df: pd.DataFrame) -> list:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    artefatos = []

    # 1 — Distribuição das classes
    fig, ax = plt.subplots(figsize=(8, 5))
    cores = ["#1D9E75", "#378ADD", "#EF9F27", "#E24B4A"]
    df["classe_risco"].value_counts().plot(kind="bar", ax=ax, color=cores, edgecolor="white")
    ax.set_title("Distribuição das Classes de Risco", fontsize=13)
    ax.set_xlabel("Classe de Risco")
    ax.set_ylabel("Quantidade de Clientes")
    ax.tick_params(axis="x", rotation=0)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = str(FIGS_DIR / "distribuicao_classes.png")
    plt.savefig(path, dpi=150)
    plt.close()
    artefatos.append(path)

    # 2 — Score de crédito por classe
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (classe, cor) in enumerate(zip(
        ["baixo", "medio", "alto", "critico"], cores
    )):
        dados = df[df["classe_risco"] == classe]["score_credito"]
        ax.hist(dados, bins=30, alpha=0.65, color=cor, label=classe, edgecolor="none")
    ax.set_title("Score de Crédito por Classe de Risco", fontsize=13)
    ax.set_xlabel("Score de Crédito")
    ax.set_ylabel("Frequência")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = str(FIGS_DIR / "score_por_classe.png")
    plt.savefig(path, dpi=150)
    plt.close()
    artefatos.append(path)

    # 3 — Heatmap de correlação
    fig, ax = plt.subplots(figsize=(11, 8))
    corr = df[FEATURES_NUMERICAS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5)
    ax.set_title("Correlação entre Features Numéricas", fontsize=13)
    plt.tight_layout()
    path = str(FIGS_DIR / "heatmap_correlacao.png")
    plt.savefig(path, dpi=150)
    plt.close()
    artefatos.append(path)

    # 4 — Ratio dívida/renda por classe
    fig, ax = plt.subplots(figsize=(9, 5))
    df.boxplot(column="ratio_divida_renda", by="classe_risco", ax=ax,
               boxprops=dict(color="#185FA5"),
               medianprops=dict(color="#D85A30", linewidth=2))
    ax.set_title("Ratio Dívida/Renda por Classe de Risco", fontsize=13)
    ax.set_xlabel("Classe de Risco")
    ax.set_ylabel("Ratio Dívida/Renda")
    plt.suptitle("")
    plt.tight_layout()
    path = str(FIGS_DIR / "ratio_divida_por_classe.png")
    plt.savefig(path, dpi=150)
    plt.close()
    artefatos.append(path)

    return artefatos


def pipeline_feature_engineering():
    configure_mlflow(EXPERIMENT, TRACKING_URI)

    path_raw = DATA_DIR / "credito_raw.csv"
    if not path_raw.exists():
        print("Dataset não encontrado. Execute gerar_dados.py primeiro.")
        sys.exit(1)

    df = pd.read_csv(path_raw)

    with mlflow.start_run(run_name="fase1_feature_engineering"):

        # Tratar outliers
        df = tratar_outliers(df)

        # Encoding categórico
        le_genero  = LabelEncoder()
        le_regiao  = LabelEncoder()
        le_faixa   = LabelEncoder()
        df["genero_enc"]     = le_genero.fit_transform(df["genero"])
        df["regiao_enc"]     = le_regiao.fit_transform(df["regiao"])
        df["faixa_etaria_enc"] = le_faixa.fit_transform(df["faixa_etaria"])

        FEATURES_MODELO = FEATURES_NUMERICAS + [
            "genero_enc", "regiao_enc", "faixa_etaria_enc"
        ]
        ATRIBUTOS_SENSIVEIS = ["genero", "regiao", "faixa_etaria",
                               "genero_enc", "regiao_enc", "faixa_etaria_enc"]

        X = df[FEATURES_MODELO].values
        y = df[TARGET].values

        # SMOTE — balancear classes
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X, y)

        # Train/val/test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        # StandardScaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        # Salvar splits
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(X_train_s, columns=FEATURES_MODELO).assign(label=y_train)\
            .to_csv(DATA_DIR / "credito_train.csv", index=False)
        pd.DataFrame(X_val_s, columns=FEATURES_MODELO).assign(label=y_val)\
            .to_csv(DATA_DIR / "credito_val.csv", index=False)
        pd.DataFrame(X_test_s, columns=FEATURES_MODELO).assign(label=y_test)\
            .to_csv(DATA_DIR / "credito_test.csv", index=False)

        # Salvar dataset de teste com atributos sensíveis para fairness
        idx_test_orig = df.sample(len(y_test), random_state=99).index
        df_test_fair = df.loc[idx_test_orig, ATRIBUTOS_SENSIVEIS + [TARGET]].reset_index(drop=True)
        df_test_fair["pred_label"] = -1  # será preenchido após treinamento
        df_test_fair.to_csv(DATA_DIR / "credito_test_fairness.csv", index=False)

        # Log MLflow
        mlflow.log_param("n_amostras_original", len(df))
        mlflow.log_param("n_amostras_apos_smote", len(X_bal))
        mlflow.log_param("n_features", len(FEATURES_MODELO))
        mlflow.log_param("balanceamento", "SMOTE")
        mlflow.log_param("scaler", "StandardScaler")
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("val_size",   len(X_val))
        mlflow.log_metric("test_size",  len(X_test))

        # EDA
        artefatos = gerar_graficos_eda(df)
        for artefato in artefatos:
            mlflow.log_artifact(artefato)

        print(f"Fase 1 concluída!")
        print(f"  Original: {len(df)} | Após SMOTE: {len(X_bal)}")
        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        print(f"  Features: {len(FEATURES_MODELO)}")
        print(f"  Artefatos EDA: {len(artefatos)}")

    return FEATURES_MODELO


if __name__ == "__main__":
    pipeline_feature_engineering()
