"""
gerar_dados.py — Fase 1: Dataset sintético de risco de crédito.

Gera 10.000 clientes com distribuições realistas baseadas em padrões
de mercado financeiro brasileiro. 4 classes de risco com desbalanceamento
proporcional ao mundo real.

Autor: Leonardo Sampaio
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 10000
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DISTRIBUICAO_CLASSES = {
    "baixo":   0.40,
    "medio":   0.30,
    "alto":    0.20,
    "critico": 0.10,
}

LABEL_MAP = {"baixo": 0, "medio": 1, "alto": 2, "critico": 3}


def gerar_por_classe(classe: str, n: int) -> dict:
    """Gera features com distribuições específicas por classe de risco."""

    if classe == "baixo":
        return dict(
            score_credito         = np.random.normal(780, 40, n).clip(650, 1000),
            renda_mensal          = np.random.normal(12000, 4000, n).clip(3000, 50000),
            divida_total          = np.random.normal(8000, 5000, n).clip(0, 40000),
            historico_pagamento   = np.random.beta(9, 1, n),
            tempo_emprego_anos    = np.random.normal(8, 3, n).clip(1, 30),
            num_contas            = np.random.randint(2, 7, n),
            num_atrasos           = np.random.poisson(0.3, n).clip(0, 3),
            idade                 = np.random.normal(42, 10, n).clip(25, 70),
            genero                = np.random.choice(["M", "F"], n, p=[0.52, 0.48]),
            regiao                = np.random.choice(["Sul", "Sudeste", "Centro-Oeste", "Norte", "Nordeste"],
                                                     n, p=[0.15, 0.45, 0.15, 0.10, 0.15]),
        )

    elif classe == "medio":
        return dict(
            score_credito         = np.random.normal(640, 50, n).clip(500, 750),
            renda_mensal          = np.random.normal(6000, 2500, n).clip(1800, 25000),
            divida_total          = np.random.normal(22000, 10000, n).clip(5000, 80000),
            historico_pagamento   = np.random.beta(5, 3, n),
            tempo_emprego_anos    = np.random.normal(4, 2, n).clip(0.5, 15),
            num_contas            = np.random.randint(1, 5, n),
            num_atrasos           = np.random.poisson(1.5, n).clip(0, 8),
            idade                 = np.random.normal(35, 9, n).clip(22, 60),
            genero                = np.random.choice(["M", "F"], n, p=[0.50, 0.50]),
            regiao                = np.random.choice(["Sul", "Sudeste", "Centro-Oeste", "Norte", "Nordeste"],
                                                     n, p=[0.18, 0.38, 0.18, 0.12, 0.14]),
        )

    elif classe == "alto":
        return dict(
            score_credito         = np.random.normal(490, 60, n).clip(350, 620),
            renda_mensal          = np.random.normal(3200, 1500, n).clip(1200, 12000),
            divida_total          = np.random.normal(45000, 20000, n).clip(10000, 150000),
            historico_pagamento   = np.random.beta(2, 4, n),
            tempo_emprego_anos    = np.random.normal(2, 1.5, n).clip(0, 8),
            num_contas            = np.random.randint(1, 4, n),
            num_atrasos           = np.random.poisson(4, n).clip(0, 15),
            idade                 = np.random.normal(31, 8, n).clip(18, 55),
            genero                = np.random.choice(["M", "F"], n, p=[0.55, 0.45]),
            regiao                = np.random.choice(["Sul", "Sudeste", "Centro-Oeste", "Norte", "Nordeste"],
                                                     n, p=[0.12, 0.30, 0.20, 0.18, 0.20]),
        )

    else:  # critico
        return dict(
            score_credito         = np.random.normal(340, 50, n).clip(200, 480),
            renda_mensal          = np.random.normal(1800, 800, n).clip(800, 6000),
            divida_total          = np.random.normal(85000, 30000, n).clip(20000, 300000),
            historico_pagamento   = np.random.beta(1, 5, n),
            tempo_emprego_anos    = np.random.normal(0.8, 0.8, n).clip(0, 4),
            num_contas            = np.random.randint(1, 3, n),
            num_atrasos           = np.random.poisson(8, n).clip(2, 30),
            idade                 = np.random.normal(28, 7, n).clip(18, 50),
            genero                = np.random.choice(["M", "F"], n, p=[0.60, 0.40]),
            regiao                = np.random.choice(["Sul", "Sudeste", "Centro-Oeste", "Norte", "Nordeste"],
                                                     n, p=[0.10, 0.25, 0.20, 0.22, 0.23]),
        )


def gerar_dataset() -> pd.DataFrame:
    """Monta o dataset completo com todas as classes."""
    frames = []
    for classe, proporcao in DISTRIBUICAO_CLASSES.items():
        n_classe = int(N * proporcao)
        dados = gerar_por_classe(classe, n_classe)
        df_classe = pd.DataFrame(dados)
        df_classe["classe_risco"]  = classe
        df_classe["label"]         = LABEL_MAP[classe]
        frames.append(df_classe)

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Features derivadas
    df["ratio_divida_renda"] = (df["divida_total"] / (df["renda_mensal"] + 1)).round(4)
    df["score_normalizado"]  = (df["score_credito"] / 1000).round(4)
    df["faixa_etaria"] = pd.cut(
        df["idade"],
        bins=[0, 25, 35, 45, 60, 100],
        labels=["18-25", "26-35", "36-45", "46-60", "60+"]
    ).astype(str)

    return df


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = gerar_dataset()
    df.to_csv(DATA_DIR / "credito_raw.csv", index=False)

    print(f"Dataset gerado: {df.shape[0]} clientes, {df.shape[1]} features")
    print(f"\nDistribuição por classe:")
    print(df["classe_risco"].value_counts().to_string())
    print(f"\nEstatísticas descritivas:")
    print(df[["score_credito", "renda_mensal", "divida_total",
              "ratio_divida_renda", "num_atrasos"]].describe().round(2).to_string())
