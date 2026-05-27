# CLAUDE.md — Memória do projeto

## Identidade do projeto
- **Nome:** lsampaio-credit-risk-dl
- **Autor:** Leonardo Sampaio (leonardod38)
- **Objetivo:** Classificação de risco de crédito com Deep Learning, SHAP e Fairness Indicators
- **Repositório local:** E:\Claude\lsampaio-credit-risk-dl
- **VM GCP:** IP variável (verificar no Console) — us-central1-a, e2-medium, Ubuntu 26.04 LTS

## Stack
- Python 3.11 (local) / Python 3.14 (GCP)
- TensorFlow 2.x + Keras
- SHAP (DeepExplainer)
- TensorFlow Model Analysis + Fairness Indicators
- MLflow (tracking, registry, serving)
- FastAPI (API REST)
- imbalanced-learn (SMOTE)
- GCP Compute Engine

## Estrutura do projeto
```
lsampaio-credit-risk-dl/
├── data/
│   ├── credito_raw.csv          # gerado por gerar_dados.py
│   ├── credito_features.csv     # features processadas
│   └── credito_test.csv         # conjunto de teste
├── src/
│   ├── gerar_dados.py           # Fase 1 — dataset 10k clientes
│   ├── feature_engineering.py  # Fase 1 — SMOTE + preprocessing
│   ├── train.py                 # Fase 2 — rede neural TF/Keras
│   ├── explicar.py              # Fase 3 — SHAP values
│   ├── fairness.py              # Fase 4 — Fairness Indicators
│   └── api.py                   # Fase 5 — FastAPI serving
├── models/                      # modelos salvos
├── notebooks/                   # análise exploratória
├── reports/figures/             # gráficos gerados
├── docs/screenshots/            # evidências MLflow UI
├── MLproject                    # entrypoints parametrizados
├── requirements.txt
├── setup.sh
├── CLAUDE.md
└── README.md
```

## Classes de risco
| Classe | Label | Perfil |
|--------|-------|--------|
| 0 | Baixo | Score alto, renda estável, sem atrasos |
| 1 | Médio | Score médio, algum histórico de atraso |
| 2 | Alto | Score baixo, dívida elevada |
| 3 | Crítico | Score muito baixo, inadimplência recorrente |

## Fases e status
| Fase | Descrição | Status |
|------|-----------|--------|
| 1 | Dataset 10k + Feature Engineering + SMOTE | ⏳ Pendente |
| 2 | Rede neural TF/Keras + MLflow tracking | ⏳ Pendente |
| 3 | SHAP — Explainable AI | ⏳ Pendente |
| 4 | Fairness Indicators — análise de bias | ⏳ Pendente |
| 5 | Model Registry + FastAPI + GitHub | ⏳ Pendente |

## Features do dataset
```python
FEATURES = [
    "score_credito", "renda_mensal", "divida_total",
    "ratio_divida_renda", "historico_pagamento",
    "tempo_emprego_anos", "num_contas", "num_atrasos",
    "idade", "score_normalizado"
]
ATRIBUTOS_SENSIVEIS = ["genero", "faixa_etaria", "regiao"]
```

## Experimento MLflow
- **Nome:** classificacao-risco-credito
- **Tracking URI:** http://127.0.0.1:5000

## Git
```bash
cd E:\Claude\lsampaio-credit-risk-dl
git init && git add . && git commit -m "feat: estrutura inicial do projeto"
```

## Próximos passos
1. Criar src/gerar_dados.py — Fase 1
2. Criar src/feature_engineering.py — Fase 1
3. Criar src/train.py — Fase 2
4. Criar src/explicar.py — Fase 3
5. Criar src/fairness.py — Fase 4
6. Criar src/api.py — Fase 5
7. Criar setup.sh
8. Criar repositório GitHub público
