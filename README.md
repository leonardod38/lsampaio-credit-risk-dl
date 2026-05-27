# Classificação de Risco de Crédito com Deep Learning

Pipeline MLOps expert para classificação de risco de crédito em 4 classes usando
TensorFlow/Keras com interpretabilidade via SHAP e análise de fairness/bias.

## Sobre o projeto

Modelo de Deep Learning que classifica clientes bancários em perfis de risco
(**baixo, médio, alto, crítico**) com explicação de cada decisão via SHAP values
e auditoria de fairness por grupo demográfico — atendendo requisitos regulatórios
de modelos financeiros.

## Stack tecnológica

- **TensorFlow 2.x + Keras** — rede neural multiclasse
- **SHAP (DeepExplainer)** — interpretabilidade local e global
- **TensorFlow Model Analysis** — métricas de fairness por grupo
- **imbalanced-learn (SMOTE)** — tratamento de desbalanceamento
- **MLflow 3.x** — tracking, registry e serving
- **FastAPI** — API REST com explicação SHAP por request
- **GCP Compute Engine** — treinamento em produção

## Estrutura do projeto

```
lsampaio-credit-risk-dl/
├── src/
│   ├── gerar_dados.py          # dataset sintético 10k clientes
│   ├── feature_engineering.py # SMOTE + preprocessing
│   ├── train.py                # rede neural TF/Keras + MLflow
│   ├── explicar.py             # SHAP values
│   ├── fairness.py             # Fairness Indicators
│   └── api.py                  # FastAPI serving
├── models/                     # modelos salvos
├── reports/figures/            # gráficos gerados
├── docs/screenshots/           # evidências MLflow UI
├── MLproject                   # pipeline parametrizado
└── setup.sh                    # bootstrap em 1 comando
```

## Fases do projeto

### Fase 1 — Dados + Feature Engineering
- Dataset sintético com 10.000 clientes e distribuições realistas
- 4 classes de risco com desbalanceamento real (baixo 40%, médio 30%, alto 20%, crítico 10%)
- SMOTE para balanceamento + StandardScaler + encoding categórico

### Fase 2 — Deep Learning com TensorFlow
- Rede neural densa com BatchNormalization, Dropout e regularização L2
- Otimização de hiperparâmetros com Keras Tuner
- Tracking completo de epochs, loss curves e confusion matrix no MLflow

### Fase 3 — Explainable AI com SHAP
- SHAP DeepExplainer para redes neurais TensorFlow
- Summary plot global + waterfall chart por classe de risco
- Explicação local: quais features classificaram cada cliente

### Fase 4 — Fairness e mitigação de bias
- Disparidade de accuracy e FPR por gênero, faixa etária e região
- Mitigação com rebalanceamento de amostras + class weights
- Comparação modelo original vs modelo mitigado no MLflow

### Fase 5 — Produção
- Model Registry com versionamento e alias "producao-com-fairness"
- API REST que retorna: classe de risco + probabilidades + top 3 SHAP features
- Bootstrap em 1 comando via setup.sh no GCP

## Como executar

### Pré-requisitos
```bash
pip install -r requirements.txt
```

### Pipeline completo
```bash
curl -sSL https://raw.githubusercontent.com/leonardod38/lsampaio-credit-risk-dl/main/setup.sh | bash
```

### Passo a passo
```bash
python src/gerar_dados.py
python src/train.py --epochs 100 --learning_rate 0.001
python src/explicar.py
python src/fairness.py
python src/api.py
```

### Acessar MLflow UI
```
http://localhost:5000
```

### Exemplo de request à API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"score_credito": 420, "renda_mensal": 2500, "divida_total": 18000,
       "historico_pagamento": 0.4, "tempo_emprego_anos": 1, "num_atrasos": 8}'
```

Resposta:
```json
{
  "classe_risco": "crítico",
  "probabilidades": {"baixo": 0.03, "médio": 0.08, "alto": 0.21, "crítico": 0.68},
  "explicacao_shap": [
    {"feature": "num_atrasos", "impacto": 0.42, "valor": 8},
    {"feature": "score_credito", "impacto": 0.31, "valor": 420},
    {"feature": "historico_pagamento", "impacto": 0.18, "valor": 0.4}
  ]
}
```

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy geral | — |
| F1-score macro | — |
| AUC-ROC | — |
| Disparidade de FPR (gênero) | — |

> Tabela atualizada após execução completa do pipeline.

## Infraestrutura GCP

- **Máquina:** e2-medium (2 vCPU, 4GB RAM)
- **Região:** us-central1-a
- **SO:** Ubuntu 26.04 LTS Minimal
- **Custo estimado:** ~US$ 0,03/hora

## Autor

**Leonardo Sampaio**
ML Engineer | 20+ anos em tecnologia
Santander · Carrefour · Accenture

---
*Projeto de portfólio MLOps — Deep Learning com interpretabilidade e fairness para posições de ML Architect/Staff ML Engineer.*
