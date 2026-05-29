# 🏦 Classificação de Risco de Crédito com Deep Learning

![Python](https://img.shields.io/badge/Python-3.14-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange)
![MLflow](https://img.shields.io/badge/MLflow-3.x-green)
![GCP](https://img.shields.io/badge/GCP-Compute_Engine-blue)
![Status](https://img.shields.io/badge/Status-Concluído-success)

Pipeline MLOps expert para classificação de risco de crédito em 4 classes usando
PyTorch com interpretabilidade via SHAP e análise de fairness por grupo demográfico.

---

## 🎯 Problema de negócio

Instituições financeiras precisam classificar automaticamente o risco de crédito de clientes em categorias (**baixo, médio, alto, crítico**) com dois requisitos regulatórios obrigatórios:

1. **Explicabilidade** — o modelo deve justificar cada decisão (SHAP)
2. **Fairness** — o modelo não pode discriminar por gênero, faixa etária ou região

---

## 📊 Resultados

| Métrica | Valor |
|---|---|
| **Accuracy** | **98.71%** |
| **F1-Score Macro** | **98.71%** |
| **AUC-ROC** | **99.95%** |
| Épocas de treinamento | 30 (early stopping) |

### Fairness — Disparidade por grupo demográfico

| Atributo | Disparidade de Accuracy | Avaliação |
|---|---|---|
| Gênero | 0.0011 | ✅ Excelente |
| Região | 0.0113 | ✅ Aceitável |
| Faixa Etária | 0.0633 | ⚠️ Monitorar |

### Top 3 features mais importantes (SHAP)
1. `score_credito` — fator mais discriminante
2. `score_normalizado` — correlacionado com score
3. `ratio_divida_renda` — segundo fator mais crítico

---

## 🏗️ Arquitetura do modelo

```
Input (13 features)
    ↓
Dense(256) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(128) → BatchNorm → ReLU → Dropout(0.21)
    ↓
Dense(64)  → BatchNorm → ReLU → Dropout(0.15)
    ↓
Dense(32)  → ReLU
    ↓
Dense(4)   → Softmax
    ↓
Output: [baixo, medio, alto, critico]
```

**Regularização:** L2 (weight decay 1e-3) + Dropout + BatchNorm
**Otimizador:** Adam + ReduceLROnPlateau
**Early Stopping:** patience=15

---

## 🔬 Fases do projeto

### Fase 1 — Dados + Feature Engineering
- Dataset sintético com **10.000 clientes** e distribuições realistas por classe
- Desbalanceamento real: 40% baixo / 30% médio / 20% alto / 10% crítico
- **SMOTE** para balanceamento → 16.000 amostras após oversampling
- Split estratificado: 70% treino / 15% val / 15% teste
- 4 gráficos EDA logados como artefatos no MLflow

### Fase 2 — Deep Learning com PyTorch
- Rede neural densa com 4 camadas ocultas
- BatchNormalization + Dropout + L2 regularization
- EarlyStopping e ReduceLROnPlateau automáticos
- Tracking completo: loss curves, confusion matrix, classification report

### Fase 3 — Explainable AI com SHAP
- **SHAP GradientExplainer** para redes neurais PyTorch
- Summary plot global + summary por classe de risco
- Waterfall chart por cliente — quais features determinaram a classificação
- Feature importance bar chart logado no MLflow

### Fase 4 — Fairness e análise de bias
- Disparidade de accuracy e F1 por **gênero, faixa etária e região**
- Relatório CSV completo por grupo demográfico
- Gráficos comparativos logados no MLflow

---

## 🚀 Como executar

### Pré-requisitos
```bash
pip install torch scikit-learn imbalanced-learn shap \
    mlflow pandas numpy matplotlib seaborn
```

### Pipeline completo em 1 comando (GCP)
```bash
curl -sSL https://raw.githubusercontent.com/leonardod38/lsampaio-credit-risk-dl/main/setup.sh | bash
```

### Passo a passo local
```bash
# 1. Iniciar MLflow
mlflow server --host 127.0.0.1 --port 5000

# 2. Gerar dados
python src/gerar_dados.py

# 3. Feature engineering + SMOTE
python src/feature_engineering.py

# 4. Treinar modelo
python src/train.py --epochs 100 --learning_rate 0.001

# 5. SHAP — Explainability
python src/explicar.py

# 6. Fairness
python src/fairness.py
```

### Acessar MLflow UI
```
http://localhost:5000
```

---

## 📁 Estrutura do projeto

```
lsampaio-credit-risk-dl/
├── src/
│   ├── gerar_dados.py           # Fase 1 — dataset 10k clientes
│   ├── feature_engineering.py  # Fase 1 — SMOTE + preprocessing
│   ├── train.py                 # Fase 2 — PyTorch MLP + MLflow
│   ├── explicar.py              # Fase 3 — SHAP GradientExplainer
│   └── fairness.py              # Fase 4 — Fairness por grupo
├── models/                      # modelos .pt salvos
├── reports/figures/             # gráficos gerados
│   ├── shap_summary_global.png
│   ├── shap_waterfall_critico.png
│   ├── fairness_accuracy.png
│   └── fairness_report.csv
├── docs/screenshots/            # evidências MLflow UI
├── MLproject                    # pipeline parametrizado
├── setup.sh                     # bootstrap GCP em 1 comando
├── requirements.txt
└── CLAUDE.md                    # memória técnica do projeto
```

---

## 🧰 Stack tecnológica

| Categoria | Tecnologia |
|---|---|
| Deep Learning | PyTorch 2.12 |
| Explainability | SHAP GradientExplainer |
| MLOps | MLflow 3.x (tracking, registry) |
| Balanceamento | SMOTE (imbalanced-learn) |
| Infraestrutura | GCP Compute Engine (e2-medium) |
| SO | Ubuntu 26.04 LTS |
| Versionamento | Git/GitHub |

---

## 💼 Contexto de portfólio

Este projeto fecha 3 lacunas críticas para posições de **ML Architect** e **Staff ML Engineer**:

| Lacuna | Como foi endereçada |
|---|---|
| Deep Learning sem projetos | Rede neural PyTorch com 98.7% de accuracy |
| Explainable AI ausente | SHAP GradientExplainer com 10 artefatos visuais |
| Fairness/Bias não documentado | Disparidade por 3 atributos demográficos auditada |

---

## 👤 Autor

**Leonardo Sampaio**
ML Engineer | 20+ anos em tecnologia
Santander · Carrefour · Accenture

🔗 [lsampaio-mlflow-segmentacao-clientes](https://github.com/leonardod38/lsampaio-mlflow-segmentacao-clientes) — projeto anterior de MLOps

---

*Pipeline executado em VM GCP com MLflow tracking completo.*
*Todos os experimentos são reprodutíveis via `setup.sh`.*
