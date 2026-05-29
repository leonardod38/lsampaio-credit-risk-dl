#!/bin/bash
# =============================================================================
# setup.sh — Bootstrap do projeto na VM GCP
# Classifica risco de credito com Deep Learning, SHAP e Fairness
#
# Uso:
#   chmod +x setup.sh && ./setup.sh
# =============================================================================

set -e

REPO_URL="https://github.com/leonardod38/lsampaio-credit-risk-dl.git"
BRANCH="main"
PROJECT_DIR="$HOME/lsampaio-credit-risk-dl"
VENV_DIR="$HOME/mlflow-env"

echo "============================================================"
echo "  Deep Learning — Classificacao de Risco de Credito"
echo "============================================================"

# 1 — Clonar ou atualizar
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "[1/6] Atualizando repositorio..."
    cd "$PROJECT_DIR" && git pull origin "$BRANCH"
else
    echo "[1/6] Clonando repositorio..."
    git clone --branch "$BRANCH" "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# 2 — Ativar ambiente virtual
echo "[2/6] Ativando ambiente virtual..."
source "$VENV_DIR/bin/activate"

# 3 — Instalar dependencias
echo "[3/6] Instalando dependencias..."
pip install -q tensorflow scikit-learn imbalanced-learn shap \
    mlflow pandas numpy matplotlib seaborn fastapi uvicorn

# 4 — Criar estrutura de pastas
echo "[4/6] Criando pastas..."
cd "$PROJECT_DIR"
mkdir -p data models reports/figures docs/screenshots

# 5 — Verificar MLflow server
echo "[5/6] Verificando MLflow server..."
if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "  Subindo MLflow server..."
    mlflow server \
        --host 0.0.0.0 --port 5000 \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlflow-artifacts \
        --allowed-hosts "*" --cors-allowed-origins "*" \
        > ~/mlflow.log 2>&1 &
    sleep 8
fi
echo "  MLflow OK: http://$(curl -s ifconfig.me 2>/dev/null || echo 'localhost'):5000"

# 6 — Executar pipeline completo
echo "[6/6] Executando pipeline..."

echo ""
echo ">>> Fase 1 — Gerando dataset (10k clientes)..."
python src/gerar_dados.py

echo ""
echo ">>> Fase 1 — Feature Engineering + SMOTE..."
python src/feature_engineering.py

echo ""
echo ">>> Fase 2 — Treinando rede neural TensorFlow..."
python src/train.py --epochs 100 --learning_rate 0.001

echo ""
echo ">>> Fase 3 — SHAP Explainability..."
python src/explicar.py

echo ""
echo ">>> Fase 4 — Fairness Analysis..."
python src/fairness.py

echo ""
echo "============================================================"
echo "  Pipeline concluido!"
echo "  MLflow UI: http://$(curl -s ifconfig.me 2>/dev/null || echo 'localhost'):5000"
echo "============================================================"
