"""
train.py — Fase 2: Rede neural multiclasse com PyTorch.

Arquitetura com BatchNorm, Dropout e regularizacao L2.
Tracking completo no MLflow: hiperparametros, loss curves,
confusion matrix e classification report.

Autor: Leonardo Sampaio
"""

import argparse
import sys
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

ROOT         = Path(__file__).resolve().parent.parent
DATA_DIR     = ROOT / "data"
MODELS_DIR   = ROOT / "models"
FIGS_DIR     = ROOT / "reports" / "figures"
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT   = "classificacao-risco-credito"
CLASSES      = ["baixo", "medio", "alto", "critico"]
N_CLASSES    = 4
FEATURES = [
    "score_credito", "renda_mensal", "divida_total",
    "ratio_divida_renda", "historico_pagamento",
    "tempo_emprego_anos", "num_contas", "num_atrasos",
    "idade", "score_normalizado",
    "genero_enc", "regiao_enc", "faixa_etaria_enc"
]


class CreditRiskNet(nn.Module):
    """Rede neural densa: Input -> 256 -> 128 -> 64 -> 32 -> Softmax(4)"""
    def __init__(self, n_features, dropout_rate=0.3, l2_lambda=0.001):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout_rate * 0.7),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, N_CLASSES)
        )

    def forward(self, x):
        return self.network(x)


def carregar_dados():
    for fname in ["credito_train.csv", "credito_val.csv", "credito_test.csv"]:
        if not (DATA_DIR / fname).exists():
            print(f"Arquivo {fname} nao encontrado. Execute feature_engineering.py.")
            sys.exit(1)
    df_train = pd.read_csv(DATA_DIR / "credito_train.csv")
    df_val   = pd.read_csv(DATA_DIR / "credito_val.csv")
    df_test  = pd.read_csv(DATA_DIR / "credito_test.csv")
    X_train  = torch.FloatTensor(df_train[FEATURES].values)
    y_train  = torch.LongTensor(df_train["label"].values)
    X_val    = torch.FloatTensor(df_val[FEATURES].values)
    y_val    = torch.LongTensor(df_val["label"].values)
    X_test   = torch.FloatTensor(df_test[FEATURES].values)
    y_test   = torch.LongTensor(df_test["label"].values)
    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_loss_curves(train_losses, val_losses, train_accs, val_accs, run_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(train_losses, label="Treino",    color="#185FA5", linewidth=2)
    ax1.plot(val_losses,   label="Validacao", color="#D85A30", linewidth=2)
    ax1.set_title("Loss por Epoca", fontsize=12)
    ax1.set_xlabel("Epoca"); ax1.set_ylabel("Loss"); ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)
    ax2.plot(train_accs, label="Treino",    color="#185FA5", linewidth=2)
    ax2.plot(val_accs,   label="Validacao", color="#D85A30", linewidth=2)
    ax2.set_title("Accuracy por Epoca", fontsize=12)
    ax2.set_xlabel("Epoca"); ax2.set_ylabel("Accuracy"); ax2.legend()
    ax2.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    path = str(FIGS_DIR / f"loss_curves_{run_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    return path


def plot_confusion_matrix(y_true, y_pred, run_name):
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"Confusion Matrix — {run_name}", fontsize=12)
    ax.set_ylabel("Real"); ax.set_xlabel("Predito")
    plt.tight_layout()
    path = str(FIGS_DIR / f"confusion_matrix_{run_name}.png")
    plt.savefig(path, dpi=150); plt.close()
    return path


def treinar(epochs=100, batch_size=64, learning_rate=0.001,
            dropout_rate=0.3, run_name="fase2_pytorch"):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_train, y_train, X_val, y_val, X_test, y_test = carregar_dados()
    n_features = X_train.shape[1]

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val),
                              batch_size=batch_size)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("framework",      "PyTorch")
        mlflow.log_param("arquitetura",    "256-128-64-32-softmax4")
        mlflow.log_param("epochs",         epochs)
        mlflow.log_param("batch_size",     batch_size)
        mlflow.log_param("learning_rate",  learning_rate)
        mlflow.log_param("dropout_rate",   dropout_rate)
        mlflow.log_param("optimizer",      "Adam + WeightDecay")
        mlflow.log_param("loss",           "CrossEntropyLoss")

        device = torch.device("cpu")
        modelo = CreditRiskNet(n_features, dropout_rate).to(device)
        criterio  = nn.CrossEntropyLoss()
        otimizador = optim.Adam(modelo.parameters(), lr=learning_rate,
                                weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            otimizador, factor=0.5, patience=7
        )

        train_losses, val_losses = [], []
        train_accs,   val_accs   = [], []
        best_val_loss = float("inf")
        patience_count = 0
        PATIENCE = 15

        print(f"\nTreinando {run_name} por ate {epochs} epocas...")
        for epoch in range(epochs):
            # Treino
            modelo.train()
            t_loss, t_correct, t_total = 0, 0, 0
            for X_b, y_b in train_loader:
                otimizador.zero_grad()
                out  = modelo(X_b)
                loss = criterio(out, y_b)
                loss.backward()
                otimizador.step()
                t_loss    += loss.item() * len(y_b)
                t_correct += (out.argmax(1) == y_b).sum().item()
                t_total   += len(y_b)

            # Validacao
            modelo.eval()
            v_loss, v_correct, v_total = 0, 0, 0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    out  = modelo(X_b)
                    loss = criterio(out, y_b)
                    v_loss    += loss.item() * len(y_b)
                    v_correct += (out.argmax(1) == y_b).sum().item()
                    v_total   += len(y_b)

            tl = t_loss / t_total; ta = t_correct / t_total
            vl = v_loss / v_total; va = v_correct / v_total
            train_losses.append(tl); val_losses.append(vl)
            train_accs.append(ta);   val_accs.append(va)
            scheduler.step(vl)

            # Log por epoca
            mlflow.log_metric("train_loss", round(tl, 4), step=epoch)
            mlflow.log_metric("val_loss",   round(vl, 4), step=epoch)
            mlflow.log_metric("train_acc",  round(ta, 4), step=epoch)
            mlflow.log_metric("val_acc",    round(va, 4), step=epoch)

            if (epoch + 1) % 10 == 0:
                print(f"  Ep {epoch+1:3d} | loss={tl:.4f} val={vl:.4f} "
                      f"acc={ta:.4f} val_acc={va:.4f}")

            # Early stopping
            if vl < best_val_loss:
                best_val_loss = vl
                patience_count = 0
                MODELS_DIR.mkdir(parents=True, exist_ok=True)
                torch.save(modelo.state_dict(),
                           str(MODELS_DIR / f"{run_name}_best.pt"))
            else:
                patience_count += 1
                if patience_count >= PATIENCE:
     
                    print(f"  Early stopping na epoca {epoch+1}")
                    break

        # Avaliar no teste
        modelo.load_state_dict(torch.load(
            str(MODELS_DIR / f"{run_name}_best.pt"), weights_only=True
        ))
        modelo.eval()
        with torch.no_grad():
            logits       = modelo(X_test)
            y_pred_proba = torch.softmax(logits, dim=1).numpy()
            y_pred       = logits.argmax(1).numpy()
            y_true       = y_test.numpy()

        acc = float(np.mean(y_pred == y_true))
        f1  = f1_score(y_true, y_pred, average="macro")
        auc = roc_auc_score(
            pd.get_dummies(pd.Series(y_true)).values,
            y_pred_proba, multi_class="ovr", average="macro"
        )

        mlflow.log_metric("test_accuracy",  round(acc, 4))
        mlflow.log_metric("test_f1_macro",  round(f1,  4))
        mlflow.log_metric("test_auc_roc",   round(auc, 4))
        mlflow.log_metric("epochs_rodadas", len(train_losses))
        mlflow.log_metric("best_val_loss",  round(best_val_loss, 4))

        path_loss = plot_loss_curves(train_losses, val_losses,
                                     train_accs,   val_accs, run_name)
        path_cm   = plot_confusion_matrix(y_true, y_pred, run_name)
        mlflow.log_artifact(path_loss)
        mlflow.log_artifact(path_cm)

        report = classification_report(y_true, y_pred, target_names=CLASSES)
        rpath  = str(FIGS_DIR / f"report_{run_name}.txt")
        with open(rpath, "w") as f:
            f.write(report)
        mlflow.log_artifact(rpath)

        mlflow.pytorch.log_model(modelo, name="modelo_pytorch",
                                  registered_model_name="credit-risk-dl-pytorch")

        print(f"\n{'='*50}")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  F1 Macro:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  Epocas:    {len(train_losses)}")
        print(f"{'='*50}")
        print(report)

    return modelo


def main():
    parser = argparse.ArgumentParser(description="Fase 2 — PyTorch")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout_rate",  type=float, default=0.3)
    args = parser.parse_args()
    treinar(epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, dropout_rate=args.dropout_rate)
    print(f"\nAcesse: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
