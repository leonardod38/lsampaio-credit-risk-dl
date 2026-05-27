"""
train.py — Fase 2: Rede neural multiclasse com TensorFlow/Keras.

Arquitetura com BatchNormalization, Dropout e regularização L2.
Tracking completo no MLflow: hiperparâmetros, loss curves,
confusion matrix e classification report.

Autor: Leonardo Sampaio
"""

import argparse
import sys
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

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


def carregar_dados():
    """Carrega os splits gerados pela Fase 1."""
    for fname in ["credito_train.csv", "credito_val.csv", "credito_test.csv"]:
        if not (DATA_DIR / fname).exists():
            print(f"Arquivo {fname} não encontrado. Execute feature_engineering.py.")
            sys.exit(1)

    df_train = pd.read_csv(DATA_DIR / "credito_train.csv")
    df_val   = pd.read_csv(DATA_DIR / "credito_val.csv")
    df_test  = pd.read_csv(DATA_DIR / "credito_test.csv")

    X_train = df_train[FEATURES].values
    y_train = df_train["label"].values
    X_val   = df_val[FEATURES].values
    y_val   = df_val["label"].values
    X_test  = df_test[FEATURES].values
    y_test  = df_test["label"].values

    return X_train, y_train, X_val, y_val, X_test, y_test


def construir_modelo(
    n_features: int,
    learning_rate: float = 0.001,
    dropout_rate: float  = 0.3,
    l2_lambda: float     = 0.001,
) -> keras.Model:
    """
    Rede neural densa com BatchNormalization e Dropout.
    Arquitetura: Input → 256 → 128 → 64 → 32 → Softmax(4)
    """
    inputs = keras.Input(shape=(n_features,), name="input")

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(l2_lambda))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate * 0.7)(x)

    x = layers.Dense(64, kernel_regularizer=regularizers.l2(l2_lambda))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)

    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(N_CLASSES, activation="softmax", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="credit_risk_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def plot_loss_curves(history, run_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(history.history["loss"],     label="Treino",    color="#185FA5", linewidth=2)
    ax1.plot(history.history["val_loss"], label="Validacao", color="#D85A30", linewidth=2)
    ax1.set_title("Loss por Epoca", fontsize=12)
    ax1.set_xlabel("Epoca"); ax1.set_ylabel("Loss"); ax1.legend()
    ax1.spines[["top", "right"]].set_visible(False)
    ax2.plot(history.history["accuracy"],     label="Treino",    color="#185FA5", linewidth=2)
    ax2.plot(history.history["val_accuracy"], label="Validacao", color="#D85A30", linewidth=2)
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


def treinar(epochs=100, batch_size=64, learning_rate=0.001, dropout_rate=0.3, run_name="fase2_rede_neural"):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    X_train, y_train, X_val, y_val, X_test, y_test = carregar_dados()
    n_features = X_train.shape[1]
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("arquitetura",   "256-128-64-32-softmax4")
        mlflow.log_param("epochs",        epochs)
        mlflow.log_param("batch_size",    batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("dropout_rate",  dropout_rate)
        mlflow.log_param("optimizer",     "Adam")
        modelo = construir_modelo(n_features, learning_rate, dropout_rate)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6, verbose=1),
        ]
        history = modelo.fit(X_train, y_train, validation_data=(X_val, y_val),
                             epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
        y_pred_proba = modelo.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        acc = float(np.mean(y_pred == y_test))
        f1  = f1_score(y_test, y_pred, average="macro")
        auc = roc_auc_score(tf.keras.utils.to_categorical(y_test, N_CLASSES),
                            y_pred_proba, multi_class="ovr", average="macro")
        mlflow.log_metric("test_accuracy", round(acc, 4))
        mlflow.log_metric("test_f1_macro", round(f1,  4))
        mlflow.log_metric("test_auc_roc",  round(auc, 4))
        mlflow.log_metric("epochs_rodadas", len(history.history["loss"]))
        path_loss = plot_loss_curves(history, run_name)
        path_cm   = plot_confusion_matrix(y_test, y_pred, run_name)
        mlflow.log_artifact(path_loss); mlflow.log_artifact(path_cm)
        report = classification_report(y_test, y_pred, target_names=CLASSES)
        rpath  = str(FIGS_DIR / f"report_{run_name}.txt")
        with open(rpath, "w") as f: f.write(report)
        mlflow.log_artifact(rpath)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        modelo.save(str(MODELS_DIR / f"modelo_{run_name}.keras"))
        mlflow.tensorflow.log_model(modelo, name="modelo_deep_learning",
                                    registered_model_name="credit-risk-dl")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return modelo


def main():
    parser = argparse.ArgumentParser(description="Fase 2 — Treinamento Deep Learning")
    parser.add_argument("--epochs",        type=int,   default=100)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout_rate",  type=float, default=0.3)
    args = parser.parse_args()
    treinar(epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, dropout_rate=args.dropout_rate)
    print(f"\nAcesse a UI: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
