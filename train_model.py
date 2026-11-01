#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  MODUL 2: SKRIP PELATIHAN NEURAL NETWORK (Lightweight MLP)
  Smart Load-Diverter Neural Network - Edge AI (TinyML)
===============================================================================
  Fungsi  : Membangun dan melatih model Deep Learning berbasis Multi-Layer
            Perceptron (MLP) yang sangat efisien untuk ditanamkan ke SoC
            ARM Cortex sebagai "Otak" pengambil keputusan.

  Arsitektur:
    - Input Layer  : 5 Neuron (jam, cuaca, daya_surya, toren_persen, suhu_heater)
    - Hidden Layer 1: 8 Neuron, aktivasi ReLU
    - Hidden Layer 2: 4 Neuron, aktivasi ReLU
    - Output Layer : 1 Neuron, aktivasi Sigmoid → Probabilitas (0.0 - 1.0)

  Sinkronisasi ke Hardware:
    Model output → probabilitas → CPU mengkonversi ke threshold →
    ditulis ke THRESHOLD_REG (offset 0x08) via APB bus →
    IP Core Verilog memicu relay_out
===============================================================================
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend untuk server/headless
import matplotlib.pyplot as plt
import json
import pickle

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
INPUT_DIR       = "output"
DATASET_FILE    = "dummy_energy_data.csv"
OUTPUT_DIR      = "output"
MODEL_DIR       = "output/model"

# Hyperparameters
EPOCHS          = 100
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001
VALIDATION_SPLIT = 0.2   # 20% data untuk validasi
TEST_SPLIT      = 0.15   # 15% data untuk testing

# Neural Network Architecture
HIDDEN_1_UNITS  = 8       # Neuron di hidden layer 1
HIDDEN_2_UNITS  = 4       # Neuron di hidden layer 2

# Early stopping patience
PATIENCE        = 15

# Feature columns
FEATURE_COLS = ["jam", "cuaca", "daya_surya", "toren_persen", "suhu_heater"]
LABEL_COL    = "alihkan"

# Random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_dataset() -> pd.DataFrame:
    """Muat dataset dari CSV."""
    path = os.path.join(INPUT_DIR, DATASET_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di '{path}'.\n"
            f"Jalankan 'generate_dataset.py' terlebih dahulu!"
        )
    df = pd.read_csv(path)
    print(f"  📂 Dataset dimuat: {path} ({len(df)} sampel)")
    return df


def preprocess_data(df: pd.DataFrame):
    """
    Preprocessing: normalisasi fitur ke rentang [0, 1] menggunakan MinMaxScaler.
    Split data menjadi train, validation, dan test.
    """
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    # Normalisasi [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # Split: train + val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )

    # Split: train vs val
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED, stratify=y_trainval
    )

    print(f"  📊 Split Data:")
    print(f"     Training   : {len(X_train):>5d} sampel")
    print(f"     Validation : {len(X_val):>5d} sampel")
    print(f"     Testing    : {len(X_test):>5d} sampel")
    print(f"     Total      : {len(X_train) + len(X_val) + len(X_test):>5d} sampel")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def build_model() -> keras.Model:
    """
    Bangun arsitektur MLP sesuai spesifikasi:
      Input(5) → Dense(8, ReLU) → Dense(4, ReLU) → Dense(1, Sigmoid)
    """
    model = keras.Sequential([
        layers.Input(shape=(len(FEATURE_COLS),), name="sensor_input"),

        # Hidden Layer 1: 8 neuron, ReLU
        layers.Dense(
            HIDDEN_1_UNITS,
            activation='relu',
            kernel_initializer='he_uniform',
            name="hidden_layer_1"
        ),

        # Hidden Layer 2: 4 neuron, ReLU
        layers.Dense(
            HIDDEN_2_UNITS,
            activation='relu',
            kernel_initializer='he_uniform',
            name="hidden_layer_2"
        ),

        # Output Layer: 1 neuron, Sigmoid → probabilitas
        layers.Dense(
            1,
            activation='sigmoid',
            name="output_probability"
        )
    ], name="SmartDiverter_MLP")

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(model: keras.Model,
                X_train, y_train,
                X_val, y_val) -> keras.callbacks.History:
    """
    Latih model dengan early stopping dan learning rate reduction.
    """
    # Callbacks
    cb_list = [
        # Early stopping: hentikan jika val_loss tidak membaik
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce LR on plateau
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print(f"\n  🧠 Memulai pelatihan Neural Network...")
    print(f"     Epochs max     : {EPOCHS}")
    print(f"     Batch size     : {BATCH_SIZE}")
    print(f"     Learning rate  : {LEARNING_RATE}")
    print(f"     Early stopping : patience={PATIENCE}")
    print(f"  {'─' * 50}")

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=cb_list,
        verbose=1
    )

    return history


def evaluate_model(model: keras.Model,
                   X_test, y_test,
                   history: keras.callbacks.History):
    """
    Evaluasi model pada test set dan tampilkan metrik.
    """
    print(f"\n  📊 EVALUASI MODEL")
    print(f"  {'=' * 50}")

    # Evaluasi pada test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  🎯 Test Loss     : {test_loss:.4f}")
    print(f"  🎯 Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Prediksi probabilitas
    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Confusion matrix manual
    tp = np.sum((y_pred == 1) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score  = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0

    print(f"\n  📋 Confusion Matrix:")
    print(f"     {'':>15s}  Predicted 0   Predicted 1")
    print(f"     {'Actual 0':>15s}  {tn:>10d}   {fp:>10d}")
    print(f"     {'Actual 1':>15s}  {fn:>10d}   {tp:>10d}")
    print(f"\n  📋 Metrik Klasifikasi:")
    print(f"     Precision : {precision:.4f}")
    print(f"     Recall    : {recall:.4f}")
    print(f"     F1-Score  : {f1_score:.4f}")

    # Simpan metrik
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "confusion_matrix": {
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn)
        },
        "epochs_trained": len(history.history['loss']),
        "final_train_loss": float(history.history['loss'][-1]),
        "final_val_loss": float(history.history['val_loss'][-1]),
        "final_train_accuracy": float(history.history['accuracy'][-1]),
        "final_val_accuracy": float(history.history['val_accuracy'][-1]),
    }

    metrics_path = os.path.join(OUTPUT_DIR, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  💾 Metrik disimpan ke: {metrics_path}")

    return metrics


def plot_training_history(history: keras.callbacks.History):
    """
    Plot grafik Loss dan Accuracy per epoch.
    Bukti Proof-of-Concept untuk proposal/juri.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Smart Load-Diverter Neural Network — Training History',
                 fontsize=14, fontweight='bold')

    epochs_range = range(1, len(history.history['loss']) + 1)

    # ── Plot 1: Loss ──
    ax1 = axes[0]
    ax1.plot(epochs_range, history.history['loss'],
             'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.plot(epochs_range, history.history['val_loss'],
             'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    ax1.set_title('Model Loss per Epoch', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (Binary Cross-Entropy)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=1)

    # ── Plot 2: Accuracy ──
    ax2 = axes[1]
    ax2.plot(epochs_range, history.history['accuracy'],
             'b-', linewidth=2, label='Training Accuracy', alpha=0.8)
    ax2.plot(epochs_range, history.history['val_accuracy'],
             'r-', linewidth=2, label='Validation Accuracy', alpha=0.8)
    ax2.set_title('Model Accuracy per Epoch', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=1)
    ax2.set_ylim(0, 1.05)

    # Garis target 90%
    ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5,
                label='Target 90%')
    ax2.legend(loc='lower right')

    plt.tight_layout()

    # Simpan plot
    plot_path = os.path.join(OUTPUT_DIR, "training_history.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Grafik training history disimpan ke: {plot_path}")

    return plot_path


def plot_prediction_distribution(model: keras.Model,
                                  X_test, y_test):
    """
    Plot distribusi probabilitas prediksi model.
    """
    y_pred_prob = model.predict(X_test, verbose=0).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Smart Load-Diverter — Prediction Analysis',
                 fontsize=14, fontweight='bold')

    # ── Plot 1: Distribusi probabilitas ──
    ax1 = axes[0]
    ax1.hist(y_pred_prob[y_test == 0], bins=50, alpha=0.7,
             label='Normal (0)', color='steelblue', edgecolor='black')
    ax1.hist(y_pred_prob[y_test == 1], bins=50, alpha=0.7,
             label='Alihkan (1)', color='coral', edgecolor='black')
    ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=2,
                label='Threshold 0.5')
    ax1.set_title('Distribusi Probabilitas Prediksi', fontsize=12,
                  fontweight='bold')
    ax1.set_xlabel('Probabilitas Output Sigmoid')
    ax1.set_ylabel('Jumlah Sampel')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Plot 2: Scatter actual vs predicted ──
    ax2 = axes[1]
    correct = ((y_pred_prob >= 0.5).astype(int) == y_test)
    ax2.scatter(range(len(y_pred_prob)), y_pred_prob,
                c=np.where(correct, 'green', 'red'), alpha=0.5, s=10)
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Prediksi per Sampel (Hijau=Benar, Merah=Salah)',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sampel Index')
    ax2.set_ylabel('Probabilitas Prediksi')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "prediction_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📊 Grafik prediction analysis disimpan ke: {plot_path}")


def save_model_and_scaler(model: keras.Model, scaler):
    """Simpan model Keras (.keras) dan scaler (pickle)."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Simpan model Keras
    model_path = os.path.join(MODEL_DIR, "smart_diverter_model.keras")
    model.save(model_path)
    print(f"  💾 Model Keras disimpan ke: {model_path}")

    # Simpan scaler
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  💾 Scaler disimpan ke: {scaler_path}")

    # Simpan informasi arsitektur
    arch_info = {
        "model_name": "SmartDiverter_MLP",
        "input_features": FEATURE_COLS,
        "architecture": [
            {"layer": "Input", "units": len(FEATURE_COLS)},
            {"layer": "Dense", "units": HIDDEN_1_UNITS, "activation": "relu"},
            {"layer": "Dense", "units": HIDDEN_2_UNITS, "activation": "relu"},
            {"layer": "Dense", "units": 1, "activation": "sigmoid"},
        ],
        "total_parameters": model.count_params(),
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss_function": "binary_crossentropy",
        "normalization": "MinMaxScaler [0, 1]",
        "feature_ranges": {
            "jam": [6.0, 18.0],
            "cuaca": [0, 2],
            "daya_surya": [0, 2500],
            "toren_persen": [0, 100],
            "suhu_heater": [25, 90],
        }
    }
    arch_path = os.path.join(MODEL_DIR, "architecture_info.json")
    with open(arch_path, 'w') as f:
        json.dump(arch_info, f, indent=2)
    print(f"  💾 Info arsitektur disimpan ke: {arch_path}")


def main():
    """Titik masuk utama pelatihan."""
    print("=" * 70)
    print("  🧠 PELATIHAN NEURAL NETWORK — Smart Load-Diverter (TinyML)")
    print("=" * 70)

    # Buat direktori output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load dataset
    df = load_dataset()

    # 2. Preprocessing
    print(f"\n  ⚙️ Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(df)

    # 3. Build model
    print(f"\n  🏗️ Membangun arsitektur Neural Network...")
    model = build_model()
    model.summary()

    # 4. Train model
    history = train_model(model, X_train, y_train, X_val, y_val)

    # 5. Evaluate model
    metrics = evaluate_model(model, X_test, y_test, history)

    # 6. Plot training history
    print(f"\n  📊 Membuat grafik evaluasi...")
    plot_training_history(history)
    plot_prediction_distribution(model, X_test, y_test)

    # 7. Save model and scaler
    print(f"\n  💾 Menyimpan model dan scaler...")
    save_model_and_scaler(model, scaler)

    # 8. Summary
    print(f"\n{'=' * 70}")
    print(f"  ✅ PELATIHAN SELESAI!")
    print(f"  {'─' * 50}")
    print(f"  📊 Test Accuracy : {metrics['test_accuracy']*100:.2f}%")
    print(f"  📊 F1-Score      : {metrics['f1_score']:.4f}")
    print(f"  📊 Epochs Trained: {metrics['epochs_trained']}")
    print(f"  📊 Total Params  : {model.count_params()}")
    target_met = "✅ YA" if metrics['test_accuracy'] >= 0.90 else "⚠️ BELUM"
    print(f"  🎯 Target >90%   : {target_met}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
