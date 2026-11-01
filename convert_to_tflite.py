#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  MODUL 3: KONVERSI MODEL KE TFLITE (Kuantisasi)
  Smart Load-Diverter Neural Network - Edge AI (TinyML)
===============================================================================
  Fungsi  : Mengompresi model AI yang tadinya berat (32-bit float) menjadi
            versi TinyML super ringan (8-bit integer) yang bisa masuk ke
            memori mikrokontroler (Edge Device / ARM Cortex).

  Proses  :
    1. Load model Keras (.keras) yang sudah dilatih
    2. Konversi ke TFLite menggunakan tf.lite.TFLiteConverter
    3. Terapkan kuantisasi (Dynamic Range + Full Integer)
    4. Simpan file .tflite yang sangat kecil (beberapa KB)
    5. Validasi akurasi model terkuantisasi vs model asli

  Hasil   : File 'smart_diverter.tflite' siap ditanamkan ke ARM Cortex-M
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle
import json

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
INPUT_DIR    = "output"
MODEL_DIR    = "output/model"
DATASET_FILE = os.path.join(INPUT_DIR, "dummy_energy_data.csv")
MODEL_FILE   = os.path.join(MODEL_DIR, "smart_diverter_model.keras")
SCALER_FILE  = os.path.join(MODEL_DIR, "scaler.pkl")

# Output TFLite
TFLITE_FLOAT_FILE  = os.path.join(MODEL_DIR, "smart_diverter_float.tflite")
TFLITE_QUANT_FILE  = os.path.join(MODEL_DIR, "smart_diverter.tflite")

FEATURE_COLS = ["jam", "cuaca", "daya_surya", "toren_persen", "suhu_heater"]
LABEL_COL    = "alihkan"


def load_resources():
    """Muat model Keras, scaler, dan dataset."""
    print(f"  📂 Memuat model Keras: {MODEL_FILE}")
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model tidak ditemukan di '{MODEL_FILE}'.\n"
            f"Jalankan 'train_model.py' terlebih dahulu!"
        )
    model = tf.keras.models.load_model(MODEL_FILE)

    print(f"  📂 Memuat scaler: {SCALER_FILE}")
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)

    print(f"  📂 Memuat dataset: {DATASET_FILE}")
    df = pd.read_csv(DATASET_FILE)

    return model, scaler, df


def representative_dataset_gen(df, scaler, num_samples=200):
    """
    Generator dataset representatif untuk kuantisasi full-integer.
    TFLite memerlukan ini untuk menentukan range nilai optimal
    saat mengkonversi bobot dari float32 ke int8.
    """
    X = df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)

    # Ambil subset representatif
    indices = np.random.choice(len(X_scaled), size=min(num_samples, len(X_scaled)),
                               replace=False)

    def gen():
        for idx in indices:
            sample = X_scaled[idx:idx+1].astype(np.float32)
            yield [sample]

    return gen


def convert_to_tflite_float(model):
    """Konversi model ke TFLite tanpa kuantisasi (baseline)."""
    print(f"\n  🔄 Konversi ke TFLite (Float32 — baseline)...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(TFLITE_FLOAT_FILE, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  ✅ TFLite Float32 disimpan: {TFLITE_FLOAT_FILE}")
    print(f"     Ukuran: {size_kb:.2f} KB ({len(tflite_model)} bytes)")

    return tflite_model


def convert_to_tflite_quantized(model, df, scaler):
    """
    Konversi model ke TFLite dengan kuantisasi (Dynamic Range + INT8).
    Ini menghasilkan model super ringan untuk mikrokontroler.
    """
    print(f"\n  🔄 Konversi ke TFLite (Quantized INT8)...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimasi: Dynamic range quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset untuk full integer quantization
    rep_gen = representative_dataset_gen(df, scaler)
    converter.representative_dataset = rep_gen

    # Target spec: INT8 ops untuk ARM Cortex
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    # Input/output tetap float32 untuk kemudahan penggunaan
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()

    with open(TFLITE_QUANT_FILE, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"  ✅ TFLite Quantized disimpan: {TFLITE_QUANT_FILE}")
    print(f"     Ukuran: {size_kb:.2f} KB ({len(tflite_model)} bytes)")

    return tflite_model


def validate_tflite_model(tflite_model_bytes, df, scaler, label="TFLite"):
    """
    Validasi akurasi model TFLite terhadap dataset.
    Memastikan kuantisasi tidak mengurangi kecerdasan secara signifikan.
    """
    print(f"\n  🔍 Validasi Model {label}...")

    # Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"     Input shape  : {input_details[0]['shape']}")
    print(f"     Input dtype  : {input_details[0]['dtype']}")
    print(f"     Output shape : {output_details[0]['shape']}")
    print(f"     Output dtype : {output_details[0]['dtype']}")

    # Prepare data
    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values
    X_scaled = scaler.transform(X).astype(np.float32)

    # Predict each sample
    predictions = []
    for i in range(len(X_scaled)):
        input_data = X_scaled[i:i+1]
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output[0][0])

    predictions = np.array(predictions)
    y_pred = (predictions >= 0.5).astype(int)

    # Hitung akurasi
    accuracy = np.mean(y_pred == y)
    print(f"     Akurasi {label}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    return accuracy


def compare_model_sizes(tflite_float, tflite_quant):
    """Bandingkan ukuran model."""
    print(f"\n  📏 PERBANDINGAN UKURAN MODEL:")
    print(f"  {'─' * 50}")

    keras_size = os.path.getsize(MODEL_FILE)
    float_size = len(tflite_float)
    quant_size = len(tflite_quant)

    print(f"     Model Keras (float32)   : {keras_size/1024:>8.2f} KB")
    print(f"     TFLite Float32          : {float_size/1024:>8.2f} KB")
    print(f"     TFLite Quantized (INT8) : {quant_size/1024:>8.2f} KB")
    print(f"  {'─' * 50}")
    print(f"     Rasio kompresi (Keras → Quant): "
          f"{keras_size/quant_size:.1f}x lebih kecil")
    print(f"     Rasio kompresi (Float → Quant): "
          f"{float_size/quant_size:.1f}x lebih kecil")

    return {
        "keras_size_bytes": keras_size,
        "tflite_float_size_bytes": float_size,
        "tflite_quant_size_bytes": quant_size,
        "compression_ratio_keras_to_quant": round(keras_size / quant_size, 2),
        "compression_ratio_float_to_quant": round(float_size / quant_size, 2),
    }


def main():
    """Titik masuk utama konversi TFLite."""
    print("=" * 70)
    print("  📦 KONVERSI MODEL KE TFLITE — Smart Load-Diverter (TinyML)")
    print("=" * 70)

    # 1. Load resources
    model, scaler, df = load_resources()

    # 2. Konversi ke TFLite Float32 (baseline)
    tflite_float = convert_to_tflite_float(model)

    # 3. Konversi ke TFLite Quantized INT8
    tflite_quant = convert_to_tflite_quantized(model, df, scaler)

    # 4. Validasi kedua model
    acc_float = validate_tflite_model(tflite_float, df, scaler, "Float32")
    acc_quant = validate_tflite_model(tflite_quant, df, scaler, "Quantized")

    # 5. Bandingkan ukuran
    size_info = compare_model_sizes(tflite_float, tflite_quant)

    # 6. Simpan info konversi
    conversion_info = {
        "tflite_float_file": TFLITE_FLOAT_FILE,
        "tflite_quant_file": TFLITE_QUANT_FILE,
        "accuracy_float": float(acc_float),
        "accuracy_quant": float(acc_quant),
        "accuracy_drop": float(acc_float - acc_quant),
        **size_info
    }

    info_path = os.path.join(MODEL_DIR, "tflite_conversion_info.json")
    with open(info_path, 'w') as f:
        json.dump(conversion_info, f, indent=2)
    print(f"\n  💾 Info konversi disimpan ke: {info_path}")

    # 7. Summary
    print(f"\n{'=' * 70}")
    print(f"  ✅ KONVERSI TFLITE SELESAI!")
    print(f"  {'─' * 50}")
    print(f"  📊 Akurasi Float32  : {acc_float*100:.2f}%")
    print(f"  📊 Akurasi Quantized: {acc_quant*100:.2f}%")
    print(f"  📊 Penurunan akurasi: {(acc_float-acc_quant)*100:.2f}%")
    print(f"  📦 Ukuran Quantized : {size_info['tflite_quant_size_bytes']/1024:.2f} KB")
    print(f"  🎯 Siap untuk ARM Cortex-M!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
