#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  MASTER PIPELINE — Smart Load-Diverter Edge AI (TinyML)
===============================================================================
  Menjalankan seluruh pipeline Edge AI secara otomatis dan berurutan:

    STEP 1: Generate Dataset Sintetis  → dummy_energy_data.csv
    STEP 2: Latih Neural Network (MLP) → smart_diverter_model.keras
    STEP 3: Konversi ke TFLite (INT8)  → smart_diverter.tflite
    STEP 4: Generate C-Header          → model.h
    STEP 5: Demo Inferensi Real-Time   → inference_demo_results.json

  Penggunaan:
    python run_pipeline.py              # Jalankan semua step
    python run_pipeline.py --step 1     # Jalankan step tertentu
    python run_pipeline.py --from 3     # Jalankan mulai dari step 3
===============================================================================
"""

import os
import sys
import time
import argparse
import importlib

# Pastikan direktori proyek ada di sys.path (untuk embedded Python)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
STEPS = [
    {
        "num": 1,
        "name": "Generate Dataset Sintetis",
        "module": "generate_dataset",
        "output": ["output/dummy_energy_data.csv"],
        "description": "Membangkitkan dataset CSV dengan 5000 sampel skenario energi surya"
    },
    {
        "num": 2,
        "name": "Latih Neural Network (MLP)",
        "module": "train_model",
        "output": [
            "output/model/smart_diverter_model.keras",
            "output/model/scaler.pkl",
            "output/training_history.png",
            "output/prediction_analysis.png",
            "output/training_metrics.json"
        ],
        "description": "Melatih model MLP: Input(5)→Dense(8,ReLU)→Dense(4,ReLU)→Dense(1,Sigmoid)"
    },
    {
        "num": 3,
        "name": "Konversi ke TFLite (Kuantisasi INT8)",
        "module": "convert_to_tflite",
        "output": [
            "output/model/smart_diverter.tflite",
            "output/model/smart_diverter_float.tflite",
            "output/model/tflite_conversion_info.json"
        ],
        "description": "Mengompresi model dari 32-bit float ke 8-bit integer untuk ARM Cortex-M"
    },
    {
        "num": 4,
        "name": "Generate C-Header (model.h)",
        "module": "generate_c_header",
        "output": ["output/model/model.h"],
        "description": "Mengkonversi TFLite binary ke C-array hexadecimal untuk firmware"
    },
    {
        "num": 5,
        "name": "Demo Inferensi Real-Time",
        "module": "inference_demo",
        "output": ["output/inference_demo_results.json"],
        "description": "Menjalankan demo prediksi pengalihan daya pada berbagai skenario"
    },
]


def print_banner():
    """Cetak banner pipeline."""
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         🧠 EDGE AI (TinyML) — FULL PIPELINE                    ║")
    print("║         Smart Load-Diverter Neural Network                     ║")
    print("║         Prototipe 2: Hardware-Software Synchronization         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()


def print_step_header(step: dict, total: int):
    """Cetak header untuk setiap step."""
    print()
    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  STEP {step['num']}/{total}: {step['name']:<50s}  ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")
    print(f"  📝 {step['description']}")
    print()


def run_step(step: dict) -> bool:
    """
    Jalankan satu step pipeline dengan dynamic import.
    Returns True jika berhasil, False jika gagal.
    """
    try:
        # Dynamic import module
        module = importlib.import_module(step["module"])

        # Reload jika sudah pernah diimport (untuk re-run)
        importlib.reload(module)

        # Jalankan fungsi main()
        if hasattr(module, 'main'):
            module.main()
        else:
            raise AttributeError(
                f"Module '{step['module']}' tidak memiliki fungsi main()"
            )

        # Verifikasi output
        missing = []
        for output_file in step["output"]:
            if not os.path.exists(output_file):
                missing.append(output_file)

        if missing:
            print(f"\n  ⚠️ WARNING: File output berikut tidak ditemukan:")
            for f in missing:
                print(f"     - {f}")
            return True  # Tetap lanjut, mungkin path berbeda

        return True

    except Exception as e:
        print(f"\n  ❌ ERROR pada Step {step['num']}:")
        print(f"     {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary(results: list, elapsed: float):
    """Cetak ringkasan hasil pipeline."""
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                    📊 RINGKASAN PIPELINE                       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    all_success = True
    for step, success, duration in results:
        status = "✅ BERHASIL" if success else "❌ GAGAL"
        if not success:
            all_success = False
        print(f"  Step {step['num']}: {step['name']:<35s} {status} ({duration:.1f}s)")

    print(f"\n  {'─' * 60}")
    print(f"  ⏱️ Total waktu: {elapsed:.1f} detik ({elapsed/60:.1f} menit)")

    if all_success:
        print(f"\n  ✅ SEMUA STEP BERHASIL!")
        print(f"\n  📁 Output Files:")
        print(f"     output/")
        print(f"     ├── dummy_energy_data.csv        → Dataset sintetis")
        print(f"     ├── training_history.png          → Grafik Loss & Accuracy")
        print(f"     ├── prediction_analysis.png       → Analisis prediksi")
        print(f"     ├── training_metrics.json         → Metrik evaluasi")
        print(f"     ├── inference_demo_results.json   → Hasil demo inferensi")
        print(f"     └── model/")
        print(f"         ├── smart_diverter_model.keras → Model Keras asli")
        print(f"         ├── scaler.pkl                 → MinMaxScaler")
        print(f"         ├── architecture_info.json     → Info arsitektur")
        print(f"         ├── smart_diverter_float.tflite → TFLite float32")
        print(f"         ├── smart_diverter.tflite       → TFLite INT8 (untuk SoC)")
        print(f"         ├── tflite_conversion_info.json → Info konversi")
        print(f"         └── model.h                     → C-Header (untuk firmware)")
        print(f"\n  🎯 SIAP UNTUK DEPLOYMENT KE ARM CORTEX-M!")
        print(f"     1. Copy 'model.h' ke proyek firmware Anda")
        print(f"     2. Lihat 'firmware_example.c' sebagai referensi integrasi")
        print(f"     3. Compile bersama TFLite for Microcontrollers")
    else:
        print(f"\n  ⚠️ Ada step yang gagal. Periksa error di atas.")


def parse_args():
    """Parse argumen command line."""
    parser = argparse.ArgumentParser(
        description="Smart Load-Diverter Edge AI — Full Pipeline"
    )
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4, 5],
        help="Jalankan hanya step tertentu"
    )
    parser.add_argument(
        "--from-step", type=int, choices=[1, 2, 3, 4, 5], default=1,
        dest="from_step",
        help="Jalankan mulai dari step tertentu (default: 1)"
    )
    return parser.parse_args()


def main():
    """Titik masuk utama pipeline."""
    args = parse_args()

    print_banner()

    # Tentukan step mana yang dijalankan
    if args.step:
        steps_to_run = [s for s in STEPS if s["num"] == args.step]
    else:
        steps_to_run = [s for s in STEPS if s["num"] >= args.from_step]

    total = len(steps_to_run)
    print(f"  📋 Akan menjalankan {total} step:")
    for s in steps_to_run:
        print(f"     Step {s['num']}: {s['name']}")

    # Jalankan pipeline
    results = []
    start_total = time.time()

    for step in steps_to_run:
        print_step_header(step, len(STEPS))
        start_step = time.time()
        success = run_step(step)
        duration = time.time() - start_step
        results.append((step, success, duration))

        if not success:
            print(f"\n  ⛔ Pipeline berhenti karena Step {step['num']} gagal.")
            break

    elapsed_total = time.time() - start_total

    # Cetak ringkasan
    print_summary(results, elapsed_total)


if __name__ == "__main__":
    main()
