#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  MODUL 5: DEMO INFERENSI REAL-TIME
  Smart Load-Diverter Neural Network - Edge AI (TinyML)
===============================================================================
  Fungsi  : Mendemonstrasikan bagaimana model AI mengambil keputusan
            pengalihan daya secara real-time. Mensimulasikan pembacaan
            sensor dan konversi output probabilitas menjadi nilai
            THRESHOLD_REG yang akan ditulis ke hardware via APB bus.

  Alur Simulasi:
    1. Baca data sensor (simulasi)
    2. Normalisasi input ke [0, 1]
    3. Jalankan inferensi TFLite
    4. Konversi probabilitas → THRESHOLD_REG
    5. Tentukan aksi hardware (relay_out ON/OFF)

  Sinkronisasi Hardware:
    - THRESHOLD_REG pada offset 0x08 di smart_diverter_apb.v
    - Probabilitas ≥ 0.5 → turunkan threshold → relay_out = 1
    - Probabilitas < 0.5 → naikkan threshold → relay_out = 0
===============================================================================
"""

import os
import sys
import numpy as np
import pickle
import json

# Pastikan TensorFlow tidak terlalu verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
MODEL_DIR        = "output/model"
TFLITE_FILE      = os.path.join(MODEL_DIR, "smart_diverter.tflite")
SCALER_FILE      = os.path.join(MODEL_DIR, "scaler.pkl")

# Threshold mapping (sesuai spesifikasi di implementation plan)
MAX_THRESHOLD_WATT = 3000   # Threshold normal (tidak alihkan)
MIN_THRESHOLD_WATT = 1000   # Threshold saat AI sangat yakin alihkan
THRESHOLD_REG_ADDR = 0x08   # Offset register di APB bus

# Feature columns (harus sesuai urutan training)
FEATURE_NAMES = ["jam", "cuaca", "daya_surya", "toren_persen", "suhu_heater"]


class SmartDiverterInference:
    """
    Kelas inferensi untuk Smart Load-Diverter.
    Mensimulasikan bagaimana AI bekerja di dalam mikrokontroler.
    """

    def __init__(self):
        """Inisialisasi: muat model TFLite dan scaler."""
        print("  🔧 Inisialisasi Smart Diverter Inference Engine...")

        # Load TFLite model
        if not os.path.exists(TFLITE_FILE):
            raise FileNotFoundError(
                f"Model TFLite tidak ditemukan: '{TFLITE_FILE}'.\n"
                f"Jalankan pipeline lengkap terlebih dahulu!"
            )
        self.interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"     ✅ Model TFLite dimuat ({os.path.getsize(TFLITE_FILE)} bytes)")

        # Load scaler
        with open(SCALER_FILE, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"     ✅ Scaler dimuat")

    def predict(self, jam: float, cuaca: int, daya_surya: float,
                toren_persen: float, suhu_heater: float) -> dict:
        """
        Jalankan inferensi pada satu set pembacaan sensor.

        Args:
            jam          : Jam operasi (6.0 - 18.0)
            cuaca        : Kondisi cuaca (0=Hujan, 1=Mendung, 2=Cerah)
            daya_surya   : Produksi daya surya (Watt)
            toren_persen : Kapasitas toren air (0-100%)
            suhu_heater  : Suhu water heater (°C)

        Returns:
            dict berisi probabilitas, threshold, dan keputusan
        """
        # 1. Susun input array
        raw_input = np.array([[jam, cuaca, daya_surya, toren_persen, suhu_heater]])

        # 2. Normalisasi menggunakan scaler yang sama dengan training
        normalized_input = self.scaler.transform(raw_input).astype(np.float32)

        # 3. Jalankan inferensi TFLite
        self.interpreter.set_tensor(
            self.input_details[0]['index'], normalized_input
        )
        self.interpreter.invoke()
        probability = float(
            self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        )

        # 4. Konversi probabilitas → THRESHOLD_REG
        #    Rumus: threshold = MAX - (prob * (MAX - MIN))
        #    Semakin tinggi prob → threshold semakin rendah → lebih mudah trigger relay
        threshold_watt = int(
            MAX_THRESHOLD_WATT - (probability * (MAX_THRESHOLD_WATT - MIN_THRESHOLD_WATT))
        )

        # 5. Tentukan keputusan
        alihkan = probability >= 0.5
        confidence = probability if alihkan else (1.0 - probability)

        return {
            "input": {
                "jam": jam,
                "cuaca": cuaca,
                "cuaca_label": ["Hujan", "Mendung", "Cerah"][cuaca],
                "daya_surya": daya_surya,
                "toren_persen": toren_persen,
                "suhu_heater": suhu_heater,
            },
            "output": {
                "probabilitas": round(probability, 4),
                "threshold_watt": threshold_watt,
                "threshold_reg_hex": f"0x{threshold_watt:04X}",
                "alihkan": alihkan,
                "confidence": round(confidence * 100, 1),
                "relay_out": 1 if alihkan else 0,
            },
            "hardware": {
                "apb_write": {
                    "PADDR": f"0x{THRESHOLD_REG_ADDR:02X}",
                    "PWRITE": 1,
                    "PWDATA": f"0x{threshold_watt:08X}",
                    "description": (
                        f"Write threshold {threshold_watt}W ke THRESHOLD_REG"
                    ),
                },
            },
        }


def print_prediction(result: dict, index: int = 0):
    """Cetak hasil prediksi dengan format visual."""
    inp = result["input"]
    out = result["output"]
    hw  = result["hardware"]["apb_write"]

    emoji = "🔴 ALIHKAN" if out["alihkan"] else "🟢 NORMAL"
    bar_len = int(out["probabilitas"] * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)

    print(f"\n  ┌─── Skenario #{index + 1} ──────────────────────────────────────────┐")
    print(f"  │ 🕐 Jam: {inp['jam']:<6.1f}  ☁️ Cuaca: {inp['cuaca_label']:<8s} "
          f"  ☀️ Daya: {inp['daya_surya']:.0f}W      │")
    print(f"  │ 💧 Toren: {inp['toren_persen']:<5.1f}%  "
          f"🌡️ Heater: {inp['suhu_heater']:.1f}°C                   │")
    print(f"  │───────────────────────────────────────────────────│")
    print(f"  │ Probabilitas: [{bar}] {out['probabilitas']:.2%}   │")
    print(f"  │ Keputusan   : {emoji:<22s} "
          f"(Confidence: {out['confidence']:.1f}%)   │")
    print(f"  │ Threshold   : {out['threshold_watt']}W "
          f"({out['threshold_reg_hex']})                        │")
    print(f"  │ relay_out   : {out['relay_out']}                   "
          f"                       │")
    print(f"  │───────────────────────────────────────────────────│")
    print(f"  │ APB Write: PADDR={hw['PADDR']}, PWRITE={hw['PWRITE']}, "
          f"PWDATA={hw['PWDATA']}   │")
    print(f"  └─────────────────────────────────────────────────────┘")


def run_demo_scenarios(engine: SmartDiverterInference):
    """Jalankan skenario demo yang mencakup berbagai kondisi."""

    scenarios = [
        # === Skenario ALIHKAN (daya melimpah, storage tersedia) ===
        {
            "desc": "Siang cerah, produksi tinggi, toren kosong",
            "jam": 12.0, "cuaca": 2, "daya_surya": 1800,
            "toren_persen": 20.0, "suhu_heater": 35.0
        },
        {
            "desc": "Siang cerah, produksi tinggi, heater dingin",
            "jam": 11.5, "cuaca": 2, "daya_surya": 1500,
            "toren_persen": 90.0, "suhu_heater": 30.0
        },
        {
            "desc": "Pagi cerah, produksi sedang, storage tersedia",
            "jam": 9.0, "cuaca": 2, "daya_surya": 900,
            "toren_persen": 40.0, "suhu_heater": 45.0
        },
        # === Skenario NORMAL (produksi rendah atau storage penuh) ===
        {
            "desc": "Pagi hujan, produksi rendah",
            "jam": 8.0, "cuaca": 0, "daya_surya": 100,
            "toren_persen": 30.0, "suhu_heater": 40.0
        },
        {
            "desc": "Siang cerah tapi semua storage penuh",
            "jam": 12.0, "cuaca": 2, "daya_surya": 1600,
            "toren_persen": 95.0, "suhu_heater": 85.0
        },
        {
            "desc": "Sore mendung, produksi rendah",
            "jam": 16.0, "cuaca": 1, "daya_surya": 300,
            "toren_persen": 50.0, "suhu_heater": 55.0
        },
        # === Skenario BOUNDARY (dekat ambang batas) ===
        {
            "desc": "Mendung, produksi pas batas, storage setengah",
            "jam": 12.0, "cuaca": 1, "daya_surya": 520,
            "toren_persen": 50.0, "suhu_heater": 50.0
        },
        {
            "desc": "Cerah pagi-pagi, daya mulai naik",
            "jam": 7.0, "cuaca": 2, "daya_surya": 600,
            "toren_persen": 60.0, "suhu_heater": 40.0
        },
    ]

    print(f"\n  🎮 MENJALANKAN {len(scenarios)} SKENARIO DEMO")
    print(f"  {'═' * 55}")

    results = []
    for i, s in enumerate(scenarios):
        print(f"\n  📌 {s['desc']}")
        result = engine.predict(
            jam=s["jam"], cuaca=s["cuaca"], daya_surya=s["daya_surya"],
            toren_persen=s["toren_persen"], suhu_heater=s["suhu_heater"]
        )
        print_prediction(result, i)
        results.append(result)

    return results


def run_interactive_mode(engine: SmartDiverterInference):
    """Mode interaktif: user memasukkan data sensor secara manual."""
    print(f"\n  🎮 MODE INTERAKTIF — Masukkan data sensor untuk prediksi")
    print(f"  Ketik 'q' untuk keluar\n")

    count = 0
    while True:
        try:
            print(f"  {'─' * 50}")
            inp = input("  Jam (6.0-18.0): ").strip()
            if inp.lower() == 'q':
                break
            jam = float(inp)

            cuaca = int(input("  Cuaca (0=Hujan, 1=Mendung, 2=Cerah): ").strip())
            daya_surya = float(input("  Daya Surya (Watt): ").strip())
            toren_persen = float(input("  Toren (0-100%): ").strip())
            suhu_heater = float(input("  Suhu Heater (°C): ").strip())

            result = engine.predict(jam, cuaca, daya_surya,
                                    toren_persen, suhu_heater)
            print_prediction(result, count)
            count += 1

        except ValueError as e:
            print(f"  ⚠️ Input tidak valid: {e}")
        except KeyboardInterrupt:
            break

    print(f"\n  👋 Selesai. Total prediksi: {count}")


def main():
    """Titik masuk utama demo inferensi."""
    print("=" * 70)
    print("  🧠 DEMO INFERENSI REAL-TIME — Smart Load-Diverter (TinyML)")
    print("=" * 70)

    # Inisialisasi engine
    engine = SmartDiverterInference()

    # Jalankan skenario demo
    results = run_demo_scenarios(engine)

    # Ringkasan
    total_alih = sum(1 for r in results if r["output"]["alihkan"])
    total_norm = len(results) - total_alih
    print(f"\n  {'═' * 55}")
    print(f"  📊 RINGKASAN DEMO:")
    print(f"     Total skenario    : {len(results)}")
    print(f"     Alihkan (relay=1) : {total_alih}")
    print(f"     Normal  (relay=0) : {total_norm}")
    print(f"  {'═' * 55}")

    # Simpan hasil demo ke JSON
    demo_output_path = os.path.join("output", "inference_demo_results.json")
    with open(demo_output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  💾 Hasil demo disimpan ke: {demo_output_path}")

    # Tanya mode interaktif
    print(f"\n{'=' * 70}")
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        run_interactive_mode(engine)
    else:
        print("  💡 Untuk mode interaktif, jalankan:")
        print("     python inference_demo.py --interactive")

    print(f"\n{'=' * 70}")
    print("  ✅ DEMO INFERENSI SELESAI!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
