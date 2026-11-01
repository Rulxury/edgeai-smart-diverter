# 🧠 Edge AI (TinyML) — Pipeline Execution Report

> **Smart Load-Diverter Neural Network — Prototipe 2**
> Executed: 2026-11-01 | Total time: 42.8 seconds

---

## ✅ Pipeline Status: ALL STEPS PASSED

| Step | Module | Status | Duration |
|:---:|--------|:---:|:---:|
| 1 | Generate Dataset Sintetis (5000 sampel) | ✅ BERHASIL | 3.4s |
| 2 | Latih Neural Network (MLP) | ✅ BERHASIL | 38.6s |
| 3 | Konversi TFLite (Kuantisasi INT8) | ✅ BERHASIL | 0.8s |
| 4 | Generate C-Header (model.h) | ✅ BERHASIL | 0.0s |
| 5 | Demo Inferensi Real-Time | ✅ BERHASIL | 0.0s |

---

## 📊 Model Performance

| Metrik | Nilai | Target |
|--------|:---:|:---:|
| **Test Accuracy** | **94.80%** | > 90% ✅ |
| **Precision** | 93.04% | — |
| **Recall** | 95.98% | — |
| **F1-Score** | 94.48% | — |
| **Test Loss** | 0.2033 | — |

### Confusion Matrix
|  | Predicted 0 | Predicted 1 |
|:---:|:---:|:---:|
| **Actual 0** | 377 (TN) | 25 (FP) |
| **Actual 1** | 14 (FN) | 334 (TP) |

---

## 📈 Training History

<img width="957" height="342" alt="image" src="https://github.com/user-attachments/assets/388c37e3-3130-44f9-9c30-f1c1fb009b2c" />


- Training dan Validation curves menunjukkan **konvergensi stabil tanpa overfitting**
- Akurasi melampaui target 90% sejak epoch ~15

## 📊 Prediction Analysis

<img width="975" height="380" alt="image" src="https://github.com/user-attachments/assets/8b52535d-89d5-448d-a271-5f2e541d58ba" />


- Distribusi probabilitas menunjukkan **separasi yang jelas** antara kelas Normal (0) dan Alihkan (1)
- Mayoritas prediksi benar (titik hijau), sangat sedikit kesalahan (titik merah)

---

## 📦 Model Compression

| Format | Ukuran | Rasio |
|--------|:---:|:---:|
| Keras (.keras) | 26.73 KB | 1x (baseline) |
| TFLite Float32 | 2.76 KB | 9.7x kompresi |
| **TFLite INT8 (Quantized)** | **3.53 KB** | **7.6x kompresi** |

> [!IMPORTANT]
> Akurasi setelah kuantisasi INT8: **94.84%** : tidak ada penurunan akurasi! (bahkan naik 0.06%)

---

## 🎮 Demo Inferensi — 8 Skenario

| # | Skenario | Prob | Threshold | Keputusan |
|:---:|----------|:---:|:---:|:---:|
| 1 | Siang cerah, produksi tinggi, toren kosong | 98.05% | 1039W | 🔴 ALIHKAN |
| 2 | Siang cerah, produksi tinggi, heater dingin | 90.72% | 1185W | 🔴 ALIHKAN |
| 3 | Pagi cerah, produksi sedang, storage tersedia | 94.89% | 1102W | 🔴 ALIHKAN |
| 4 | Pagi hujan, produksi rendah | 2.97% | 2940W | 🟢 NORMAL |
| 5 | Siang cerah tapi semua storage penuh | 48.83% | 2023W | 🟢 NORMAL |
| 6 | Sore mendung, produksi rendah | 2.73% | 2945W | 🟢 NORMAL |
| 7 | Mendung, produksi pas batas | 64.84% | 1703W | 🔴 ALIHKAN |
| 8 | Cerah pagi, daya mulai naik | 94.92% | 1101W | 🔴 ALIHKAN |

---

## 📁 Output Files

```
output/
├── dummy_energy_data.csv         (130 KB)  → Dataset 5000 sampel
├── training_history.png          (94 KB)   → Grafik Loss & Accuracy
├── prediction_analysis.png       (192 KB)  → Analisis distribusi prediksi
├── training_metrics.json                   → Metrik evaluasi lengkap
├── inference_demo_results.json             → Hasil 8 skenario demo
└── model/
    ├── smart_diverter_model.keras (26 KB)  → Model Keras asli
    ├── scaler.pkl                          → MinMaxScaler (untuk normalisasi)
    ├── architecture_info.json              → Informasi arsitektur NN
    ├── smart_diverter_float.tflite (2.7 KB)→ TFLite float32
    ├── smart_diverter.tflite     (3.5 KB)  → TFLite INT8 ← UNTUK SoC
    ├── tflite_conversion_info.json         → Info konversi & akurasi
    └── model.h                   (25 KB)   → C-Header ← UNTUK FIRMWARE
```

## 🔌 Langkah Deployment ke Hardware

1. Copy [`model.h`](file:///c:/edgeai_seno/output/model/model.h) ke proyek firmware ARM Cortex
2. Lihat [`firmware_example.c`](file:///c:/edgeai_seno/firmware_example.c) sebagai referensi integrasi
3. Compile bersama TensorFlow Lite for Microcontrollers
4. AI akan menulis threshold ke `THRESHOLD_REG` (offset `0x08`) di `smart_diverter_apb.v` via APB bus
