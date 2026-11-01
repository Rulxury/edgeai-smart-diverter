# 🧠 Edge AI (TinyML) — Smart Load-Diverter Neural Network

> **Prototipe 2:** Smart Load-Diverter dengan sinkronisasi Hardware-Software  
> AI berbasis Neural Network yang ditanamkan ke SoC ARM Cortex untuk pengambilan keputusan pengalihan daya surya secara *real-time*.

---

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan sistem AI Edge (*TinyML*) yang berfungsi sebagai "Otak" pengambil keputusan untuk mengalihkan kelebihan energi surya ke penyimpanan (toren air / water heater) agar tidak bocor ke jaringan PLN (*Grid-Defection curtailment*).

### Arsitektur Neural Network
```
Input(5) → Dense(8, ReLU) → Dense(4, ReLU) → Dense(1, Sigmoid)
```

| Layer | Neuron | Aktivasi | Fungsi |
|-------|--------|----------|--------|
| Input | 5 | - | jam, cuaca, daya_surya, toren_persen, suhu_heater |
| Hidden 1 | 8 | ReLU | Ekstraksi pola utama |
| Hidden 2 | 4 | ReLU | Penyederhanaan fitur |
| Output | 1 | Sigmoid | Probabilitas pengalihan (0.0 - 1.0) |

### Sinkronisasi Hardware
```
AI Probability → CPU Threshold Calculation → APB Write → THRESHOLD_REG (0x08) → relay_out
```

---

## 📁 Struktur Proyek

```
edgeai_seno/
├── run_pipeline.py           # 🚀 Master pipeline (jalankan ini)
├── generate_dataset.py       # 📊 Step 1: Generator dataset sintetis
├── train_model.py            # 🧠 Step 2: Pelatihan Neural Network
├── convert_to_tflite.py      # 📦 Step 3: Konversi & kuantisasi TFLite
├── generate_c_header.py      # 🔧 Step 4: Generator C-header (model.h)
├── inference_demo.py         # 🎮 Step 5: Demo inferensi real-time
├── firmware_example.c        # 💾 Contoh firmware ARM Cortex
├── requirements.txt          # 📋 Dependensi Python
├── README.md                 # 📖 Dokumentasi proyek
│
└── output/                   # 📂 Hasil output (auto-generated)
    ├── dummy_energy_data.csv
    ├── training_history.png
    ├── prediction_analysis.png
    ├── training_metrics.json
    ├── inference_demo_results.json
    └── model/
        ├── smart_diverter_model.keras
        ├── scaler.pkl
        ├── architecture_info.json
        ├── smart_diverter_float.tflite
        ├── smart_diverter.tflite      # ← File ini masuk ke SoC
        ├── tflite_conversion_info.json
        └── model.h                     # ← File ini untuk firmware
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependensi
```bash
pip install -r requirements.txt
```

### 2. Jalankan Full Pipeline
```bash
python run_pipeline.py
```

### 3. Atau Jalankan Per-Step
```bash
python run_pipeline.py --step 1    # Hanya generate dataset
python run_pipeline.py --step 2    # Hanya training
python run_pipeline.py --from-step 3  # Mulai dari step 3
```

### 4. Demo Inferensi Interaktif
```bash
python inference_demo.py --interactive
```

---

## 📊 Pipeline Steps

### Step 1: Generate Dataset Sintetis
- Membangkitkan 5000 sampel data skenario energi surya
- Fitur input: `jam`, `cuaca`, `daya_surya`, `toren_persen`, `suhu_heater`
- Label: `alihkan` (0 = Normal, 1 = Alihkan Daya)
- Logika: *Jika daya melimpah DAN penyimpanan belum penuh → Alihkan*

### Step 2: Latih Neural Network
- Arsitektur MLP: 5→8→4→1
- Optimizer: Adam (lr=0.001)
- Loss: Binary Cross-Entropy
- Early stopping + LR reduction
- Output: Model Keras, grafik evaluasi, metrik

### Step 3: Konversi TFLite (Kuantisasi)
- Float32 → INT8 quantization
- Representative dataset calibration
- Validasi akurasi pasca-kuantisasi
- Output: `.tflite` super ringan (beberapa KB)

### Step 4: Generate C-Header
- Konversi `.tflite` → C-array hexadecimal
- Include macro `PROB_TO_THRESHOLD`
- Define `THRESHOLD_REG_OFFSET 0x08`
- Siap di-compile bersama firmware

### Step 5: Demo Inferensi
- 8 skenario: Normal, Alihkan, dan Boundary
- Simulasi APB register write
- Output probabilitas → threshold → relay_out

---

## 🔌 Integrasi Hardware

### Mapping Probabilitas → Threshold
| Probabilitas AI | Threshold (Watt) | Aksi |
|:---:|:---:|:---|
| ≥ 0.90 | 1000W | Segera alihkan daya |
| ≥ 0.70 | 1500W | Alihkan moderat |
| ≥ 0.50 | 2000W | Alihkan ringan |
| < 0.50 | 3000W (max) | Normal, tidak alihkan |

### APB Bus Write ke smart_diverter_apb.v
```
PADDR  = 4'h8    (offset 0x08 → THRESHOLD_REG)
PWRITE = 1       (mode tulis)
PWDATA = [threshold_value]
```

### Kompilasi Firmware
```bash
# Contoh menggunakan ARM GCC
arm-none-eabi-gcc -o firmware.elf \
    firmware_example.c \
    -I./output/model \       # Lokasi model.h
    -I./tflite-micro/        # TFLite for Microcontrollers
    -mcpu=cortex-m4 \
    -mthumb
```

---

## 📈 Target Performa
- **Akurasi model:** > 90%
- **Ukuran TFLite:** < 10 KB (muat di flash mikrokontroler)
- **Latensi inferensi:** < 1 ms pada ARM Cortex-M4

---

*Dokumen ini adalah bagian dari Prototipe 2: Smart Load-Diverter Neural Network.*
