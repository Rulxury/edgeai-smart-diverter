#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  MODUL 1: GENERATOR DATASET SINTETIS
  Smart Load-Diverter Neural Network - Edge AI (TinyML)
===============================================================================
  Fungsi  : Membangkitkan data sintetis ('dummy_energy_data.csv') sebagai
            "buku pelajaran" bagi model Neural Network untuk mengenali pola
            kapan energi surya harus disimpan ke toren/pemanas agar tidak
            bocor ke jaringan PLN (mencegah Grid-Defection curtailment).

  Fitur Input (X):
    - jam          : 6.00 - 18.00  (ketersediaan matahari)
    - cuaca        : 0=Hujan, 1=Mendung, 2=Cerah
    - daya_surya   : Watt (produksi listrik aktual)
    - toren_persen : 0-100% (ruang penyimpanan air)
    - suhu_heater  : °C (status panas water heater)

  Label Output (Y):
    - alihkan      : 1 (Alihkan Daya) atau 0 (Normal)
    Logika: Jika daya surya melimpah DAN penyimpanan belum penuh → 1
===============================================================================
"""

import numpy as np
import pandas as pd
import os

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
NUM_SAMPLES       = 5000          # Jumlah baris data yang di-generate
RANDOM_SEED       = 42            # Untuk reproduktifitas
OUTPUT_DIR        = "output"      # Direktori output
OUTPUT_FILENAME   = "dummy_energy_data.csv"

# Threshold logika pengalihan daya
DAYA_SURYA_MIN_ALIH  = 500   # Watt minimal agar dianggap "melimpah"
TOREN_MAX_PENUH      = 85    # Persen — di atas ini toren dianggap penuh
SUHU_MAX_PENUH       = 70    # °C — di atas ini heater dianggap sudah cukup panas

np.random.seed(RANDOM_SEED)


def generate_jam(n: int) -> np.ndarray:
    """Bangkitkan jam operasi panel surya (6.00 - 18.00)."""
    return np.round(np.random.uniform(6.0, 18.0, size=n), 2)


def generate_cuaca(n: int) -> np.ndarray:
    """Bangkitkan kondisi cuaca: 0=Hujan, 1=Mendung, 2=Cerah."""
    return np.random.choice([0, 1, 2], size=n, p=[0.2, 0.3, 0.5])


def generate_daya_surya(jam: np.ndarray, cuaca: np.ndarray) -> np.ndarray:
    """
    Bangkitkan daya surya (Watt) berdasarkan jam dan cuaca.
    - Puncak produksi di sekitar jam 12 siang.
    - Cuaca cerah → daya tinggi, hujan → daya rendah.
    """
    # Faktor jam: gaussian peak di jam 12
    faktor_jam = np.exp(-0.5 * ((jam - 12.0) / 2.5) ** 2)

    # Faktor cuaca
    faktor_cuaca = np.where(cuaca == 2, 1.0,
                   np.where(cuaca == 1, 0.5, 0.15))

    # Daya dasar maksimum 2000W
    daya_base = 2000.0 * faktor_jam * faktor_cuaca

    # Tambahkan noise realistis (±10%)
    noise = np.random.normal(1.0, 0.1, size=len(jam))
    daya = np.clip(daya_base * noise, 0, 2500)

    return np.round(daya, 1)


def generate_toren_persen(n: int) -> np.ndarray:
    """Bangkitkan status toren air (0-100%)."""
    return np.round(np.random.uniform(0, 100, size=n), 1)


def generate_suhu_heater(n: int) -> np.ndarray:
    """Bangkitkan suhu water heater (25°C - 90°C)."""
    return np.round(np.random.uniform(25, 90, size=n), 1)


def compute_label(daya_surya: np.ndarray,
                  toren_persen: np.ndarray,
                  suhu_heater: np.ndarray) -> np.ndarray:
    """
    Tentukan label pengalihan daya berdasarkan logika:
      Jika daya surya melimpah (>500W) DAN
         (toren belum penuh (<85%) ATAU heater belum panas (<70°C))
      → Alihkan (1), selain itu Normal (0).

    Ditambahkan sedikit noise label (~3%) untuk realisme.
    """
    kondisi_daya   = daya_surya >= DAYA_SURYA_MIN_ALIH
    kondisi_toren  = toren_persen < TOREN_MAX_PENUH
    kondisi_heater = suhu_heater < SUHU_MAX_PENUH

    # Daya melimpah DAN salah satu storage belum penuh
    label = (kondisi_daya & (kondisi_toren | kondisi_heater)).astype(int)

    # Noise ~3%: flip beberapa label untuk realisme
    noise_mask = np.random.random(len(label)) < 0.03
    label[noise_mask] = 1 - label[noise_mask]

    return label


def generate_dataset() -> pd.DataFrame:
    """Pipeline utama: bangkitkan seluruh kolom dan gabungkan."""
    print("=" * 70)
    print("  📊 GENERATOR DATASET SINTETIS — Smart Load-Diverter")
    print("=" * 70)

    n = NUM_SAMPLES

    # Generate setiap fitur
    jam          = generate_jam(n)
    cuaca        = generate_cuaca(n)
    daya_surya   = generate_daya_surya(jam, cuaca)
    toren_persen = generate_toren_persen(n)
    suhu_heater  = generate_suhu_heater(n)

    # Hitung label
    alihkan = compute_label(daya_surya, toren_persen, suhu_heater)

    # Gabungkan ke DataFrame
    df = pd.DataFrame({
        "jam"          : jam,
        "cuaca"        : cuaca,
        "daya_surya"   : daya_surya,
        "toren_persen" : toren_persen,
        "suhu_heater"  : suhu_heater,
        "alihkan"      : alihkan
    })

    return df


def print_statistics(df: pd.DataFrame):
    """Cetak statistik ringkas dari dataset."""
    print(f"\n  ✅ Dataset berhasil dibangkitkan: {len(df)} sampel")
    print(f"  {'─' * 50}")
    print(f"  📈 Distribusi Label:")
    counts = df["alihkan"].value_counts()
    total  = len(df)
    for label_val in sorted(counts.index):
        count = counts[label_val]
        pct   = count / total * 100
        desc  = "Alihkan Daya" if label_val == 1 else "Normal"
        print(f"     Label {label_val} ({desc:>13s}): {count:>5d} sampel ({pct:.1f}%)")

    print(f"\n  📊 Statistik Fitur:")
    print(f"  {'─' * 50}")
    for col in ["jam", "cuaca", "daya_surya", "toren_persen", "suhu_heater"]:
        print(f"     {col:<14s}: min={df[col].min():>8.1f}  "
              f"max={df[col].max():>8.1f}  "
              f"mean={df[col].mean():>8.1f}")


def main():
    """Titik masuk utama."""
    df = generate_dataset()

    # Buat direktori output jika belum ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Simpan CSV
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    df.to_csv(output_path, index=False)
    print(f"\n  💾 Dataset disimpan ke: {output_path}")

    # Tampilkan statistik
    print_statistics(df)

    # Preview 10 baris pertama
    print(f"\n  👀 Preview (10 baris pertama):")
    print(f"  {'─' * 50}")
    print(df.head(10).to_string(index=False))
    print(f"\n{'=' * 70}")

    return df


if __name__ == "__main__":
    main()
