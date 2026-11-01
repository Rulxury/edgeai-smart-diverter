#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
  MODUL 4: GENERATOR C-BYTE ARRAY HEADER (model.h)
  Smart Load-Diverter Neural Network - Edge AI (TinyML)
===============================================================================
  Fungsi  : Jembatan pamungkas antara dunia Software dan Hardware.
            Mengkonversi file .tflite menjadi C-array hexadecimal yang bisa
            langsung di-compile bersama firmware mikrokontroler (ARM Cortex).

  Format Output:
    const unsigned char smart_diverter_tflite[] = {
        0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
    };
    const unsigned int smart_diverter_tflite_len = <size>;

  Alur Kerja Final di SoC:
    1. model.h di-compile bersama firmware utama
    2. TF Lite for Microcontrollers membaca array ini sebagai "Otak"
    3. CPU baca sensor → minta keputusan dari Otak
    4. CPU write ke THRESHOLD_REG (offset 0x08) via APB
    5. IP Core Verilog memicu relay_out
===============================================================================
"""

import os
import sys
import datetime

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
MODEL_DIR       = "output/model"
TFLITE_FILE     = os.path.join(MODEL_DIR, "smart_diverter.tflite")
OUTPUT_HEADER   = os.path.join(MODEL_DIR, "model.h")

# Nama variabel C
ARRAY_NAME      = "smart_diverter_tflite"
BYTES_PER_LINE  = 12    # Jumlah byte hex per baris untuk keterbacaan


def read_tflite_model(filepath: str) -> bytes:
    """Baca file TFLite biner."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File TFLite tidak ditemukan: '{filepath}'.\n"
            f"Jalankan 'convert_to_tflite.py' terlebih dahulu!"
        )

    with open(filepath, 'rb') as f:
        data = f.read()

    print(f"  📂 File TFLite dimuat: {filepath}")
    print(f"     Ukuran: {len(data)} bytes ({len(data)/1024:.2f} KB)")

    return data


def generate_c_header(tflite_data: bytes,
                      array_name: str = ARRAY_NAME,
                      bytes_per_line: int = BYTES_PER_LINE) -> str:
    """
    Konversi data biner TFLite ke format C-array header.

    Returns:
        String berisi konten file .h lengkap.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_size = len(tflite_data)

    # Header guards dan komentar
    lines = []
    lines.append("/*")
    lines.append(" * ═══════════════════════════════════════════════════════════════════")
    lines.append(" *   AUTO-GENERATED FILE — DO NOT EDIT MANUALLY")
    lines.append(" *   Smart Load-Diverter Neural Network — TFLite Model")
    lines.append(f" *   Generated: {timestamp}")
    lines.append(f" *   Model Size: {model_size} bytes ({model_size/1024:.2f} KB)")
    lines.append(" *")
    lines.append(" *   Arsitektur: Input(5) → Dense(8,ReLU) → Dense(4,ReLU) → Dense(1,Sigmoid)")
    lines.append(" *   Kuantisasi: INT8 (Full Integer Quantization)")
    lines.append(" *   Target: ARM Cortex-M (TensorFlow Lite for Microcontrollers)")
    lines.append(" *")
    lines.append(" *   Penggunaan dalam firmware:")
    lines.append(" *     #include \"model.h\"")
    lines.append(" *     const tflite::Model* model = tflite::GetModel(smart_diverter_tflite);")
    lines.append(" * ═══════════════════════════════════════════════════════════════════")
    lines.append(" */")
    lines.append("")
    lines.append(f"#ifndef SMART_DIVERTER_MODEL_H")
    lines.append(f"#define SMART_DIVERTER_MODEL_H")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")

    # Alignment untuk ARM
    lines.append("/* Model data — aligned untuk akses optimal pada ARM Cortex-M */")
    lines.append(f"alignas(16) const unsigned char {array_name}[] = {{")

    # Konversi setiap byte ke format hex
    for i in range(0, model_size, bytes_per_line):
        chunk = tflite_data[i:i + bytes_per_line]
        hex_values = [f"0x{b:02x}" for b in chunk]
        line = "    " + ", ".join(hex_values)

        # Tambahkan koma kecuali baris terakhir
        if i + bytes_per_line < model_size:
            line += ","
        lines.append(line)

    lines.append("};")
    lines.append("")

    # Ukuran model
    lines.append(f"const unsigned int {array_name}_len = {model_size};")
    lines.append("")

    # Info tambahan untuk firmware
    lines.append("/* ── Informasi Model untuk Firmware ── */")
    lines.append(f"#define MODEL_INPUT_SIZE      5      /* jam, cuaca, daya_surya, toren_persen, suhu_heater */")
    lines.append(f"#define MODEL_OUTPUT_SIZE     1      /* probabilitas pengalihan (0.0 - 1.0) */")
    lines.append(f"#define MODEL_TENSOR_ARENA    4096   /* Ukuran arena tensor (bytes) — sesuaikan jika perlu */")
    lines.append("")

    # Mapping threshold
    lines.append("/*")
    lines.append(" * ── Mapping Probabilitas ke Threshold Register ──")
    lines.append(" * Contoh konversi probabilitas AI → nilai THRESHOLD_REG:")
    lines.append(" *")
    lines.append(" *   Probabilitas  |  Threshold (Watt)  |  Aksi")
    lines.append(" *   ------------- | ------------------ | --------")
    lines.append(" *   >= 0.90       |  1000              |  Segera alihkan")
    lines.append(" *   >= 0.70       |  1500              |  Alihkan moderat")
    lines.append(" *   >= 0.50       |  2000              |  Alihkan ringan")
    lines.append(" *   <  0.50       |  3000 (max)        |  Normal/tidak alihkan")
    lines.append(" *")
    lines.append(" * Rumus: threshold = MAX_THRESHOLD - (prob * (MAX_THRESHOLD - MIN_THRESHOLD))")
    lines.append(" *        threshold = 3000 - (prob * 2000)")
    lines.append(" *")
    lines.append(" * Threshold ditulis ke THRESHOLD_REG pada offset 0x08")
    lines.append(" * melalui APB bus: PADDR=4'h8, PWRITE=1, PWDATA=[threshold]")
    lines.append(" */")
    lines.append(f"#define THRESHOLD_REG_OFFSET  0x08")
    lines.append(f"#define MAX_THRESHOLD_WATT    3000")
    lines.append(f"#define MIN_THRESHOLD_WATT    1000")
    lines.append("")
    lines.append("/* Macro konversi probabilitas → threshold */")
    lines.append(f"#define PROB_TO_THRESHOLD(prob) \\")
    lines.append(f"    (MAX_THRESHOLD_WATT - (unsigned int)((prob) * "
                 f"(MAX_THRESHOLD_WATT - MIN_THRESHOLD_WATT)))")
    lines.append("")
    lines.append(f"#endif /* SMART_DIVERTER_MODEL_H */")
    lines.append("")

    return "\n".join(lines)


def verify_header(header_content: str, original_data: bytes):
    """Verifikasi bahwa header yang dihasilkan cocok dengan data asli."""
    print(f"\n  🔍 Verifikasi integritas header...")

    # Ekstrak byte hex dari header
    import re
    hex_pattern = re.compile(r'0x([0-9a-f]{2})')
    hex_matches = hex_pattern.findall(header_content)
    extracted_bytes = bytes([int(h, 16) for h in hex_matches])

    # Bandingkan dengan data asli
    if extracted_bytes == original_data:
        print(f"  ✅ Verifikasi BERHASIL: {len(extracted_bytes)} bytes cocok sempurna")
        return True
    else:
        print(f"  ❌ Verifikasi GAGAL!")
        print(f"     Original : {len(original_data)} bytes")
        print(f"     Extracted: {len(extracted_bytes)} bytes")
        return False


def main():
    """Titik masuk utama generator C-header."""
    print("=" * 70)
    print("  🔧 GENERATOR C-BYTE ARRAY HEADER — Smart Load-Diverter")
    print("=" * 70)

    # 1. Baca file TFLite
    tflite_data = read_tflite_model(TFLITE_FILE)

    # 2. Generate C header
    print(f"\n  ⚙️ Mengkonversi {len(tflite_data)} bytes ke C-array...")
    header_content = generate_c_header(tflite_data)

    # 3. Simpan header
    with open(OUTPUT_HEADER, 'w', encoding='utf-8') as f:
        f.write(header_content)
    print(f"  ✅ Header C disimpan ke: {OUTPUT_HEADER}")

    # 4. Verifikasi
    verify_header(header_content, tflite_data)

    # 5. Statistik
    header_size = os.path.getsize(OUTPUT_HEADER)
    print(f"\n  📏 Statistik:")
    print(f"     TFLite binary  : {len(tflite_data):>8d} bytes ({len(tflite_data)/1024:.2f} KB)")
    print(f"     C header file  : {header_size:>8d} bytes ({header_size/1024:.2f} KB)")
    print(f"     Jumlah baris   : {header_content.count(chr(10)):>8d} baris")

    # 6. Preview
    print(f"\n  👀 Preview model.h (20 baris pertama + 10 baris terakhir):")
    print(f"  {'─' * 50}")
    h_lines = header_content.split('\n')
    for line in h_lines[:20]:
        print(f"  {line}")
    print(f"  ... ({len(h_lines) - 30} baris lainnya) ...")
    for line in h_lines[-10:]:
        print(f"  {line}")

    print(f"\n{'=' * 70}")
    print(f"  ✅ C-HEADER SIAP!")
    print(f"  File '{OUTPUT_HEADER}' siap di-compile bersama firmware ARM Cortex-M")
    print(f"  Gunakan: #include \"model.h\"")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
