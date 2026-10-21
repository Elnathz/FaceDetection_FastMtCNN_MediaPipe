# Rule: Sinkronisasi Otomatis README.md

Setiap kali terjadi perubahan pada:
1. Total jumlah dataset citra primer (misalnya penambahan kelompok baru).
2. Total citra augmentasi dan jumlah siklus inferensi benchmarking.
3. Nilai metrik evaluasi kuantitatif (Precision, Recall, F1-Score, IoU, Latency, FPS).
4. Penambahan atau perubahan metode detektor / skenario augmentasi.

Anda WAJIB memperbarui file `README.md` agar seluruh tabel metrik, matriks breakdown skenario, dan deskripsi metadata tetap 100% akurat dan sinkron dengan artefak data terbaru di folder `results/`.
Format penulisan README.md harus selalu bebas dari elemen AI slop: tidak menggunakan emoji, tidak menggunakan em-dash atau en-dash, dan menggunakan gaya bahasa technical benchmark report yang objektif dan padat.
