# Panduan Lengkap dan Bedah Teknis Eksperimen Deteksi Wajah
## Komparasi Kinerja Fast MTCNN vs MediaPipe pada Dataset DPO Bareskrim

Dokumen ini disusun sebagai panduan teknis mendalam untuk memahami alur kerja, logika algoritma, istilah teknis pengolahan citra digital (PCD) dan computer vision (CV), serta interpretasi data hasil eksekusi dari notebook `face_detection_comparison.ipynb`.

---

## DAFTAR ISI
1. [Glosarium Istilah Teknis untuk Pemula](#1-glosarium-istilah-teknis-untuk-pemula)
   - [Ground Truth (GT)](#11-ground-truth-gt)
   - [Bounding Box (BBox)](#12-bounding-box-bbox)
   - [Parser dan Parsing](#13-parser-dan-parsing)
   - [Intersection over Union (IoU)](#14-intersection-over-union-iou)
   - [Confusion Matrix Deteksi Objek (TP, FP, FN)](#15-confusion-matrix-deteksi-objek-tp-fp-fn)
   - [Precision, Recall, dan F1-Score](#16-precision-recall-dan-f1-score)
   - [Latency dan Throughput (FPS)](#17-latency-dan-throughput-fps)
   - [Bounding Box-Preserving Augmentation](#18-bounding-box-preserving-augmentation)
   - [Cascade CNN vs Single-Shot Detector](#19-cascade-cnn-vs-single-shot-detector)
   - [Micro-Averaging](#110-micro-averaging)
2. [Bedah Rinci Tiap Cell Notebook](#2-bedah-rinci-tiap-cell-notebook)
   - [Cell 0: Judul, Abstrak & Roadmap](#cell-0-markdown-judul-abstrak--roadmap)
   - [Cell 1: Setup Environment (Deskripsi)](#cell-1-markdown-setup-environment-deskripsi)
   - [Cell 2: Instalasi Dependensi Pustaka](#cell-2-code--cell-1a-instalasi-dependensi-pustaka)
   - [Cell 3: Import, Hardware Profiling & Konfigurasi Global](#cell-3-code--cell-1b-import-hardware-profiling--konfigurasi-global)
   - [Cell 4: Dokumentasi Strategi Ground Truth](#cell-4-markdown-dokumentasi-strategi-ground-truth)
   - [Cell 5: Auto-Discovery Citra Asli Dataset](#cell-5-code--cell-2a-auto-discovery-citra-asli-dataset)
   - [Cell 6: Standardisasi Format dan Penamaan Citra](#cell-6-code--cell-2b-standardisasi-format-dan-penamaan-citra)
   - [Cell 7: Pembuatan Database Ground Truth Incremental](#cell-7-code--cell-2c-pembuatan-database-ground-truth-incremental)
   - [Cell 8: Pengantar Exploratory Data Analysis (EDA) Asli](#cell-8-markdown-pengantar-exploratory-data-analysis-eda-asli)
   - [Cell 9: Ekstraksi Statistik Citra Asli](#cell-9-code--cell-3a-ekstraksi-statistik-citra-asli)
   - [Cell 10: Visualisasi Grafik Statistik EDA](#cell-10-code--cell-3b-visualisasi-grafik-statistik-eda)
   - [Cell 11: Grid Overlay Bounding Box GT pada Citra Asli](#cell-11-code--cell-3c-grid-overlay-bounding-box-gt-pada-citra-asli)
   - [Cell 12: Dasar Teori 10 Skenario Augmentasi Data](#cell-12-markdown-dasar-teori-10-skenario-augmentasi-data)
   - [Cell 13: Definisi Pipeline Transformasi Albumentations](#cell-13-code--cell-4a-definisi-pipeline-transformasi-albumentations)
   - [Cell 14: Eksekusi Batch Augmentasi 120 Citra](#cell-14-code--cell-4b-eksekusi-batch-augmentasi-120-citra)
   - [Cell 15: Pengantar EDA Pasca-Augmentasi](#cell-15-markdown-pengantar-eda-pasca-augmentasi)
   - [Cell 16: Verifikasi Integritas Bounding Box](#cell-16-code--cell-5a-verifikasi-integritas-bounding-box)
   - [Cell 17: Analisis Distribusi Kecerahan dan Kontras](#cell-17-code--cell-5b-analisis-distribusi-kecerahan-dan-kontras)
   - [Cell 18: Visualisasi Grid Sample 10 Skenario](#cell-18-code--cell-5c-visualisasi-grid-sample-10-skenario)
   - [Cell 19: Teori Wrapper & Unified Interface](#cell-19-markdown-teori-wrapper--unified-interface)
   - [Cell 20: Dataclass & Wrapper Fast MTCNN](#cell-20-code--cell-6a-dataclass--wrapper-fast-mtcnn)
   - [Cell 21: Wrapper MediaPipe & Sanity Check](#cell-21-code--cell-6b-wrapper-mediapipe--sanity-check)
   - [Cell 22: Formulasi Matematika Metrik Evaluasi](#cell-22-markdown-formulasi-matematika-metrik-evaluasi)
   - [Cell 23: Algoritma IoU dan Greedy Matching](#cell-23-code--cell-7a-algoritma-iou-dan-greedy-matching)
   - [Cell 24: Engine Benchmarking Utama (240 Pengujian)](#cell-24-code--cell-7b-engine-benchmarking-utama-240-pengujian)
   - [Cell 25: Pengantar Agregasi Micro-Averaging](#cell-25-markdown-pengantar-agregasi-micro-averaging)
   - [Cell 26: Kalkulasi Metrik Global & Tabel Komparasi](#cell-26-code--cell-8a-kalkulasi-metrik-global--tabel-komparasi)
   - [Cell 27: Breakdown Performa per Skenario & Heatmap](#cell-27-code--cell-8b-breakdown-performa-per-skenario--heatmap)
   - [Cell 28: Pengantar Analisis Visual & Failure Case](#cell-28-markdown-pengantar-analisis-visual--failure-case)
   - [Cell 29: Bar Chart Komparasi 4 Metrik Kunci](#cell-29-code--cell-9a-bar-chart-komparasi-4-metrik-kunci)
   - [Cell 30: Visualisasi Side-by-Side 3 Panel](#cell-30-code--cell-9b-visualisasi-side-by-side-3-panel)
   - [Cell 31: Radar Chart Komparasi Multi-Dimensi](#cell-31-code--cell-9c-radar-chart-komparasi-multi-dimensi)
   - [Cell 32: Isolasi Kasus Kegagalan (Failure Cases)](#cell-32-code--cell-9d-isolasi-kasus-kegagalan-failure-cases)
   - [Cell 33: Dokumentasi Ekspor Hasil](#cell-33-markdown-dokumentasi-ekspor-hasil)
   - [Cell 34: Ekspor Berkas CSV, JSON & Print Ringkasan](#cell-34-code--cell-10a-ekspor-berkas-csv-json--print-ringkasan)
   - [Cell 35: Verifikasi Inventori Artefak Output](#cell-35-code--cell-10b-verifikasi-inventori-artefak-output)
   - [Cell 36: Daftar Pustaka Akademik](#cell-36-markdown-daftar-pustaka-akademik)
3. [Evaluasi Kritis & Blind Spot Metodologis](#3-evaluasi-kritis--blind-spot-metodologis)

---

## 1. Glosarium Istilah Teknis untuk Pemula

### 1.1. Ground Truth (GT)
* **Apa artinya?**  
  Ground Truth adalah **kunci jawaban mutlak** atau data fakta sebenarnya di lapangan yang dianggap 100% benar. Dalam konteks deteksi wajah, Ground Truth adalah koordinat kotak yang menandai di mana letak wajah manusia sebenarnya pada sebuah foto.
* **Analogi sederhana**:  
  Jika model AI adalah siswa yang sedang mengerjakan ujian dengan melingkari gambar wajah, Ground Truth adalah lembar kunci jawaban milik guru. Kita membandingkan lingkaran siswa dengan lingkaran kunci jawaban guru untuk menentukan apakah jawabannya benar atau salah.
* **Dari mana asalnya?**  
  Idealnya, Ground Truth dibuat oleh **manusia (anotator manual)** menggunakan software pelabelan seperti *LabelImg* atau *CVAT*. Namun di notebook ini (Cell 7), karena dataset asli tidak memiliki file anotasi formal, Ground Truth diestimasi otomatis menggunakan algoritma OpenCV Haar Cascade sebagai *proxy* (pengganti sementara).

### 1.2. Bounding Box (BBox)
* **Apa artinya?**  
  Bounding Box adalah **persegi panjang pembatas** yang membungkus objek target (dalam hal ini wajah). Bounding box didefinisikan oleh 4 nilai angka spasial.
* **Format Koordinat yang Umum**:
  1. **Pascal VOC format (`[xmin, ymin, xmax, ymax]`)**:  
     Menyimpan titik sudut kiri-atas `(xmin, ymin)` dan titik sudut kanan-bawah `(xmax, ymax)` dalam satuan piksel absolut. Ini format yang dipakai di notebook ini.
  2. **COCO format (`[xmin, ymin, width, height]`)**:  
     Menyimpan titik kiri-atas ditambah lebar (*width*) dan tinggi (*height*) kotak.
  3. **YOLO format (`[x_center, y_center, width, height]`)**:  
     Koordinat titik tengah kotak beserta lebar dan tingginya, biasanya dinormalisasi ke rentang `[0, 1]`.
  4. **Koordinat Relatif / Normalisasi (MediaPipe)**:  
     Angka desimal dari 0.0 sampai 1.0 relatif terhadap dimensi gambar. Untuk mengubahnya ke piksel absolut:
     $$\text{xmin\_piksel} = \text{xmin\_relatif} \times \text{lebar\_gambar}$$
     $$\text{ymin\_piksel} = \text{ymin\_relatif} \times \text{tinggi\_gambar}$$

### 1.3. Parser dan Parsing
* **Apa artinya?**  
  *Parsing* adalah proses **membaca, mengurai, dan menerjemahkan** teks mentah (raw text) atau format data terstruktur (seperti JSON, XML, string CSV) ke dalam tipe data yang dapat dimanipulasi oleh logika program (seperti Dictionary, List, Object di Python), atau sebaliknya. Program atau fungsi yang bertugas melakukan ini disebut *Parser*.
* **Contoh konkret di kode notebook**:
  * `json.loads(string_data)`: Bertindak sebagai parser yang membaca teks string JSON `"[10, 20, 100, 150]"` dan mengubahnya menjadi objek `list` Python `[10, 20, 100, 150]`.
  * `json.dumps(objek_python)`: Melakukan kebalikannya (*serialization*), mengubah array Python menjadi teks string agar bisa disimpan di kolom tabel CSV.
  * Parser jalur file (`Path.suffix`): Membaca ekstensi nama file untuk memisahkan apakah suatu file berupa `.jpg` atau bukan.

### 1.4. Intersection over Union (IoU)
* **Apa artinya?**  
  IoU (disebut juga Jaccard Index) adalah angka pecahan desimal dari `0.0` sampai `1.0` yang mengukur **seberapa akurat dan tumpang tindih** kotak tebakan model dibandingkan kotak kunci jawaban (Ground Truth).
* **Rumus**:
  $$\text{IoU} = \frac{\text{Luas Irisan (Intersection)}}{\text{Luas Gabungan (Union)}}$$
  * Luas Irisan: Luas area di mana kotak tebakan dan kotak GT saling menumpuk.
  * Luas Gabungan: Total luas area yang dicakup oleh kedua kotak jika disatukan (tanpa menghitung area tumpang tindih dua kali).
* **Analogi**:  
  Bayangkan Anda menjiplak gambar koin dengan meletakkan koin asli di atas gambar Anda. Jika koin tepat berada di atas gambar, nilai IoU = 1.0 (100% overlap). Jika hanya pinggirannya yang bersenggolan sedikit, IoU mendekati 0.
* **Ambang Batas (IoU Threshold = 0.5)**:  
  Standar internasional PASCAL VOC menetapkan: jika IoU $\ge 0.5$, maka deteksi dianggap **sukses / pas sasaran**. Jika IoU $< 0.5$, dianggap meleset.

### 1.5. Confusion Matrix Deteksi Objek (TP, FP, FN)
Dalam klasifikasi biasa, confusion matrix dihitung per baris data. Namun dalam deteksi objek berbasis gambar, hitungannya berbasis bounding box:
1. **True Positive (TP)**:  
   Model menebak ada wajah, dan tebakan tersebut mengenai wajah asli (IoU $\ge 0.5$). Ini adalah **tebakan sukses**.
2. **False Positive (FP)**:  
   Model menebak ada wajah, tetapi di lokasi tersebut tidak ada wajah asli (halusinasi detektor), ATAU kotak tebakannya melenceng terlalu jauh (IoU $< 0.5$), ATAU model menebak 2 kotak di 1 wajah yang sama (kotak kedua dihitung sebagai FP / deteksi dobel). Ini adalah **alarm palsu**.
3. **False Negative (FN)**:  
   Ada wajah asli (Ground Truth), tetapi model **sama sekali tidak mendeteksinya** atau melewatkannya. Ini adalah **kegagalan deteksi (lolos)**.
4. *Catatan*: Dalam deteksi objek, tidak ada nilai True Negative (TN) karena jumlah area bukan wajah di gambar tidak terhingga.

### 1.6. Precision, Recall, dan F1-Score
* **Precision (Ketepatan)**:
  $$\text{Precision} = \frac{TP}{TP + FP}$$
  * Pertanyaan yang dijawab: *"Dari semua kotak yang dituduh model sebagai wajah, berapa persen yang benar-benar wajah asli?"*
  * Precision tinggi = model sangat berhati-hati, jarang membunyikan alarm palsu.
* **Recall (Sensitivitas / Kelengkapan)**:
  $$\text{Recall} = \frac{TP}{TP + FN}$$
  * Pertanyaan yang dijawab: *"Dari seluruh wajah yang ada di foto, berapa persen yang berhasil ditemukan oleh model?"*
  * Recall tinggi = model sangat jeli, hampir tidak ada wajah buronan yang lolos.
* **F1-Score**:
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  * Rata-rata harmonik antara Precision dan Recall. Dipakai untuk menilai keseimbangan performa model secara menyeluruh.

### 1.7. Latency dan Throughput (FPS)
* **Latency (Latensi)**:  
  Waktu komputasi yang dihabiskan prosesor untuk memproses 1 citra, diukur dalam milidetik (ms). Semakin kecil angkanya, semakin cepat model bekerja.
* **Throughput (Frames Per Second / FPS)**:  
  Jumlah frame citra yang sanggup diselesaikan model dalam 1 detik.
  $$\text{FPS} = \frac{1000}{\text{Mean Latency (ms)}}$$
  * Kecepatan video standar adalah 24–30 FPS. Jika FPS $\ge 30$, model sudah memenuhi syarat operasional *real-time*.

### 1.8. Bounding Box-Preserving Augmentation
* **Apa artinya?**  
  Augmentasi citra adalah teknik memanipulasi gambar (seperti memutar, menggelapkan, memberi noise) untuk menguji ketangguhan AI.  
  *Bounding Box-Preserving* artinya koordinat kotak penanda wajah ikut **disinkronkan dan dihitung ulang secara otomatis** mengikuti perubahan fisik gambar.
* **Contoh**:  
  Jika foto dibalik horizontal (kanan jadi kiri), koordinat wajah yang tadinya ada di `xmin=100` harus dipindahkan secara matematis ke posisi cerminnya di sisi kanan. Jika pustaka augmentasi tidak memiliki fitur ini, koordinat wajah akan tetap di kiri sementara gambarnya sudah pindah ke kanan (Ground Truth rusak). Notebook ini menggunakan library `albumentations` yang secara otomatis menjaga sinkronisasi ini.

### 1.9. Cascade CNN vs Single-Shot Detector
* **Fast MTCNN (Cascade CNN)**:  
  Bekerja secara bertahap melalui 3 jaringan saraf konvolusional (CNN) berurutan:
  1. *P-Net (Proposal Network)*: Menjelajah gambar untuk mengusulkan kandidat area yang mirip wajah secara kasar.
  2. *R-Net (Refine Network)*: Menyaring kandidat, membuang kotak-kotak yang jelas-jelas bukan wajah.
  3. *O-Net (Output Network)*: Memberikan keputusan final, menyesuaikan posisi bounding box, dan menentukan posisi landmark wajah.
  * *Kelebihan*: Sangat teliti dan tahan gangguan noise karena disaring bertahap.
  * *Kekurangan*: Beban komputasi tinggi (lambat).
* **MediaPipe Face Detection (BlazeFace - Single Shot)**:  
  Arsitektur saraf yang didesain Google khusus untuk komputasi kilat di perangkat mobile/GPU. Menggunakan konsep *Single-Shot Detector*: gambar hanya dilewatkan satu kali melalui jaringan untuk langsung memprediksi lokasi dan ukuran semua wajah.
  * *Kelebihan*: Sangat ringan, sub-millisecond, FPS tembus ratusan.
  * *Kekurangan*: Rentan gagal jika struktur gambar mengalami noise acak frekuensi tinggi (*salt-and-pepper / gaussian noise*).

### 1.10. Micro-Averaging
Menghitung total seluruh $TP$, $FP$, dan $FN$ dari seluruh 120 gambar secara global, lalu baru menghitung rumus Precision dan Recall. Metode ini mencegah bias yang timbul jika ada gambar yang tidak memiliki wajah atau jumlah deteksinya timpang.

---

## 2. Bedah Rinci Tiap Cell Notebook

### Cell 0 [Markdown]: Judul, Abstrak & Roadmap
* **Tujuan**: Menyediakan identitas akademik riset, ringkasan masalah (kebutuhan identifikasi DPO Bareskrim di bawah kondisi degradasi visual), dan tabel struktur notebook dari Cell 1 hingga Cell 10.
* **Hasil**: Teks dokumentasi awal laporan.

### Cell 1 [Markdown]: Setup Environment (Deskripsi)
* **Tujuan**: Memberikan petunjuk kompatibilitas dependensi (MediaPipe, PyTorch, Albumentations) dan penetapan parameter seed acak `SEED = 42`.
* **Hasil**: Teks pengantar instalasi.

### Cell 2 [Code / Cell 1A]: Instalasi Dependensi Pustaka
* **Cara Kerja**:  
  Mendefinisikan daftar 12 library Python, lalu mengeksekusinya menggunakan modul `subprocess` melalui perintah `pip install <package> -q`.
* **Istilah Teknis**:  
  *Subprocess*: Menjalankan perintah terminal/command prompt dari dalam script Python.
* **Hasil Nyata**:  
  Seluruh dependensi terverifikasi berhasil terpasang dengan status `✅`.

### Cell 3 [Code / Cell 1B]: Import, Hardware Profiling & Konfigurasi Global
* **Cara Kerja**:  
  1. Melakukan import library inti (`cv2`, `torch`, `albumentations`, `pandas`, `seaborn`, dll.).
  2. Mengunci random seed: `random.seed(42)`, `np.random.seed(42)`.
  3. Menjalankan fungsi `get_hardware_info()`: memanggil PowerShell `Get-CimInstance Win32_Processor` dan pustaka `psutil` untuk membaca data fisik komputer.
  4. Mendefinisikan struktur path folder menggunakan `pathlib.Path`: `dataset/original`, `dataset/augmented`, `results`, `results/plots`.
* **Istilah Teknis**:  
  *Reproducibility*: Jaminan bahwa eksperimen jika diulang oleh peneliti lain di komputer berbeda dengan seed yang sama akan menghasilkan angka yang identik.
* **Hasil Nyata**:  
  Sistem mendeteksi CPU: `Intel Core Ultra 7 255HX (20 Cores/20 Threads)`, RAM: `31.43 GB`, OS: `Windows 11 (64bit)`. Seluruh folder kerja otomatis terbuat.

### Cell 4 [Markdown]: Dokumentasi Strategi Ground Truth
* **Tujuan**: Menjelaskan struktur file `ground_truth.json` dan memaparkan alasan digunakannya Haar Cascade sebagai jembatan estimasi awal ketiadaan anotasi bawaan.
* **Hasil**: Teks petunjuk format JSON data.

### Cell 5 [Code / Cell 2A]: Auto-Discovery Citra Asli Dataset
* **Cara Kerja**:  
  Fungsi `collect_images()` menyusuri direktori `Dataset PCD 4703 - DPO` secara rekursif (`rglob('*')`), memfilter hanya file dengan ekstensi citra valid (`.jpg`, `.jpeg`, `.png`), dan mengurutkannya berdasarkan folder kelompok mahasiswa.
* **Istilah Teknis**:  
  *Auto-Discovery*: Pencarian file otomatis tanpa perlu menuliskan nama file satu per satu di dalam kode (hardcoding).
* **Hasil Nyata**:  
  Ditemukan tepat **12 citra asli** dari subfolder Kelompok 1, Kelompok 2, dan Kelompok 3.

### Cell 6 [Code / Cell 2B]: Standardisasi Format dan Penamaan Citra
* **Cara Kerja**:  
  Membaca tiap citra yang ditemukan via OpenCV (`cv2.imread`), kemudian menyalin dan menulis ulang citra tersebut ke folder `dataset/original` dengan format nama seragam `dpo_01.jpg` sampai `dpo_12.jpg` menggunakan kualitas kompresi JPEG 95.
* **Istilah Teknis**:  
  *Data Standardization*: Penyeragaman format, penamaan, dan ekstensi file untuk mempermudah proses iterasi dan otomasi loop program.
* **Hasil Nyata**:  
  12 file citra tersimpan rapi dan seragam di direktori `dataset/original`.

### Cell 7 [Code / Cell 2C]: Pembuatan Database Ground Truth Incremental
* **Cara Kerja**:  
  1. Memeriksa file `ground_truth.json`.
  2. Jika ada citra yang belum terdaftar, fungsi `estimate_gt_with_haar()` membaca citra, mengubah ke grayscale, menerapkan perataan histogram (`cv2.equalizeHist`), lalu menjalankan `detectMultiScale` Haar Cascade bawaan OpenCV.
  3. Mengambil kotak wajah terbesar dan menyimpannya sebagai Ground Truth.
  4. Jika Haar gagal menemukan wajah, fungsi membuat kotak fallback darurat di tengah citra.
* **Istilah Teknis**:  
  *Proxy Ground Truth*: Menggunakan algoritma lain untuk menebak kunci jawaban sementara ketika belum ada label buatan manusia.  
  *Histogram Equalization*: Teknik pemerataan distribusi intensitas piksel untuk meningkatkan kontras gambar grayscale.
* **Hasil Nyata**:  
  File `ground_truth.json` terbentuk sempurna berisi 12 entri gambar, masing-masing memiliki resolusi dan 1 bounding box wajah terdaftar.

### Cell 8 [Markdown]: Pengantar Exploratory Data Analysis (EDA) Asli
* **Tujuan**: Menguraikan empat fokus analisis eksplorasi terhadap 12 foto DPO asli.
* **Hasil**: Dokumentasi tujuan EDA.

### Cell 9 [Code / Cell 3A]: Ekstraksi Statistik Citra Asli
* **Cara Kerja**:  
  Melakukan perulangan pada 12 gambar:
  1. Menghitung kecerahan: nilai rata-rata (*mean*) piksel grayscale.
  2. Menghitung kontras: standar deviasi (*std*) piksel grayscale.
  3. Menghitung luas wajah relatif: `(luas bounding box) / (lebar citra * tinggi citra)`.
  4. Menghitung mean intensitas masing-masing kanal warna Red, Green, Blue.
  5. Memasukkan seluruh data ke DataFrame Pandas `df_eda`.
* **Istilah Teknis**:  
  *Descriptive Statistics*: Pengukuran dasar statistik (mean, median, standar deviasi, min, max) untuk memahami sebaran data mentah.
* **Hasil Nyata**:  
  Dataset memiliki variasi resolusi lebar 348–996 px, tinggi 591–1600 px. Kecerahan berkisar antara 66.69 (gelap) hingga 166.69 (terang). Area wajah mencakup rata-rata 33% dari luas keseluruhan foto.

### Cell 10 [Code / Cell 3B]: Visualisasi Grafik Statistik EDA
* **Cara Kerja**:  
  Membuat figure Matplotlib 2 baris x 3 kolom yang merangkum hasil `df_eda`:
  * (a) Scatter plot resolusi spasial (lebar vs tinggi).
  * (b) Bar plot rasio aspek per citra.
  * (c) Scatter plot kecerahan vs kontras bergradasi warna rasio area wajah.
  * (d) Bar plot mean intensitas warna R, G, B dengan error bar.
  * (e) Bar plot persentase luas wajah terhadap frame.
  * (f) Bar plot jumlah wajah per citra.
* **Hasil Nyata**:  
  Grafik statistik tersimpan di `results/plots/eda_statistics.png`. Terlihat bahwa dataset sangat heterogen dalam dimensi dan kondisi pencahayaan awal.

### Cell 11 [Code / Cell 3C]: Grid Overlay Bounding Box GT pada Citra Asli
* **Cara Kerja**:  
  Menyusun grid 4 kolom x 3 baris. Setiap citra asli digambari kotak hijau Ground Truth menggunakan fungsi `cv2.rectangle` dan diberi label teks ID gambar serta dimensinya.
* **Hasil Nyata**:  
  Montase gambar tersimpan di `results/plots/grid_original_gt.png`, membuktikan bahwa 12 citra primer memiliki wajah tunggal bertipe portrait/pasfoto.

### Cell 12 [Markdown]: Dasar Teori 10 Skenario Augmentasi Data
* **Tujuan**: Menjelaskan justifikasi ilmiah pemilihan 10 jenis augmentasi berdasarkan referensi literatur jurnal pemrosesan citra digital surveillance dan CCTV.
* **Hasil**: Tabel perbandingan 10 skenario dan sitasi paper rujukan.

### Cell 13 [Code / Cell 4A]: Definisi Pipeline Transformasi Albumentations
* **Cara Kerja**:  
  Mengonfigurasi objek `BboxParams(format='pascal_voc', min_visibility=0.3, min_area=100)`. Kemudian membuat list dictionary berisi 10 skenario transformasi menggunakan fungsi-fungsi `albumentations` (seperti `GaussianBlur`, `MotionBlur`, `GaussNoise`, `ImageCompression`, `Rotate`, `RandomBrightnessContrast`).
* **Istilah Teknis**:  
  *min_visibility*: Ambang batas minimum persentase bagian kotak wajah yang harus tetap terlihat setelah transformasi (misal setelah diputar) agar tidak dihapus.
* **Hasil Nyata**:  
  10 pipeline transformasi terdaftar dan tervalidasi di memori program.

### Cell 14 [Code / Cell 4B]: Eksekusi Batch Augmentasi 120 Citra
* **Cara Kerja**:  
  Melakukan nested loop: 12 citra asli $\times$ 10 skenario = 120 kombinasi citra.
  1. Citra dan bounding box dimasukkan ke pipeline Albumentations.
  2. Bounding box hasil transformasi divalidasi oleh `validate_bbox()` (memastikan koordinat tidak minus, tidak melampaui lebar/tinggi gambar, dan luas $\ge 100\text{ px}^2$).
  3. Citra augmentasi disimpan fisik ke folder `dataset/augmented/`.
  4. Anotasi baru disimpan ke `results/augmented_annotations.csv`.
* **Hasil Nyata**:  
  Tepat **120 file citra augmentasi** terbuat (masing-masing skenario memiliki 12 citra dengan total 12 wajah per skenario).

### Cell 15 [Markdown]: Pengantar EDA Pasca-Augmentasi
* **Tujuan**: Menjelaskan pentingnya melakukan verifikasi matematis dan visual untuk memastikan operasi augmentasi tidak merusak posisi bounding box.
* **Hasil**: Teks pengantar verifikasi integritas data.

### Cell 16 [Code / Cell 5A]: Verifikasi Integritas Bounding Box
* **Cara Kerja**:  
  Membaca ulang `augmented_annotations.csv` dan memeriksa setiap baris bounding box terhadap 3 kriteria error:
  1. *Out-of-bounds*: Ada koordinat $< 0$ atau $> \text{lebar/tinggi}$.
  2. *Degenerate*: Kotak terbalik (`xmax <= xmin` atau `ymax <= ymin`).
  3. *Too-small*: Luas kotak $< 100\text{ px}^2$.
* **Hasil Nyata**:  
  `Semua 120 bounding box valid!`: Tidak ditemukan satupun kotak cacat geometri. Rata-rata wajah per citra tepat 1.00.

### Cell 17 [Code / Cell 5B]: Analisis Distribusi Kecerahan dan Kontras
* **Cara Kerja**:  
  Mengambil sampel 80 citra augmentasi, mengukur kembali rata-rata kecerahan dan kontrasnya, lalu merender sepasang Boxplot horizontal yang mengelompokkan nilai-nilai tersebut per skenario. Garis putus-putus merah menandai rata-rata citra asli.
* **Istilah Teknis**:  
  *Boxplot*: Diagram kotak-garis yang menunjukkan nilai median, kuartil bawah (Q1), kuartil atas (Q3), dan nilai pencilan (outlier).
* **Hasil Nyata**:  
  Plot tersimpan di `results/plots/eda_augmentation_brightness.png`. Terbukti skenario Low Light dan Komposit berhasil menekan tingkat kecerahan secara drastis, sedangkan Overexposure meningkatkannya secara signifikan.

### Cell 18 [Code / Cell 5C]: Visualisasi Grid Sample 10 Skenario
* **Cara Kerja**:  
  Mengambil 1 citra contoh (`dpo_01.jpg`) dan merender ke-10 wujud modifikasinya dalam grid 2 baris x 5 kolom bersama kotak hijau Ground Truth.
* **Hasil Nyata**:  
  Gambar tersimpan di `results/plots/augmentation_sample_grid.png`. Secara visual tampak jelas bahwa pada skenario rotasi maupun flip, kotak hijau tetap melekat presisi di wajah.

### Cell 19 [Markdown]: Teori Wrapper & Unified Interface
* **Tujuan**: Menguraikan prinsip *Unified Interface* (antarmuka seragam): kedua detektor yang berbeda arsitektur dibungkus ke dalam class yang memiliki method input-output yang sama persis agar pengujian komparasi berjalan adil (*fair benchmarking*).
* **Hasil**: Teks penjelasan arsitektur model.

### Cell 20 [Code / Cell 6A]: Dataclass & Wrapper Fast MTCNN
* **Cara Kerja**:  
  1. Membuat dataclass `DetectionResult` yang menampung atribut `bboxes`, `scores`, `latency_ms`, dan `error`.
  2. Membuat class `FastMTCNNDetector` yang menginisialisasi model `facenet_pytorch.MTCNN`.
  3. Method `detect()` mengubah array gambar OpenCV menjadi objek PIL Image, mengukur durasi inferensi milidetik via `time.perf_counter()`, menjalankan model, dan mengembalikan koordinat piksel absolut.
* **Istilah Teknis**:  
  *Wrapper*: Pola desain pemrograman di mana suatu pustaka/fungsi dibungkus ke dalam class kustom agar cara memanggilnya sesuai dengan format yang kita inginkan.
* **Hasil Nyata**:  
  Class `FastMTCNNDetector` siap dipakai.

### Cell 21 [Code / Cell 6B]: Wrapper MediaPipe & Sanity Check
* **Cara Kerja**:  
  1. Membuat class `MediaPipeDetector` membungkus `mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)`.
  2. Mengonversi koordinat rasio normalisasi MediaPipe `(0.0 - 1.0)` menjadi koordinat piksel bilangan bulat absolut.
  3. Menginstansiasi kedua model dan melakukan *Sanity Check* (uji coba fungsi) pada citra pertama.
* **Istilah Teknis**:  
  *Sanity Check*: Pengujian kilat awal pada 1 data sampel untuk memastikan fungsi/model tidak error sebelum loop besar dijalankan.
* **Hasil Nyata**:  
  * FastMTCNN termuat pada CPU (1 deteksi, latensi 123.7 ms).
  * MediaPipe termuat pada mode full-range (1 deteksi, latensi 12.8 ms).
  * Kedua model sukses mendeteksi wajah pada pengujian awal.

### Cell 22 [Markdown]: Formulasi Matematika Metrik Evaluasi
* **Tujuan**: Menjabarkan rumus matematika IoU, batas ambang kecocokan PASCAL VOC (IoU $\ge 0.5$), Precision, Recall, F1-Score, dan FPS.
* **Hasil**: Teks dokumentasi matematis.

### Cell 23 [Code / Cell 7A]: Algoritma IoU dan Greedy Matching
* **Cara Kerja**:  
  1. Fungsi `compute_iou(boxA, boxB)` menghitung luas interseksi dan union dari dua kotak.
  2. Fungsi `match_detections_to_gt(pred_bboxes, gt_bboxes)` membuat matriks IoU berukuran $(N_{\text{pred}} \times N_{\text{GT}})$. Mengambil nilai IoU tertinggi secara serakah (*greedy*), memasangkannya jika $\ge 0.5$, lalu mengosongkan baris dan kolom yang sudah berpasangan.
  3. Menghitung nilai TP, FP, dan FN.
  4. Menjalankan unit test kotak overlap 50x50 piksel.
* **Istilah Teknis**:  
  *Greedy Matching*: Algoritma yang langsung memilih opsi terbaik saat itu juga (pasangan kotak dengan irisan terluas) satu per satu secara berurutan tanpa kembali ke belakang.
* **Hasil Nyata**:  
  Unit test IoU menghasilkan angka tepat `0.1429` dengan status `PASS`.

### Cell 24 [Code / Cell 7B]: Engine Benchmarking Utama (240 Pengujian)
* **Cara Kerja**:  
  Loop utama eksperimen: Menguji seluruh 120 citra augmentasi ke Fast MTCNN dan MediaPipe (total 240 kali deteksi).
  1. Citra RGB dimasukkan ke method `detect()`.
  2. Hasil bounding box dipasangkan ke GT via `match_detections_to_gt()`.
  3. Data TP, FP, FN, mean IoU, dan waktu latensi dicatat per baris ke list `eval_records`.
  4. List diekspor menjadi DataFrame dan disimpan ke `results/eval_raw_results.csv`.
* **Hasil Nyata**:  
  Selesai dalam 11 detik. Ringkasan mentah global:
  * FastMTCNN: TP = 120, FP = 3, FN = 0, Rata-rata Latensi = 85.81 ms.
  * MediaPipe: TP = 103, FP = 1, FN = 17, Rata-rata Latensi = 5.63 ms.

### Cell 25 [Markdown]: Pengantar Agregasi Micro-Averaging
* **Tujuan**: Memaparkan alasan penggunaan metode agregasi micro-averaging dibanding macro-averaging untuk menghindari distorsi bobot saat pembagian metrik per citra.
* **Hasil**: Teks pengantar konsep statistik evaluasi.

### Cell 26 [Code / Cell 8A]: Kalkulasi Metrik Global & Tabel Komparasi
* **Cara Kerja**:  
  Fungsi `compute_aggregate_metrics()` menghitung Precision, Recall, F1-Score, rata-rata IoU (hanya pada deteksi yang sukses/TP), rata-rata latensi, dan FPS untuk masing-masing model.
* **Hasil Nyata (Tabel Komparasi Utama)**:
  ```
      Model  Total TP  Total FP  Total FN  Precision  Recall  F1-Score  Avg IoU  Mean Latency (ms)  Throughput (FPS)
  FastMTCNN       120         3         0     0.9756  1.0000    0.9877   0.7520              85.81             11.65
  MediaPipe       103         1        17     0.9904  0.8583    0.9196   0.7699               5.63            177.52
  ```

### Cell 27 [Code / Cell 8B]: Breakdown Performa per Skenario & Heatmap
* **Cara Kerja**:  
  Mengelompokkan data evaluasi berdasarkan nama skenario augmentasi dan nama model, lalu menghitung metrik Precision, Recall, dan F1-Score. Data diubah menjadi tabel pivot dan divisualisasikan dalam 3 buah Heatmap berdampingan (Merah = Rendah, Hijau = Sempurna).
* **Istilah Teknis**:  
  *Heatmap*: Representasi visual 2 dimensi di mana nilai angka dinyatakan dalam gradasi warna.
* **Hasil Nyata**:  
  Gambar tersimpan di `results/plots/heatmap_breakdown.png`.  
  Terungkap fakta krusial:
  * Pada skenario Baseline, Low Light, Overexposure, Blur, JPEG, Flip, dan Rotasi, kedua model sama-sama memperoleh skor sempurna (F1 = 1.00).
  * Pada skenario **(6) ISO Noise**, MediaPipe mengalami penurunan drastis: Recall hanya **0.167** (hanya menemukan 2 dari 12 wajah) dan F1 anjlok ke **0.286**. Sementara Fast MTCNN tetap mencatat Recall **1.000**.
  * Pada skenario **(10) Komposit**, MediaPipe mencatat Recall **0.417** dan F1 **0.556**, sementara Fast MTCNN tetap stabil di Recall **1.000** dan F1 **0.960**.

### Cell 28 [Markdown]: Pengantar Analisis Visual & Failure Case
* **Tujuan**: Mengarahkan fokus investigasi ke representasi grafis, inspeksi visual panel, dan pembuktian penyebab terjadinya false negative.
* **Hasil**: Teks pengantar analisis kualitatif.

### Cell 29 [Code / Cell 9A]: Bar Chart Komparasi 4 Metrik Kunci
* **Cara Kerja**:  
  Merender 4 diagram batang (*bar chart*) berdampingan untuk metrik: Precision, Recall, F1-Score, dan Throughput (FPS) menggunakan palet warna standar (Biru untuk Fast MTCNN, Merah untuk MediaPipe).
* **Hasil Nyata**:  
  Grafik tersimpan di `results/plots/metric_comparison_bar.png`. Menunjukkan secara visual bahwa Fast MTCNN unggul pada Recall & F1, sedangkan MediaPipe unggul telak pada FPS.

### Cell 30 [Code / Cell 9B]: Visualisasi Side-by-Side 3 Panel
* **Cara Kerja**:  
  Fungsi `visualize_side_by_side()` membuat gambar komparasi 3 panel:
  * Panel Kiri: Citra asli + Kotak Hijau Ground Truth.
  * Panel Tengah: Citra + Kotak Biru Fast MTCNN beserta waktu komputasinya.
  * Panel Kanan: Citra + Kotak Merah MediaPipe beserta waktu komputasinya.  
  Dijalankan untuk 6 skenario sampel: `normal`, `low_light`, `gaussian_blur`, `jpeg_artifact`, `rotation`, dan `composite`.
* **Hasil Nyata**:  
  Tersimpan 6 file gambar di `results/plots/` (`sidebyside_normal.png`, `sidebyside_low_light.png`, dst.).

### Cell 31 [Code / Cell 9C]: Radar Chart Komparasi Multi-Dimensi
* **Cara Kerja**:  
  Membuat grafik radar / jaring laba-laba (*spider chart*) dengan 5 sumbu evaluasi: Precision, Recall, F1-Score, Average IoU, dan Speed (FPS yang dinormalisasi ke skala 0–1 terhadap nilai maksimum).
* **Hasil Nyata**:  
  Grafik tersimpan di `results/plots/radar_chart.png`. Menampilkan poligon visual yang memperlihatkan profil trade-off masing-masing arsitektur.

### Cell 32 [Code / Cell 9D]: Isolasi Kasus Kegagalan (Failure Cases)
* **Cara Kerja**:  
  Menyaring baris evaluasi yang memiliki nilai `fn > 0` (False Negative / wajah tidak terdeteksi). Menghitung jumlah kegagalan per skenario dan merender gambar kasus kegagalan terburuk dengan menumpuk kotak Ground Truth (hijau) dan kotak prediksi (merah).
* **Hasil Nyata**:  
  * Fast MTCNN: `0 FN dari 120 gambar` (Sempurna, tidak ada kasus kegagalan lolos).
  * MediaPipe: `17 FN dari 120 gambar` (10 kasus di ISO Noise dan 7 kasus di Komposit).  
  Plot gambar kegagalan tersimpan di `results/plots/failure_mediapipe.png`, memperlihatkan citra berbintik pasir tebal di mana MediaPipe sama sekali tidak memunculkan kotak prediksi.

### Cell 33 [Markdown]: Dokumentasi Ekspor Hasil
* **Tujuan**: Menjelaskan bahwa data hasil benchmarking diekspor ke format terstruktur untuk keperluan sitasi tabel Bab 4 laporan skripsi / artikel jurnal.
* **Hasil**: Teks pengantar ekspor.

### Cell 34 [Code / Cell 10A]: Ekspor Berkas CSV, JSON & Print Ringkasan
* **Cara Kerja**:  
  1. Menyimpan `df_comparison` ke `results_comparison.csv`.
  2. Menyimpan `df_breakdown` ke `results_breakdown_by_scenario.csv`.
  3. Menyimpan `df_eval` ke `eval_raw_results.csv`.
  4. Mengompilasi seluruh metadata sistem, seed, konfigurasi, skor numerik, dan status pemenang ke dalam dictionary, lalu diekspor menjadi format `experiment_summary.json`.
  5. Mencetak tabel rekapitulasi akhir pemenang per metrik di konsol.
* **Hasil Nyata**:  
  Empat berkas data terstruktur berhasil ditulis ke disk, dan papan skor akhir tercetak di layar:
  * Pemenang Precision: **MediaPipe** (0.9904 vs 0.9756)
  * Pemenang Recall: **Fast MTCNN** (1.0000 vs 0.8583)
  * Pemenang F1-Score: **Fast MTCNN** (0.9877 vs 0.9196)
  * Pemenang Avg IoU: **MediaPipe** (0.7699 vs 0.7520)
  * Pemenang Latency / FPS: **MediaPipe** (5.63 ms / 177.5 FPS vs 85.81 ms / 11.6 FPS)

### Cell 35 [Code / Cell 10B]: Verifikasi Inventori Artefak Output
* **Cara Kerja**:  
  Menelusuri seluruh file di dalam folder `results/` secara rekursif menggunakan `Path.rglob('*')`, lalu mencetak nama file relatif dan ukurannya dalam satuan kilobyte (KB).
* **Hasil Nyata**:  
  Terdata 21 file output lengkap (4 file tabel/data dan 17 file grafik plot PNG) dengan total ukuran penyimpanan $\approx 26\text{ MB}$.

### Cell 36 [Markdown]: Daftar Pustaka Akademik
* **Tujuan**: Menyediakan 15 daftar pustaka artikel jurnal dan prosiding konferensi internasional IEEE/Scopus (2021–2026) yang dikutip di sepanjang eksperimen untuk memperkuat landasan teori pada penulisan laporan penelitian.
* **Hasil**: Daftar sitasi formal.

---

## 3. Evaluasi Kritis & Blind Spot Metodologis

Tiga catatan evaluasi objektif yang wajib dipahami sebelum menyusun kesimpulan akhir:

1. **Ukuran Sampel Awal (N = 12 Individu)**:  
   Meskipun diekspansi menjadi 120 citra melalui 10 variasi transformasi terarah, sampel biometrik dasar hanya berasal dari 12 orang. Augmentasi menguji ketahanan terhadap degradasi optik dan sensor, bukan keberagaman demografis wajah manusia (usia, ras, oklusi kacamata/masker). Jangan mengklaim model sudah teruji secara universal pada populasi besar.
2. **Ketergantungan Label Awal pada Haar Cascade**:  
   Karena koordinat Ground Truth di Cell 2C dibangun otomatis via OpenCV Haar Cascade, akurasi nilai IoU sangat bergantung pada ketepatan kotak awal tersebut. Bounding box Haar Cascade cenderung sedikit longgar dibanding anotasi manual manusia.
3. **Trade-off Arsitektural dalam Konteks Lapangan**:  
   MediaPipe sangat unggul dari segi efisiensi komputasi (177 FPS), menjadikannya kandidat terbaik untuk aplikasi interaktif mobile atau webcam dengan pencahayaan terkontrol. Namun, untuk sistem pengawasan CCTV keamanan malam hari (kamera surveillance DPO) yang rentan terhadap sensor noise, MediaPipe tidak dapat diandalkan sendirian tanpa bantuan modul preprocessing denoiser karena tingkat kegagalannya mencapai 83.3% pada kondisi derau ISO tinggi. Fast MTCNN jauh lebih tangguh menghadapi gangguan visual meskipun harus mengorbankan kecepatan komputasi.
