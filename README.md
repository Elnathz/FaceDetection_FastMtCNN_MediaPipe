# Benchmark Deteksi Wajah: Fast MTCNN vs MediaPipe pada Dataset DPO Bareskrim

Komparasi empiris performa deteksi wajah antara arsitektur multi-stage cascaded CNN (Fast MTCNN) dan single-shot anchor-based detector (MediaPipe BlazeFace) pada skenario citra terdegradasi.

## Metadata Penelitian
* Mata Kuliah: Pengolahan Citra Digital (PCD)
* Institusi: Universitas Dian Nuswantoro (UDINUS)
* Dataset: 12 citra primer DPO Bareskrim diperluas menjadi 120 citra uji (10 skenario degradasi terkontrol)
* Standar Evaluasi: PASCAL VOC (Intersection over Union >= 0.5)
* Lingkungan Komputasi: Intel Core Ultra 7 255HX (20C/20T), 31.43 GB RAM, OS Windows 11 (64-bit)

---

## 1. Ringkasan Eksekutif dan Trade-off

Eksperimen ini mengevaluasi batas operasional dua paradigma deteksi wajah modern ketika dihadapkan pada degradasi optik dan sensor.

```text
Trade-off Arsitektur:
[Fast MTCNN]   High Robustness (Recall 1.000)  <--->  Low Throughput (11.65 FPS)
[MediaPipe]    Low Robustness (Recall 0.858)   <--->  High Throughput (177.52 FPS)
```

### Temuan Kritis:
1. Fast MTCNN unggul mutlak dalam keandalan deteksi: Mencapai Recall 1.000 (tidak ada target wajah yang terlewat, FN = 0) dan F1-Score 0.9877 pada seluruh kondisi uji.
2. MediaPipe unggul mutlak dalam efisiensi komputasi: Menghasilkan latensi inferensi 5.63 ms per frame (177.52 FPS pada CPU), sekitar 15.2 kali lebih cepat dibanding Fast MTCNN (85.81 ms / 11.65 FPS).
3. Titik Kegagalan MediaPipe: MediaPipe mengalami degradasi performa drastis pada citra dengan noise frekuensi tinggi (ISO sensor noise), dengan Recall merosot ke 0.167 (10 dari 12 wajah gagal terdeteksi).

---

## 2. Tabel Hasil Komparasi Global

Metrik dihitung menggunakan micro-averaging pada total 120 citra uji (240 siklus inferensi):

| Metrik Evaluasi | Fast MTCNN | MediaPipe | Selisih Absolut | Model Terbaik |
| :--- | :---: | :---: | :---: | :---: |
| True Positive (TP) | 120 | 103 | +17 frame | Fast MTCNN |
| False Positive (FP) | 3 | 1 | -2 frame | MediaPipe |
| False Negative (FN) | 0 | 17 | -17 frame | Fast MTCNN |
| Precision | 0.9756 | 0.9904 | +0.0148 | MediaPipe |
| Recall | 1.0000 | 0.8583 | +0.1417 | Fast MTCNN |
| F1-Score | 0.9877 | 0.9196 | +0.0681 | Fast MTCNN |
| Average IoU | 0.7520 | 0.7699 | +0.0179 | MediaPipe |
| Latensi Rata-rata (ms) | 85.81 ms | 5.63 ms | -80.18 ms | MediaPipe |
| Throughput (FPS) | 11.65 FPS | 177.52 FPS | +165.87 FPS | MediaPipe |

---

## 3. Analisis Performa per Skenario Degradasi

12 citra primer ditransformasikan ke dalam 10 kondisi visual menggunakan pustaka Albumentations dengan sinkronisasi koordinat bounding box (format Pascal VOC: `[xmin, ymin, xmax, ymax]`):

| ID | Skenario Degradasi | Parameter Transformasi | Fast MTCNN (F1 / Recall) | MediaPipe (F1 / Recall) |
|---|---|---|:---:|:---:|
| 01 | Normal / Baseline | Tanpa modifikasi (kontrol) | 1.000 / 1.000 | 1.000 / 1.000 |
| 02 | Low Light | Brightness [-0.45, -0.30], Contrast [-0.20, 0.00] | 1.000 / 1.000 | 1.000 / 1.000 |
| 03 | Overexposure | Brightness [0.25, 0.40], Contrast [0.00, 0.15] | 1.000 / 1.000 | 1.000 / 1.000 |
| 04 | Gaussian Blur | Kernel limit [7, 15] | 0.960 / 1.000 | 1.000 / 1.000 |
| 05 | Motion Blur | Kernel limit [9, 19] | 0.960 / 1.000 | 1.000 / 1.000 |
| 06 | ISO Noise | Gaussian noise variance [800, 2500] | 1.000 / 1.000 | 0.286 / 0.167 |
| 07 | JPEG Artifact | Compression quality [5, 20] | 1.000 / 1.000 | 1.000 / 1.000 |
| 08 | Horizontal Flip | Cermin horizontal 100% | 1.000 / 1.000 | 1.000 / 1.000 |
| 09 | Rotasi Ringan | Rotasi acak limit [-15, +15] derajat | 1.000 / 1.000 | 1.000 / 1.000 |
| 10 | Komposit | Redup (-35%) + Noise (var 1200) + Blur (kernel 7) | 0.960 / 1.000 | 0.556 / 0.417 |

---

## 4. Analisis Arsitektural: Penyebab Disparitas Kinerja

Perbedaan performa kedua model berakar pada desain arsitektur jaringan neural:

```text
Fast MTCNN (3-Stage Cascade Pipeline):
Input Citra -> [P-Net: Kandidat Kasar] -> [R-Net: Eliminasi Non-Wajah] -> [O-Net: Regresi BBox & Landmark] -> Output Final

MediaPipe (BlazeFace Single-Shot Pipeline):
Input Citra -> [Feature Extractor (BlazeBlock)] -> [Single-Pass Multi-Scale Anchors] -> Output Final
```

### Mengapa Fast MTCNN Tahan Terhadap Noise?
Struktur bertingkat P-Net, R-Net, dan O-Net menerapkan penyaringan bertahap. Jika piksel noise mengaburkan tepi wajah, P-Net tetap mengusulkan kandidat area berdasarkan struktur global, sementara O-Net memverifikasi ulang fitur internal wajah (mata, hidung, mulut). Kelemahannya, proses multi-tahap ini memerlukan siklus komputasi berulang sehingga latensi mencapai 85.81 ms.

### Mengapa MediaPipe Gagal pada Noise Frekuensi Tinggi?
BlazeFace dirancang untuk operasi sub-milidetik pada prosesor mobile dengan memotong kedalaman jaringan dan menggunakan konvolusi kedalaman terpisah (depthwise separable convolution). Noise acak berintensitas tinggi merusak kontur gradien spasial yang dibutuhkan anchor detector satu tahap, menyebabkan model melewatkan wajah sepenuhnya tanpa tahap verifikasi sekunder.

---

## 5. Struktur Direktori Repositori

```text
.
|-- Dataset PCD 4703 - DPO/          # Data mentah 12 foto DPO per kelompok mahasiswa
|-- dataset/
|   |-- original/                    # 12 citra standar (dpo_01.jpg s/d dpo_12.jpg)
|   `-- augmented/                   # 120 citra hasil 10 skenario degradasi
|-- results/
|   |-- plots/                       # 17 visualisasi analitis (resolusi 300 DPI)
|   |   |-- augmentation_sample_grid.png
|   |   |-- eda_augmentation_brightness.png
|   |   |-- eda_statistics.png
|   |   |-- failure_mediapipe.png
|   |   |-- heatmap_breakdown.png
|   |   |-- metric_comparison_bar.png
|   |   |-- radar_chart.png
|   |   `-- sidebyside_*.png
|   |-- augmented_annotations.csv    # Anotasi BBox 120 citra augmentasi
|   |-- eval_raw_results.csv         # Log mentah 240 iterasi pengujian
|   |-- experiment_summary.json      # Metadata hardware dan ringkasan metrik
|   |-- ground_truth.json            # Database koordinat ground truth 12 citra primer
|   |-- results_breakdown_by_scenario.csv
|   `-- results_comparison.csv
|-- face_detection_comparison.ipynb  # Notebook implementasi dan eksekusi eksperimen
|-- PANDUAN_TEKNIS_DAN_BEDAH_NOTEBOOK.md # Bedah teknis komprehensif per cell dan teori CV
|-- Analisis_Referensi_Paper.xlsm    # Matriks analisis referensi penelitian
|-- Referensi Paper.xlsx             # Rekapitulasi pustaka ilmiah pendukung
`-- README.md                        # Dokumentasi teknis proyek
```

---

## 6. Prosedur Replikasi Eksperimen

### Prasyarat
* Python 3.10 atau versi lebih baru
* pip package manager

### Langkah Instalasi
1. Clone repositori:
```bash
git clone https://github.com/Elnathz/FaceDetection_FastMtCNN_MediaPipe.git
cd FaceDetection_FastMtCNN_MediaPipe
```

2. Pasang pustaka dependensi:
```bash
pip install mediapipe facenet-pytorch "albumentations>=1.3.0" opencv-python pandas seaborn matplotlib scikit-learn Pillow tqdm torch torchvision psutil
```

3. Jalankan notebook:
```bash
jupyter notebook face_detection_comparison.ipynb
```
Pilih opsi **Run All Cells** untuk menjalankan seluruh pipeline mulai dari ekstraksi ground truth, augmentasi, benchmarking, visualisasi komparatif, hingga ekspor artefak data.

---

## 7. Rekomendasi Penerapan di Lapangan

* Skenario CCTV Pengawasan dan Penegakan Hukum (DPO):
  Gunakan Fast MTCNN. Dalam investigasi kriminal, nilai Recall 100% adalah prioritas absolut untuk menghindari risiko lolosnya subjek buronan. Throughput 11 FPS memadai untuk analisis rekaman video pengawas.
* Skenario Interaktif (Webcam / Presensi / Edge Devices):
  Gunakan MediaPipe. Throughput 177 FPS memungkinkan eksekusi real-time tanpa latensi terasa pada perangkat berspesifikasi rendah. Jika dioperasikan pada kondisi malam hari, integrasikan filter denoiser (misal: bilateral filter) pada tahap prapemrosesan sebelum citra diumpankan ke model.

---

## 8. Referensi Akademik Terkait

1. Shah, H. G., Suthar, V. B., Thakkar, S. P., & Thumar, V. M. (2024). Real-time Performance Comparison of Face Detection Algorithms using Raspberry Pi. International Research Journal on Advanced Engineering and Hub (IRJAEH), 2(10), 1224-1230.
2. Tripathi, M. K., Deshmukh, G., Reddy, K. N. S., Adithya, D., Dhruv, G., & Reddy, G. Y. (2025). Towards Safer Communities: Real-Time MTCNN-Based Face Recognition for Criminal and Missing Person Identification. Proc. IEEE ICAICCIT 2025.
3. Go, A. A. D., Alzami, F., & Naufal, M. (2025). Comprehensive Benchmark of Yolov11n, SSD MobileNet, CenterFace, Yunet, FastMtCnn, HaarCascade, and LBP for Face Detection in Video Based Driver Drowsiness. BITS, 7(3), 8678-8689.
4. Fajar, P., Sahriani, & Mustamin, S. B. (2026). Analysis of the Effect of Lighting Variations on the Performance of a Face Recognition System Based on FaceNet and MTCNN. IJSTH, 4(2), 72-81.
5. Dahiya, N., Jaiganesh, H., Mangla, I., & Mool, A. (2026). Face Detection for Surveillance and Security: A Comparative Analysis of Deep Learning Architectures. Proc. IEEE ICSSCNA 2026.
6. Priya, S., & Minu. (2026). From Shadows to Faces: Retinex-GAN and TransYOLO-FaceLite for Tiny and Occluded Face Detection in Low-Light CCTV. IJIES, 19(4), 410-422.
7. Khabarlak, K. (2021). Mobile Face Detection Algorithm Inference Traits. System Technologies, 6(137), 35-46.
8. Bazarevsky, V., et al. (2019). BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs. arXiv:1907.05047.
9. Zhang, K., et al. (2016). Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks. IEEE SPL, 23(10), 1499-1503.
