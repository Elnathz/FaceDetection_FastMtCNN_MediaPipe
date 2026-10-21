# Analisis Perbandingan Kinerja Fast MTCNN dan MediaPipe dalam Deteksi Wajah pada Dataset DPO Bareskrim

> **Mata Kuliah**: Pengolahan Citra Digital (PCD) — Universitas Dian Nuswantoro  
> **Dataset**: DPO Bareskrim Polri (12 Citra Primer $\rightarrow$ 120 Citra Augmentasi Bounding Box-Preserving)  
> **Model yang Diuji**: Fast MTCNN (*Cascade CNN*) vs MediaPipe Face Detection (*BlazeFace Single-Shot*)  
> **Metode Evaluasi**: Precision, Recall, F1-Score, Average IoU, Latency (ms), Throughput (FPS)

---

## 📌 Ringkasan Eksekutif (Executive Summary)

Penelitian ini mengevaluasi kinerja empiris dua arsitektur deteksi wajah modern di bawah skenario penegakan hukum dan identifikasi Daftar Pencarian Orang (DPO) kriminal yang rentan terhadap degradasi visual:
1. **Fast MTCNN**: Arsitektur *Multi-Task Cascaded Convolutional Networks* 3 tahap (P-Net, R-Net, O-Net).
2. **MediaPipe Face Detection**: Arsitektur *single-shot detector* berbasis *BlazeFace* yang dioptimasi untuk komputasi instan (*sub-millisecond*).

### Temuan Kunci (Key Findings):
* **Fast MTCNN unggul mutlak dalam ketahanan visual (Robustness)**: Meraih **Recall 100%** ($FN=0$) dan **F1-Score 0.9877** di seluruh 10 skenario pengujian. Fast MTCNN terbukti kebal terhadap derau sensor (*sensor noise*) dan kekaburan ekstrem (*motion/gaussian blur*).
* **MediaPipe unggul mutlak dalam efisiensi komputasi (Speed)**: Mencapai throughput **177.52 FPS** dengan latensi hanya **5.63 ms** pada pengujian CPU, sekitar **15 kali lebih cepat** dibandingkan Fast MTCNN (11.65 FPS / 85.81 ms).
* **Titik Kritis MediaPipe**: MediaPipe mengalami kegagalan fatal pada skenario **ISO Noise** (Recall anjlok ke **16.7%**, 10 dari 12 wajah gagal terdeteksi) dan skenario **Komposit** (Recall **41.7%**). Tanpa filter *denoiser*, MediaPipe tidak dapat diandalkan untuk kamera pengawas malam hari berkualitas rendah.

---

## 📊 Hasil Komparasi Kinerja Global

Pengujian dilakukan secara objektif pada **120 citra uji augmentasi** dengan ambang batas kecocokan PASCAL VOC ($\text{IoU} \ge 0.5$):

| Metrik Evaluasi | Fast MTCNN | MediaPipe | Pemenang | Catatan Analisis |
| :--- | :---: | :---: | :---: | :--- |
| **True Positive (TP)** | **120** | 103 | Fast MTCNN | Fast MTCNN mendeteksi seluruh target wajah. |
| **False Positive (FP)** | 3 | **1** | MediaPipe | MediaPipe sangat jarang membuat deteksi palsu. |
| **False Negative (FN)** | **0** | 17 | Fast MTCNN | MediaPipe melewatkan 17 wajah buronan. |
| **Precision** | 0.9756 | **0.9904** | MediaPipe | MediaPipe unggul tipis dalam kemurnian tebakan. |
| **Recall** | **1.0000** | 0.8583 | Fast MTCNN | Fast MTCNN sempurna tanpa ada wajah yang lolos. |
| **F1-Score** | **0.9877** | 0.9196 | Fast MTCNN | Keseimbangan akurasi total dipegang Fast MTCNN. |
| **Average IoU** | 0.7520 | **0.7699** | MediaPipe | Kerapatan kotak deteksi MediaPipe sedikit lebih pas. |
| **Mean Latency (ms)** | 85.81 ms | **5.63 ms** | MediaPipe | MediaPipe 15.2x lebih cepat per frame. |
| **Throughput (FPS)** | 11.65 FPS | **177.52 FPS** | MediaPipe | MediaPipe sangat ideal untuk *real-time processing*. |

---

## 🔬 Rincian Performa per Skenario (Breakdown Matrix)

12 citra primer ditransformasikan ke dalam 10 variasi kondisi visual berbasis temuan literatur surveillance:

| # | Skenario Augmentasi | Fast MTCNN (F1 / Recall) | MediaPipe (F1 / Recall) | Analisis Ketahanan |
|---|---|:---:|:---:|---|
| 1 | **Normal / Baseline** | 1.000 / 1.000 | 1.000 / 1.000 | Kedua model sempurna pada kondisi ideal. |
| 2 | **Low Light** (Pencahayaan Redup) | 1.000 / 1.000 | 1.000 / 1.000 | Tahan terhadap penurunan intensitas cahaya. |
| 3 | **Overexposure** (Cahaya Berlebih) | 1.000 / 1.000 | 1.000 / 1.000 | Tahan terhadap saturasi cahaya terang. |
| 4 | **Gaussian Blur** (Defokus Kamera) | 0.960 / 1.000 | 1.000 / 1.000 | Keduanya mampu menemukan wajah buram. |
| 5 | **Motion Blur** (Pergerakan Cepat) | 0.960 / 1.000 | 1.000 / 1.000 | Tahan terhadap blur pergerakan objek. |
| 6 | **ISO Noise** (Derau Sensor CCTV) | **1.000 / 1.000** | **0.286 / 0.167** | **Kelemahan fatal MediaPipe (10 FN).** |
| 7 | **JPEG Artifact** (Kompresi Ekstrem) | 1.000 / 1.000 | 1.000 / 1.000 | Keduanya kebal terhadap artefak kompresi transmisi. |
| 8 | **Horizontal Flip** (Cermin Sudut) | 1.000 / 1.000 | 1.000 / 1.000 | Simetri wajah terdeteksi sempurna. |
| 9 | **Rotasi Ringan** ($\pm 15^\circ$) | 1.000 / 1.000 | 1.000 / 1.000 | Tahan terhadap kemiringan kepala. |
| 10 | **Komposit** (Gelap + Blur + Noise) | **0.960 / 1.000** | **0.556 / 0.417** | **MediaPipe gagal pada 7 dari 12 citra.** |

---

## 🛠️ Arsitektur & Metodologi

### 1. Fast MTCNN (Multi-Task Cascaded CNN)
* **Library**: `facenet-pytorch`
* **Mekanisme**:
  * **P-Net (Proposal Network)**: Mengusulkan calon jendela wajah secara cepat melalui piramida skala citra.
  * **R-Net (Refine Network)**: Menyaring kandidat kasar dan menolak sebagian besar *false positive*.
  * **O-Net (Output Network)**: Menghasilkan koordinat *bounding box* final dan regresi 5 titik landmark wajah.
* **Karakteristik**: Sangat tahan banting terhadap *noise*, namun membutuhkan iterasi konvolusi multi-tahap yang membebani komputasi CPU.

### 2. MediaPipe Face Detection (BlazeFace)
* **Library**: `mediapipe` (mode `model_selection=1` untuk jangkauan jarak jauh / full-range)
* **Mekanisme**:
  * Arsitektur saraf *single-shot* berbasis anchor dengan modul *BlazeBlock* ringan.
  * Mampu memprediksi kotak dan titik acuan wajah dalam sekali jalan (*single forward pass*).
* **Karakteristik**: Kecepatan ultra-tinggi (*sub-millisecond*), namun rentan gagal mengenali batas tepi wajah ketika citra terkontaminasi piksel acak berfrekuensi tinggi (ISO noise).

### 3. Pipeline Ground Truth & Evaluasi
* **Inisialisasi GT**: Menggunakan OpenCV Haar Cascade sebagai *proxy annotation* yang disimpan di `results/ground_truth.json`.
* **Sinkronisasi Augmentasi**: Library `albumentations` mengompensasi dan menghitung ulang koordinat geometris *bounding box* secara matematis saat citra dirotasi atau di-flip.
* **PASCAL VOC Matching**: Algoritma *greedy matching* memasangkan kotak tebakan terhadap Ground Truth dengan syarat $\text{IoU} \ge 0.5$.

---

## 📁 Struktur Direktori Repositori

```text
FaceDetection_FastMtCNN_MediaPipe/
├── Dataset PCD 4703 - DPO/          # Folder dataset asli dari kelompok mahasiswa
├── dataset/
│   ├── original/                    # 12 Citra terstandarisasi (dpo_01.jpg s/d dpo_12.jpg)
│   └── augmented/                   # 120 Citra hasil 10 skenario augmentasi
├── results/
│   ├── plots/                       # 17 Grafik visualisasi & diagram publikasi (300 DPI)
│   │   ├── augmentation_sample_grid.png
│   │   ├── eda_augmentation_brightness.png
│   │   ├── eda_statistics.png
│   │   ├── failure_fastmtcnn.png
│   │   ├── failure_mediapipe.png
│   │   ├── grid_original_gt.png
│   │   ├── heatmap_breakdown.png
│   │   ├── metric_comparison_bar.png
│   │   ├── radar_chart.png
│   │   └── sidebyside_*.png
│   ├── augmented_annotations.csv    # Anotasi koordinat BBox 120 citra augmentasi
│   ├── eval_raw_results.csv         # Rekaman mentah 240 iterasi pengujian
│   ├── experiment_summary.json      # Metadata hardware, konfigurasi, & ringkasan hasil
│   ├── ground_truth.json            # Database koordinat Ground Truth 12 citra asli
│   ├── results_breakdown_by_scenario.csv
│   └── results_comparison.csv
├── face_detection_comparison.ipynb  # Notebook Jupyter utama seluruh pipeline eksperimen
├── PANDUAN_TEKNIS_DAN_BEDAH_NOTEBOOK.md # Panduan komprehensif bedah tiap cell & istilah CV
├── Analisis_Referensi_Paper.xlsm    # Matriks pemetaan referensi penelitian
├── Referensi Paper.xlsx             # Rekapitulasi paper jurnal pendukung
└── README.md                        # Dokumentasi utama proyek
```

---

## 🚀 Panduan Menjalankan Eksperimen

### 1. Prasyarat Sistem
* Python 3.10 atau lebih baru
* Sistem Operasi: Windows 10/11, macOS, atau Linux
* Direkomendasikan RAM $\ge 8\text{ GB}$

### 2. Instalasi Dependensi
Clone repositori dan pasang pustaka yang diperlukan:
```bash
git clone https://github.com/Elnathz/FaceDetection_FastMtCNN_MediaPipe.git
cd FaceDetection_FastMtCNN_MediaPipe

pip install mediapipe facenet-pytorch "albumentations>=1.3.0" opencv-python pandas seaborn matplotlib scikit-learn Pillow tqdm torch torchvision psutil
```

### 3. Eksekusi Notebook
Jalankan Jupyter Notebook atau Jupyter Lab:
```bash
jupyter notebook face_detection_comparison.ipynb
```
Pilih opsi **Run All Cells**. Notebook akan secara otomatis:
1. Mendeteksi hardware CPU/RAM sistem Anda.
2. Memuat citra primer dan menyinkronkan Ground Truth.
3. Menghasilkan 120 citra augmentasi.
4. Menjalankan pengujian komparasi 240 inferensi.
5. Menyimpan tabel CSV, ringkasan JSON, serta seluruh plot visualisasi ke folder `results/`.

---

## 📚 Referensi Ilmiah Terpilih

1. **Shah et al. (2024)**. Real-time Performance Comparison of Face Detection Algorithms using Raspberry Pi. *IRJAEH*, 2(10), 1224–1230.
2. **Tripathi et al. (2025)**. Towards Safer Communities: Real-Time MTCNN-Based Face Recognition for Criminal and Missing Person Identification. *Proc. IEEE ICAICCIT 2025*.
3. **Go, Alzami, & Naufal (2025)**. Comprehensive Benchmark of Yolov11n, SSD MobileNet, CenterFace, Yunet, FastMtCnn, HaarCascade, and LBP for Face Detection in Video Based Driver Drowsiness. *BITS*, 7(3), 8678–8689.
4. **Fajar et al. (2026)**. Analysis of the Effect of Lighting Variations on the Performance of a Face Recognition System Based on FaceNet and MTCNN. *IJSTH*, 4(2), 72–81.
5. **Dahiya et al. (2026)**. Face Detection for Surveillance and Security: A Comparative Analysis of Deep Learning Architectures. *Proc. IEEE ICSSCNA 2026*.
6. **Priya & Minu (2026)**. From Shadows to Faces: Retinex–GAN and TransYOLO-FaceLite for Tiny and Occluded Face Detection in Low-Light CCTV. *IJIES*, 19(4), 410–422.
7. **Khabarlak, K. (2021)**. Mobile Face Detection Algorithm Inference Traits. *System Technologies*, 6(137), 35–46.

---

## 👥 Tim Penyusun
Penelitian ini disusun untuk memenuhi tugas mata kuliah **Pengolahan Citra Digital (PCD)**, Program Studi Teknik Informatika, **Universitas Dian Nuswantoro (UDINUS)**.
