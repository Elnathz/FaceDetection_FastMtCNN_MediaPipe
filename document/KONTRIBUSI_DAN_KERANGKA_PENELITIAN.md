# Kontribusi Ilmiah dan Kerangka Teoretis Penelitian
## Analisis Komparatif Kinerja Fast MTCNN dan MediaPipe pada Dataset DPO Bareskrim

Dokumen ini memuat artikulasi formal kontribusi penelitian, landasan teoretis komparasi, draf penulisan laporan akademik (Bab 1, Bab 4, dan Bab 5), serta skrip pertahanan argumen saat ujian sidang atau presentasi tugas Pengolahan Citra Digital (PCD).

---

## 1. Artikulasi Tiga Pilar Kontribusi Penelitian

Dalam penulisan naskah akademik (skripsi, tugas besar, atau artikel jurnal), kontribusi tidak boleh sekadar menyebutkan aktivitas teknis (seperti "kami menggunakan library X"). Kontribusi harus dijabarkan ke dalam tiga pilar utama: metodologis, empiris, dan rekayasa praktis.

### 1.1. Pilar Kontribusi Metodologis
* **Masalah yang Diatasi**:  
  Dataset Daftar Pencarian Orang (DPO) riil memiliki kendala intrinsik berupa keterbatasan data primer (hanya 1 foto pasfoto statis per individu) dan ketiadaan file anotasi bounding box resmi. Selain itu, pengujian deteksi wajah pada umumnya hanya dilakukan pada citra berkualitas tinggi (dataset bersih seperti CelebA atau LFW) yang tidak mencerminkan realitas optik kamera CCTV pengawas di lapangan.
* **Kontribusi Nyata**:  
  Membangun protokol *stress-test* multi-degradasi terstandarisasi berbasis *bounding box-preserving transformation*. Penelitian ini merancang 10 skenario degradasi visual terukur (ISO sensor noise, motion blur, defokus Gaussian, overexposure, kompresi JPEG agresif, hingga degradasi komposit ekstrem) dengan jaminan sinkronisasi koordinat kartesius Ground Truth secara otomatis. Hal ini membuktikan bahwa keterbatasan jumlah foto primer dapat diekspansi secara valid tanpa merusak akurasi kunci jawaban spasial.

### 1.2. Pilar Kontribusi Empiris (Temuan Fakta Ilmiah)
* **Masalah yang Diatasi**:  
  Banyak literatur terapan mengasumsikan bahwa detektor wajah modern generasi terbaru (seperti MediaPipe BlazeFace) selalu lebih unggul dibanding arsitektur klasik (seperti MTCNN) di semua aspek karena kecepatannya yang mencapai ratusan FPS.
* **Kontribusi Nyata**:  
  Menyingkap dan memetakan batas toleransi kegagalan (*failure boundary*) arsitektur *single-shot detector* berbobot ringan:
  1. **Kerapuhan Fatal terhadap Sensor Noise**: Membuktikan secara kuantitatif bahwa MediaPipe mengalami degradasi performa drastis ketika citra terkontaminasi oleh *high-frequency Gaussian noise* (simulasi sensor ISO tinggi malam hari), di mana nilai Recall anjlok hingga **0.071 (7.1%)** dengan kegagalan deteksi pada 13 dari 14 citra uji.
  2. **Ketahanan Mutlak Cascaded CNN**: Membuktikan keunggulan struktur piramida skala spasial pada Fast MTCNN (P-Net, R-Net, O-Net) yang secara alami berfungsi sebagai *low-pass filter*, mempertahankan Recall sempurna **1.000 (100%)** di seluruh skenario degradasi tanpa ada satu pun target DPO yang terlewat.
  3. **Analisis Batas Ambang IoU (Borderline IoU)**: Mengungkap fenomena sanksi ganda (penalti FP dan FN sekaligus) pada standar evaluasi PASCAL VOC (threshold 0.50) akibat perbedaan filosofi anotasi antara *facial-core* (BlazeFace) dan *full-head* (Haar/MTCNN), seperti yang teridentifikasi pada citra `dpo_01` (IoU 0.4879).

### 1.3. Pilar Kontribusi Praktis dan Rekayasa Sistem
* **Masalah yang Diatasi**:  
  Ketiadaan panduan komparatif pemilihan detektor wajah yang mempertimbangkan trade-off operasional antara ketelitian identifikasi kriminal berbanding keterbatasan daya komputasi di pos pengawasan lokal (*edge computing*).
* **Kontribusi Nyata**:  
  Merumuskan matriks keputusan rekayasa:
  1. Untuk sistem investigasi hukum kriminalitas (DPO Bareskrim), metrik **Recall mutlak adalah prioritas tertinggi** (biaya sosial lolosnya buronan jauh lebih besar dibanding beban komputasi), sehingga Fast MTCNN adalah pilihan arsitektur yang valid meskipun berkecepatan 11.56 FPS pada CPU.
  2. Memberikan rekomendasi arsitektur hibrida: jika sistem terpaksa menggunakan MediaPipe karena membutuhkan pemrosesan multi-kamera berkecepatan tinggi (179.57 FPS), pipeline wajib mengintegrasikan modul prapemrosesan *denoiser* (misalnya *bilateral filtering*) sebelum citra diumpankan ke model deteksi.

---

## 2. Draf Teks Siap Pakai untuk Naskah Laporan Ilmiah

Gunakan draf teks di bawah ini untuk disalin atau diadaptasi ke dalam dokumen laporan akhir Anda:

### Draf Bab 1: Pendahuluan

#### 1.1 Latar Belakang Masalah
Identifikasi Daftar Pencarian Orang (DPO) melalui rekaman visual merupakan instrumen krusial dalam penegakan hukum modern. Dalam skenario operasional di lapangan, sistem pengawasan video sering kali dihadapkan pada rekaman kamera pengawas (CCTV) berkualitas rendah yang mengalami berbagai degradasi optik dan transmisi, seperti pencahayaan minim di malam hari, derau sensor ber-ISO tinggi, kekaburan akibat gerakan subjek, serta artefak kompresi digital. Kondisi ini menuntut algoritma deteksi wajah yang tidak hanya beroperasi secara efisien, tetapi juga memiliki ketahanan tinggi terhadap degradasi visual ekstrem.

Dua arsitektur deteksi wajah yang banyak diadopsi saat ini adalah Multi-Task Cascaded Convolutional Networks (Fast MTCNN) dan MediaPipe Face Detection yang berbasis BlazeFace. Fast MTCNN mengandalkan pendekatan piramida bertingkat tiga tahap (Proposal, Refine, Output Network) yang dikenal akurat namun menuntut beban komputasi intensif. Di sisi lain, MediaPipe menawarkan efisiensi komputasi sangat tinggi melalui arsitektur single-shot yang dirancang khusus untuk perangkat berdaya rendah. Namun, evaluasi komparatif mendalam mengenai batas ketahanan kedua arsitektur ini di bawah simulasi degradasi terukur pada dataset lokal DPO Bareskrim Polri masih sangat terbatas.

#### 1.2 Rumusan Masalah
1. Bagaimana perbandingan performa kuantitatif (Precision, Recall, F1-Score, dan Average IoU) antara Fast MTCNN dan MediaPipe di bawah berbagai skenario degradasi visual?
2. Bagaimana efisiensi komputasi (latensi inferensi per frame dan throughput FPS) kedua model pada lingkungan pemrosesan berbasis CPU?
3. Skenario degradasi visual apa yang menjadi titik kritis kegagalan dari masing-masing arsitektur detektor?

#### 1.3 Tujuan Penelitian
1. Mengukur dan membandingkan akurasi deteksi dan lokalisasi wajah Fast MTCNN dan MediaPipe pada dataset DPO Bareskrim yang diperluas dengan 10 skenario augmentasi terkontrol.
2. Menganalisis trade-off antara keandalan tangkapan wajah (Recall) dan efisiensi waktu pemrosesan (FPS) pada lingkungan komputasi terbatas.
3. Memberikan rekomendasi teknis arsitektur deteksi wajah yang optimal untuk implementasi sistem penegakan hukum dan identifikasi kriminal.

#### 1.4 Batasan Masalah
1. Citra primer menggunakan 14 foto DPO Bareskrim Polri yang bersumber dari data tugas mahasiswa.
2. Pengujian dilakukan pada lingkungan komputasi CPU (Intel Core Ultra 7 255HX) tanpa akselerasi kartu grafis diskrit (GPU) guna merepresentasikan infrastruktur pos pengawas standar.
3. Batas ambang tumpang tindih keberhasilan deteksi mengacu pada standar PASCAL VOC (Intersection over Union >= 0.50).

#### 1.5 Kontribusi Penelitian
Kontribusi dari penelitian ini dijabarkan sebagai berikut:
1. **Kontribusi Metodologis**: Menyusun protokol benchmarking terstandarisasi berbasis 10 skenario degradasi optik dan sensor yang mempertahankan sinkronisasi koordinat Ground Truth kartesius (bounding box-preserving), mengatasi keterbatasan data foto primer tunggal tanpa merusak integritas evaluasi.
2. **Kontribusi Empiris**: Memetakan secara kuantitatif batas toleransi operasional kedua arsitektur, khususnya mengungkap kerentanan fatal detektor single-shot BlazeFace pada derau sensor (ISO noise Recall anjlok ke 7.1%), berbanding ketahanan 100% Recall pada arsitektur cascaded CNN Fast MTCNN.
3. **Kontribusi Praktis**: Menghasilkan matriks rekomendasi seleksi model untuk sistem surveillance penegakan hukum DPO yang memprioritaskan ketahanan Recall di atas kecepatan pemrosesan, serta merumuskan mitigasi modul denoiser pada integrasi MediaPipe.

---

### Draf Bab 4: Pembahasan Kritis (Discussion)

#### 4.3 Analisis Kritis Disparitas Kinerja dan Titik Kegagalan Model
Berdasarkan data evaluasi 140 citra uji (Tabel 4.1), Fast MTCNN membuktikan keunggulan mutlak dalam aspek ketahanan deteksi dengan meraih Recall sempurna 1.000 (tidak ada satu pun wajah yang terlewat, FN = 0) dan F1-Score 0.9790. Sebaliknya, MediaPipe mendominasi aspek efisiensi komputasi dengan throughput mencapai 179.57 FPS (latensi 5.57 ms), atau sekitar 15.5 kali lebih cepat dibanding Fast MTCNN (11.56 FPS / 86.49 ms).

Perbedaan performa yang paling mencolok teramati pada skenario (6) ISO Noise. Pada skenario ini, Recall MediaPipe merosot tajam ke angka 0.071, di mana model gagal mendeteksi 13 dari 14 wajah DPO yang diuji. Secara arsitektural, BlazeFace mengandalkan konvolusi ringan satu tahap (depthwise separable convolution 5x5) yang mengekstraksi gradien tepi halus untuk mencocokkan anchor wajah. Bintik derau acak berfrekuensi tinggi dari distribusi Gaussian merusak kontinuitas gradien intensitas tersebut. Hal ini menyebabkan skor probabilitas internal BlazeFace jatuh di bawah ambang batas deteksi (0.30), sehingga model tidak mengembalikan bounding box prediksi sama sekali.

Sebaliknya, Fast MTCNN mempertahankan Recall 1.000 pada skenario yang sama. Ketahanan ini dimungkinkan oleh mekanisme Image Pyramid pada Proposal Network (P-Net). Sebelum dilakukan konvolusi, citra masukan diubah ke berbagai resolusi spasial yang lebih kecil. Proses downsampling ini secara alami berfungsi sebagai filter pelemah derau (low-pass smoothing), sehingga pada skala citra yang lebih kasar, fitur makro wajah (struktur dahi, mata, dan hidung) tetap dapat dikenali oleh P-Net untuk kemudian divalidasi posisinya oleh R-Net dan O-Net.

Fenomena menarik lainnya ditemukan pada citra `dpo_01_composite.jpg`, di mana MediaPipe mencatatkan False Negative (FN = 1) dan False Positive (FP = 1) secara simultan, meskipun secara visual kotak prediksi merah melingkupi area mata, hidung, dan mulut subjek dengan baik. Analisis koordinat menunjukkan bahwa irisan antara tebakan MediaPipe dan Ground Truth menghasilkan IoU sebesar 0.4879. Mengacu pada aturan kaku PASCAL VOC (threshold IoU >= 0.5000), tebakan tersebut gugur hanya karena selisih 0.0121 (1.2%). Kasus ini membuktikan adanya disparitas filosofi anotasi antara mazhab facial-core (BlazeFace yang hanya mencakup alis hingga dagu) dan mazhab full-head (Ground Truth Haar Cascade yang menyertakan area rambut dan dahi).

---

### Draf Bab 5: Kesimpulan dan Saran

#### 5.1 Kesimpulan
1. Terdapat trade-off yang tajam antara keandalan lokalisasi dan efisiensi waktu komputasi: Fast MTCNN unggul dalam ketahanan deteksi (Recall 1.000, F1-Score 0.9790), sedangkan MediaPipe unggul dalam throughput komputasi CPU (179.57 FPS, latensi 5.57 ms).
2. Arsitektur single-shot MediaPipe memiliki kelemahan kritis terhadap gangguan sensor berfrekuensi tinggi (ISO noise), dengan penurunan Recall hingga 7.1%, sehingga tidak direkomendasikan untuk kamera pengawas malam hari tanpa prapemrosesan.
3. Untuk kebutuhan penegakan hukum dan identifikasi buronan DPO Bareskrim yang mengutamakan keberhasilan tangkapan subjek (zero false negative), Fast MTCNN merupakan pilihan arsitektur yang paling andal secara operasional.

#### 5.2 Saran
1. Pada penelitian lanjutan, disarankan melakukan anotasi ulang Ground Truth secara manual menggunakan poligon atau segmentasi wajah penuh guna menghindari distorsi nilai IoU pada model-model berkarakteristik facial-core.
2. Untuk implementasi praktis MediaPipe pada sistem surveillance malam hari, direkomendasikan mengintegrasikan modul prapemrosesan reduksi derau (seperti fast bilateral filter atau median filter) sebelum proses inferensi dilakukan.

---

## 3. Matriks Pertahanan Argumen saat Presentasi / Ujian Sidang

| Potensi Pertanyaan Dosen | Jawaban Teknis dan Bernas Anda |
|---|---|
| *"Kenapa datasetnya cuma 14 orang? Apakah itu cukup untuk sebuah penelitian?"* | "14 foto primer adalah representasi data DPO riil yang tersedia. Fokus riset kami bukan menguji keberagaman biometrik demografis, melainkan melakukan stress-test ketahanan algoritma terhadap 10 skenario degradasi visual terkontrol (total 140 citra uji). Jadi fokusnya adalah uji ketahanan terhadap degradasi optik kamera." |
| *"Kenapa tidak melatih ulang (fine-tuning) modelnya saja?"* | "Tujuan penelitian ini adalah benchmark komparatif out-of-the-box guna mengukur kinerja arsitektur dasar kedua model dalam kondisi zero-shot inference saat dideploy langsung pada infrastruktur pos lapangan." |
| *"MediaPipe kan jauh lebih cepat, kenapa tidak pakai MediaPipe saja untuk DPO?"* | "Kecepatan 179 FPS tidak ada artinya jika buronan kriminal gagal terdeteksi di kamera malam hari (Recall MediaPipe anjlok ke 7.1%). Pada kasus hukum kriminal, biaya kegagalan deteksi (FN) jauh lebih fatal dibanding penurunan kecepatan ke 11 FPS milik Fast MTCNN." |
| *"Apa penyebab kotak prediksi MediaPipe gagal di dpo_01?"* | "Itu adalah fenomena borderline IoU (IoU tercatat 0.4879 berbanding threshold 0.50). MediaPipe mendeteksi wajah dengan rapat (facial core), sedangkan Ground Truth awal mencakup rambut atas. Selisih 1.2% membuat sistem PASCAL VOC memberi penalti ganda (FN=1, FP=1) secara otomatis." |
