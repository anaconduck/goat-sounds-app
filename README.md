# 🐐 SapaKambing AI: Dataset Klasifikasi Suara Kambing Etawa (Data Primer)

Selamat datang di repositori pembelajaran **SapaKambing AI**! Repositori ini berisi dataset audio primer suara Kambing Etawa (Peranakan Etawa - PE) yang dikumpulkan langsung dari peternakan, serta diolah menggunakan kecerdasan buatan untuk mengklasifikasi kondisi emosi dan fisiologis kambing menjadi tiga kategori utama: **Lapar**, **Normal**, dan **Stress**.

Dokumen ini dirancang sebagai **media pembelajaran** interaktif untuk memahami konsep *Audio Signal Processing* (Pemrosesan Sinyal Suara) dan implementasi *Machine Learning* dalam dunia peternakan modern (*Smart Farming*).

---

## 📌 Daftar Isi
1. [Latar Belakang & Data Primer](#1-latar-belakang--data-primer)
2. [Kategori Suara & Labeling](#2-kategori-suara--labeling)
3. [Spesifikasi & Struktur Dataset](#3-spesifikasi--struktur-dataset)
4. [Alur Preprocessing Audio](#4-alur-preprocessing-audio)
5. [Ekstraksi Fitur: Mengenal MFCC](#5-ekstraksi-fitur-mengenal-mfcc)
6. [Eksperimen Pemodelan Machine Learning](#6-eksperimen-pemodelan-machine-learning)
7. [Panduan Menjalankan Project](#7-panduan-menjalankan-project)
8. [Panduan Belajar: Kode Sederhana](#8-panduan-belajar-kode-sederhana)

---

## 1. Latar Belakang & Data Primer

### 💡 Mengapa Mengidentifikasi Suara Kambing?
Dalam industri peternakan, mendeteksi kesejahteraan hewan (*animal welfare*) secara dini sangat krusial. Kambing berkomunikasi melalui suara (embikan). Perubahan frekuensi, amplitudo, dan pola embikan mencerminkan kondisi internal kambing. Dengan bantuan AI, peternak dapat memonitor kondisi ratusan kambing secara otomatis tanpa harus berada di kandang secara konstan.

### 🚜 Apa itu Data Primer?
Dataset ini bersifat **Data Primer**, artinya data dikumpulkan secara langsung oleh tim peneliti dari subjek asli di lapangan (bukan mengambil dataset sekunder/publik dari internet). 
* **Subjek**: Kambing Peranakan Etawa (PE).
* **Metode Perekaman**: Menggunakan perekam suara berkualitas tinggi dengan skenario terkontrol di kandang untuk mengisolasi kebisingan sekitar (background noise) semaksimal mungkin.

---

## 2. Kategori Suara & Labeling

Suara kambing dalam dataset ini diklasifikasikan ke dalam 3 kategori berdasarkan pengamatan perilaku (*behavioral observations*) saat perekaman:

| Ikon | Kategori | Penjelasan Perilaku | Karakteristik Suara |
| :---: | :--- | :--- | :--- |
| 🍽️ | **Lapar** | Kambing aktif mendekati tempat pakan, mengembik berulang saat jam pakan terlambat. | Suara cenderung ritmis, bernada sedang-tinggi, dan berdurasi pendek secara berulang. |
| 😊 | **Normal** | Kambing sedang rileks, mengunyah rumput (rumenasi), atau bersosialisasi secara tenang. | Suara cenderung bernada rendah, tenang, berdurasi pendek, dan jarang terdengar. |
| 😰 | **Stress** | Kambing terisolasi dari kelompok, merasa terancam, kesakitan, atau suhu lingkungan terlalu panas. | Suara bernada sangat tinggi (melengking), amplitudo tidak stabil, dan durasi hembusan suara lebih panjang. |

---

## 3. Spesifikasi & Struktur Dataset

### 📊 Distribusi Sampel
Dataset akhir terdiri dari **717 sampel audio** dalam format `.wav` berdurasi **2.0 detik** yang telah seimbang secara struktur namun memiliki ketidakseimbangan kelas (*class imbalance*) alami yang ditangani di level pemodelan (*class weighting*):

> [!IMPORTANT]
> **Akses Dataset**: Dikarenakan ukuran file audio yang besar, folder `Dataset/` dan `Rawdataset/` diabaikan oleh Git dan tidak diunggah ke repositori GitHub ini. Jika Anda ingin menggunakan dataset ini untuk keperluan pembelajaran atau riset, silakan ajukan permintaan akses melalui tautan Google Drive berikut:
> 🔗 **[Ajukan Akses Google Drive Dataset](https://link-ke-google-drive-anda-disini.com)**

```text
📊 Distribusi Kelas:
  ├── 🍽️ Lapar   : 247 file (34.4%)  [██████████░░░░░░░░░░░░░░]
  ├── 😊 Normal  : 373 file (52.0%)  [████████████████░░░░░░░░]
  └── 😰 Stress  :  97 file (13.5%)  [████░░░░░░░░░░░░░░░░░░░░]
  └── 📂 Total   : 717 file
```

### 📁 Struktur Folder Project

```text
Kambing/
│
├── .venv/                    # Virtual Environment Python (opsional)
├── Dataset/                  # Dataset audio siap pakai (2.0s .wav) -> Terbagi per kelas
│   ├── lapar/                # Audio kambing lapar hasil segmentasi
│   ├── normal/               # Audio kambing normal hasil segmentasi
│   └── stress/               # Audio kambing stress hasil segmentasi
│
├── Rawdataset/               # Dataset audio mentah (.m4a) & hasil konversi awal
│   ├── lapar/                # File rekaman suara lapar mentah (.m4a)
│   ├── normal/               # File rekaman suara normal mentah (.m4a)
│   ├── lapar_wav/            # Hasil konversi & pemotongan suara lapar (.wav 2.0s & noise)
│   └── normal_wav/           # Hasil konversi & pemotongan suara normal (.wav 2.0s & noise)
│
├── Notebooks/                # Eksperimen & pengolahan data (Jupyter Notebooks)
│   ├── preprocessing.ipynb   # Konversi m4a ke wav penuh, segmentasi 2 detik, & eliminasi noise
│   ├── eda.ipynb             # Analisis data eksploratif (visualisasi waveform & spektrogram)
│   └── modeling.ipynb        # Eksperimen pelatihan model AI (DNN, CNN, LSTM)
│
├── app.py                    # Aplikasi web interaktif klasifikasi suara berbasis Streamlit
├── requirements.txt          # Daftar dependensi library Python
├── best_model_dnn_final.keras # Model terbaik hasil training yang siap digunakan di app.py
├── label_encoder.pkl         # Encoder label kelas (lapar, normal, stress)
├── norm_params.npy           # Parameter normalisasi fitur audio
└── README.md                 # Dokumentasi panduan project (file ini)
```

---

## 4. Alur Preprocessing Audio

Sebelum suara kambing dapat dimasukkan ke dalam algoritma Machine Learning/Deep Learning, suara tersebut harus melewati tahapan pembersihan dan penyeragaman berikut:

**Alur Tahapan Preprocessing:**
`Audio Mentah (.m4a)` ──> `Konversi ke WAV (ffmpeg)` ──> `Segmentasi (2.0s)` ──> `Filter Noise & Silence` ──> `Ekstraksi Fitur MFCC`

**Detail Tahapan Pembelajaran:**
1. **Audio Mentah (.m4a)**: File audio rekaman dari kandang dengan durasi yang tidak seragam.
2. **Konversi ke WAV**: Mengubah format m4a menjadi `.wav` mono dengan sample rate 22050 Hz menggunakan subprocess FFmpeg.
3. **Segmentasi 2.0 Detik**: Memotong file audio wav panjang menjadi potongan-potongan kecil berdurasi tetap 2.0 detik.
4. **Filter Noise & Silence**: Memisahkan segmen hening atau noise (misal gesekan angin) menggunakan analisis RMS Energy, Zero-Crossing Rate (ZCR), Spectral Centroid, dan Spectral Flatness. Segmen suara kambing valid diberi nama `{kelas}_{index}.wav`, sedangkan noise diberi nama `noise_{index}.wav`.
5. **Ekstraksi Fitur MFCC**: Transformasi klip audio 2 detik menjadi matriks fitur Mel-Frequency Cepstral Coefficients (MFCC) 2D yang merepresentasikan karakteristik suara biologis kambing.

---

## 5. Ekstraksi Fitur: Mengenal MFCC

Suara dalam bentuk gelombang waktu (*waveform*) sangat sulit dipahami oleh komputer karena terlalu sensitif terhadap waktu. Oleh karena itu, kita mengubahnya ke domain frekuensi menggunakan **MFCC (Mel-Frequency Cepstral Coefficients)**.

### ❓ Apa itu MFCC?
MFCC adalah representasi spektrum jangka pendek dari suara yang didasarkan pada persepsi pendengaran biologis. Karena telinga manusia (dan mamalia lain seperti kambing) tidak mendengar frekuensi secara linear, MFCC menggunakan **Skala Mel** (skala logaritmik) agar komputer memproses suara mirip dengan cara telinga biologis mendengarnya.

### 🧬 Fitur yang Diekstrak dalam Project Ini:
Untuk setiap file audio, kita mengekstrak:
1. **MFCC Utama (40 Koefisien)**: Menangkap bentuk dasar amplop spektral (timbre suara).
2. **Delta MFCC (40 Koefisien)**: Turunan pertama dari MFCC, menunjukkan bagaimana suara berubah dari satu bingkai (*frame*) ke bingkai berikutnya (kecepatan perubahan).
3. **Delta-Delta MFCC (40 Koefisien)**: Turunan kedua, menunjukkan akselerasi perubahan spektral.

**Total Fitur Mentah**: 40 + 40 + 40 = 120 fitur per-frame.

### 📊 Representasi Vektor untuk Model
Karena durasi audio diseragamkan ke 2.0 detik dengan hop length 256, kita mendapatkan sekitar **125 frames**. 
Untuk klasifikasi standar menggunakan Deep Neural Network (DNN), matriks dimensi (120 x 125) ini dirata-ratakan (*mean*) dan dihitung deviasi standarnya (*std*) di sepanjang sumbu waktu, menghasilkan **240 fitur flat** (120 mean + 120 std) yang siap di-input ke model.

---

## 6. Eksperimen Pemodelan Machine Learning

Tiga arsitektur Deep Learning populer diuji coba dalam notebook `Notebooks/modeling.ipynb`:
1. **DNN (Deep Neural Network)**: Menggunakan fitur flat (240 dimensi) dengan layer padat (*Dense layers*), Dropout, dan Batch Normalization. Sangat cepat dan efisien.
2. **CNN (Convolutional Network)**: Memperlakukan spektrogram MFCC 2D (120 x 125) seperti gambar untuk menangkap pola tekstur visual suara.
3. **LSTM (Long Short-Term Memory)**: Model sekuensial untuk menangkap pola waktu (*temporal sequence*) dari perubahan embikan dari awal hingga akhir detik ke-2.

*Catatan: Model terbaik disimpan dalam berkas `best_model_dnn_final.keras` untuk dijalankan secara langsung di aplikasi Streamlit.*

---

## 7. Panduan Menjalankan Project

Ikuti langkah-langkah di bawah ini untuk menjalankan project ini dari awal di komputer lokal Anda:

### ⚙️ Prasyarat
* Python versi **3.10** atau **3.12**
* Pip (Python Package Installer)

### 🏃 Langkah-Langkah Running

#### 1. Clone Repositori
Clone project ini ke komputer lokal Anda dan masuk ke direktori project:
```bash
git clone https://github.com/username-anda/Kambing.git
cd Kambing
```

#### 2. Siapkan Struktur Folder Kosong & Download Dataset
Karena ukuran file audio yang besar, dataset tidak disertakan di GitHub. Setelah melakukan clone, Anda wajib melakukan hal berikut agar terhindar dari *error* file tidak ditemukan:
1. **Download Dataset**: Minta ijin akses ke link Google Drive dataset (bisa hubungi email kontributor di bawah), lalu download file rekaman mentahnya.
2. **Buat Folder Kosong**: Buat struktur folder berikut di dalam folder utama project (jika belum ada):
   - `Rawdataset/lapar`
   - `Rawdataset/normal`
   - `Rawdataset/stress`
   - `Dataset/`
   - `Results/`
3. Letakkan file rekaman mentah (biasanya format `.m4a`) yang sudah didownload ke dalam masing-masing folder di dalam `Rawdataset/` (misal: audio lapar ke `Rawdataset/lapar`).

#### 3. Buat & Aktifkan Virtual Environment
Sangat disarankan menggunakan virtual environment agar library tidak bentrok dengan sistem global.
* **Windows**:
  ```bash
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **Linux / macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### 4. Instal Dependensi (Requirements)
Instal semua library yang dibutuhkan menggunakan file `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 5. Preprocessing & Segmentasi Suara
Buka notebook [Notebooks/preprocessing.ipynb](file:///c:/S2-Ilmu%20Komputer%2024/Project/Kambing/Notebooks/preprocessing.ipynb) di editor Jupyter Notebook atau VS Code, lalu jalankan semua cell (Run All). Notebook ini akan otomatis:
1. Mendeteksi file rekaman mentah `.m4a` di folder `Rawdataset/lapar` dan `Rawdataset/normal`.
2. Mengonversinya ke format `.wav` dan memotongnya menjadi segmen 2.0 detik.
3. Memfilter dan memilah mana yang merupakan **suara kambing** (masuk ke file `{kelas}_{index}.wav`) dan mana yang merupakan **noise/hening** (masuk ke file `noise_{index}.wav`).

#### 6. Eksplorasi Data & Modeling
* Jalankan `Notebooks/eda.ipynb` untuk melihat analisis visual sinyal suara.
* Jalankan `Notebooks/modeling.ipynb` untuk melakukan ekstraksi fitur MFCC secara massal dan melatih model jaringan saraf (Deep Learning). Model terbaik akan disimpan secara otomatis.

#### 7. Jalankan Aplikasi Web Streamlit
Untuk menjalankan antarmuka web klasifikasi suara kambing interaktif secara lokal, jalankan perintah berikut:
```bash
streamlit run app.py
```
Buka alamat URL lokal yang muncul (biasanya `http://localhost:8501`) di browser Anda untuk mengunggah file suara kambing dan melihat hasil prediksinya secara real-time.

---

## 8. Panduan Belajar: Kode Sederhana

Gunakan kode Python berikut untuk mencoba mengekstrak fitur MFCC secara mandiri dari salah satu file audio dalam dataset ini:

```python
import librosa
import numpy as np

# 1. Muat file audio
audio_path = "Dataset/normal/normal 1.wav"
y, sr = librosa.load(audio_path, sr=16000, duration=2.0)

# 2. Ekstrak MFCC dasar (40 koefisien)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=512, hop_length=256)

# 3. Hitung Delta & Delta-Delta
delta_mfcc = librosa.feature.delta(mfcc)
delta2_mfcc = librosa.feature.delta(mfcc, order=2)

# 4. Gabungkan ketiga fitur
combined_features = np.vstack([mfcc, delta_mfcc, delta2_mfcc]) # Output: (120, 126)

# 5. Ekstrak statistik mean dan std untuk input DNN
mean_features = np.mean(combined_features, axis=1)
std_features = np.std(combined_features, axis=1)
final_feature_vector = np.hstack([mean_features, std_features]) # Output: (240,)

print("Dimensi fitur akhir untuk input AI:", final_feature_vector.shape)
```

---

## 👥 Kontributor & Kontak

* **Pembuat / Pemilik Repositori**: Wahyu Kusuma Wardhana
* **Email**: [wahyukusumaw29@gmail.com](mailto:wahyukusumaw29@gmail.com)
* **Tujuan**: Penelitian & Pengembangan Smart Farming Klasifikasi Audio Kambing Etawa.

---

*Dibuat dengan 💚 oleh Wahyu Kusuma Wardhana untuk Media Pembelajaran Smart Farming & Teknologi AI.*
