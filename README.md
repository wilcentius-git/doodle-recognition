# 🎨 Unified Sketch Classification System Using Deep Learning, Machine Learning, and Smart-Crop Preprocessing

Project ini adalah aplikasi klasifikasi gambar sketsa (doodle) menggunakan pendekatan **Hybrid AI**: Deep Learning (CNN, ResNet, MobileNet) dan Machine Learning (Random Forest, SVM) dengan fitur ekstraksi manual.

Aplikasi ini dilengkapi dengan **Smart Crop Algorithm** untuk memotong area putih pada canvas agar fokus pada gambar sketsa sebelum diproses oleh AI.

## 📂 Struktur Project
```text
doodle-recognition/
├── src/         # Source code utama (Training, Models, Logic)
├── app/         # Source code GUI & Inference
├── data/        # Folder dataset 
├── notebooks/   # Eksperimen Jupyter Notebook
├── outputs/     # Hasil training (Model .pth & Logs)
├── report/      # Dokumentasi berupa paper
└── config/      # Konfigurasi Hyperparameter
```

---
```markdown
## 🚀 Panduan Instalasi & Penggunaan

Ikuti langkah-langkah berikut secara berurutan untuk menjalankan aplikasi ini di komputer Anda.

### 1️⃣ Persiapan Environment
Pastikan Anda sudah menginstall **Python 3.8** atau versi yang lebih baru.

Buka terminal (Command Prompt / PowerShell / Terminal) di dalam folder utama project (`doodle-recognition/`), lalu jalankan perintah berikut untuk menginstall library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

> **Note:** Pastikan koneksi internet stabil karena library seperti `torch` dan `torchvision` memiliki ukuran yang cukup besar.

---

### 2️⃣ Persiapan Dataset
Dataset harus diletakkan pada struktur folder yang spesifik agar script dapat membacanya.

1. Siapkan file dataset (biasanya berupa file `.zip` atau folder hasil download).
2. Ekstrak dataset tersebut ke dalam folder `data/`.
3. Pastikan struktur foldernya terlihat **persis** seperti di bawah ini:

```text
doodle-recognition/
│
├── data/
│   └── Doodle Dataset by Ashish Jangra/
│       └── doodle/
│           ├── airplane/   <-- Berisi gambar pesawat
│           ├── candle/     <-- Berisi gambar lilin
│           ├── car/        <-- Berisi gambar mobil
│           ├── diamond/    <-- Berisi gambar berlian
│           └── fish/       <-- Berisi gambar ikan
```

> ⚠️ **PENTING:** Jangan sampai ada folder ganda (nested folders) seperti `doodle/doodle/airplane`. Pastikan path-nya sesuai dengan struktur di atas.

---

### 3️⃣ Menjalankan Aplikasi
Setelah dataset siap dan library terinstall, Anda siap menjalankan aplikasi.

Jalankan perintah berikut di terminal (dari folder root project):

```bash
python main.py
```

### ⏳ Apa yang terjadi setelah perintah dijalankan?

1. **Pengecekan Model:** Sistem akan mengecek apakah file model (`outputs/models/doodle_models.pth`) sudah ada.
2. **Training Otomatis (Jika Model Belum Ada):**
   - Jika ini pertama kali dijalankan, sistem akan otomatis melakukan **Training** (Deep Learning & Machine Learning).
   - Proses ini memakan waktu beberapa menit tergantung spesifikasi komputer (GPU/CPU).
   - Anda akan melihat progress bar training di terminal.
3. **Membuka GUI:**
   - Setelah training selesai (atau jika model sudah ada), jendela aplikasi **"Ultimate Sketch Battle"** akan terbuka.
   - Anda bisa mulai menggambar di canvas dan menekan tombol **PREDICT ALL**.

---

## 🛠 Features
- **Smart Crop**: Otomatis memotong canvas kosong dan memusatkan gambar.
- **Real-time Leaderboard**: Membandingkan probabilitas 6 model sekaligus.
- **Inverted Processing**: Mengubah input canvas (Putih) menjadi data training (Hitam) secara otomatis.
- **Hybrid Architecture**: Menggabungkan kekuatan Neural Network modern dengan algoritma ML klasik.

## 🧠 Model List
1. **Deep Learning:** MLP (Dropout), Custom CNN, ResNet18, MobileNetV2.
2. **Machine Learning:** Random Forest, SVM (Linear Kernel).

---

**Created for Deep Learning Course - AOL Project**
