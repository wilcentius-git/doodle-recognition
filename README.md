# 🎨 Ultimate Sketch Battle: Smart Crop Edition

Project ini adalah aplikasi klasifikasi gambar sketsa (doodle) menggunakan pendekatan **Hybrid AI**: Deep Learning (CNN, ResNet, MobileNet) dan Machine Learning (Random Forest, SVM) dengan fitur ekstraksi manual.

Aplikasi ini dilengkapi dengan **Smart Crop Algorithm** untuk memotong area putih pada canvas agar fokus pada gambar sketsa.

## 📂 Struktur Project
```text
sketch-classification/
├── src/         # Source code utama (Training, Models, Logic)
├── app/         # Source code GUI & Inference
├── data/        # Folder dataset (Raw & Processed)
├── notebooks/   # Eksperimen Jupyter Notebook
├── outputs/     # Hasil training (Model .pth & Logs)
└── config/      # Konfigurasi Hyperparameter