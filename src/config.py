import torch
import os

# ==========================================
# KONFIGURASI
# ==========================================

# Asumsi kita menjalankan main.py dari root folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path Dataset (Sesuaikan jika nama foldernya beda)
DATASET_PATH = os.path.join(BASE_DIR, "data", "Doodle Dataset by Ashish Jangra", "doodle")
OUTPUT_MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
MODEL_SAVE_PATH = os.path.join(OUTPUT_MODEL_DIR, "doodle_models.pth")

os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

SELECTED_CLASSES = ['airplane', 'candle', 'car', 'diamond', 'fish'] 

FORCE_RETRAIN = False

# Hyperparameters
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE_ML = 32
