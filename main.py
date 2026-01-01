import os
import tkinter as tk
import sys

# Tambahkan root ke python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import DEVICE, FORCE_RETRAIN, MODEL_SAVE_PATH
from src.train import train_all_models
from app.model_loader import load_models_for_inference
from app.gui import SketchApp

if __name__ == "__main__":
    print(f"Using Device: {DEVICE}")
    print(f"Model path: {MODEL_SAVE_PATH}")
    
    model_exists = os.path.exists(MODEL_SAVE_PATH)
    
    if FORCE_RETRAIN or not model_exists:
        print("\n=== MEMULAI TRAINING ULANG ===")
        try:
            train_all_models()
            model_exists = True
        except Exception as e:
            print(f"\nError Training: {e}")
            import traceback
            traceback.print_exc()
            model_exists = False

# LAMA
    if model_exists:
        print("\n=== MEMBUKA APLIKASI GUI ===")
        try:
            dl_models, ml_models, scaler, classes = load_models_for_inference()
            
            root = tk.Tk()
            app = SketchApp(root, dl_models, ml_models, scaler, classes)
            root.mainloop()

# BARU (Hapus scaler)
    if model_exists:
        print("\n=== MEMBUKA APLIKASI GUI ===")
        try:
            # Scaler dihapus dari return value
            dl_models, ml_models, classes = load_models_for_inference()
            
            root = tk.Tk()
            # Scaler dihapus dari argumen
            app = SketchApp(root, dl_models, ml_models, classes)
            root.mainloop()
