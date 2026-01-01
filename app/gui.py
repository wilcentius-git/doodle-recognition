import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as transforms

from src.config import DEVICE, IMG_SIZE, IMG_SIZE_ML
from src.feature_engineering import extract_sketch_features
from src.smart_crop import preprocess_sketch_smart_crop

class SketchApp:
    def __init__(self, root, dl_models, ml_models, scaler, classes):
        self.root = root
        self.root.title("Ultimate Sketch Battle: Smart Crop Edition")
        self.dl_models = dl_models
        self.ml_models = ml_models
        self.classes = classes
        
        self.main_frame = tk.Frame(root, padx=10, pady=10)
        self.main_frame.pack()
        
        # Canvas
        self.canvas_frame = tk.LabelFrame(self.main_frame, text="Draw Here", font=("Arial", 12, "bold"))
        self.canvas_frame.grid(row=0, column=0, padx=10)
        
        self.canvas_width = 650
        self.canvas_height = 650
        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_width, height=self.canvas_height, bg="white", cursor="cross")
        self.canvas.pack()
        
        # Image Memory: BG Putih
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.btn_frame = tk.Frame(self.canvas_frame)
        self.btn_frame.pack(pady=5)
        self.btn_predict = tk.Button(self.btn_frame, text="PREDICT ALL", command=self.predict, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        self.btn_predict.pack(side=tk.LEFT, padx=5)
        self.btn_clear = tk.Button(self.btn_frame, text="CLEAR", command=self.clear_canvas, bg="#f44336", fg="white", font=("Arial", 10, "bold"))
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # Result
        self.result_frame = tk.LabelFrame(self.main_frame, text="Leaderboard", font=("Arial", 12, "bold"))
        self.result_frame.grid(row=0, column=1, padx=10, sticky="ns")
        
        self.bars = {}
        def create_section(name, color):
            frame = tk.Frame(self.result_frame, pady=2)
            frame.pack(fill="x")
            tk.Label(frame, text=name, font=("Arial", 9, "bold"), fg=color).pack(anchor="w")
            return self.create_bars(frame, color)

        tk.Label(self.result_frame, text="--- Deep Learning ---", font=("Arial", 8)).pack()
        self.bars["CNN (BatchNorm)"] = create_section("CNN (BatchNorm)", "purple")
        self.bars["ResNet18"] = create_section("ResNet18", "darkgreen")
        
        tk.Label(self.result_frame, text="--- Machine Learning ---", font=("Arial", 8)).pack(pady=(10,0))
        self.bars["Random Forest"] = create_section("Random Forest", "brown")

    def create_bars(self, parent, color):
        bars = {}
        style_name = f"{color}.Horizontal.TProgressbar"
        style = ttk.Style()
        style.configure(style_name, background=color)
        for cls in self.classes:
            frame = tk.Frame(parent)
            frame.pack(fill="x")
            tk.Label(frame, text=cls, width=8, anchor="w", font=("Arial", 8)).pack(side=tk.LEFT)
            progress = ttk.Progressbar(frame, orient="horizontal", length=100, mode="determinate", style=style_name)
            progress.pack(side=tk.LEFT, padx=5)
            perc_lbl = tk.Label(frame, text="0%", width=4, font=("Arial", 8))
            perc_lbl.pack(side=tk.LEFT)
            bars[cls] = (progress, perc_lbl)
        return bars

    def paint(self, event):
        BRUSH_DISPLAY = 5
        BRUSH_DATA = 20 
        r = BRUSH_DISPLAY // 2
        self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill="black", outline="black")
        self.draw.line([(event.x, event.y), (event.x, event.y)], fill=0, width=BRUSH_DATA, joint="curve")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)
        for model_name in self.bars:
            self.reset_bars(self.bars[model_name])

    def reset_bars(self, bars):
        for cls in self.classes:
            bars[cls][0]['value'] = 0
            bars[cls][1].config(text="0%")

    def update_bars(self, bars, probabilities):
        for i, cls in enumerate(self.classes):
            prob_percent = probabilities[i] * 100
            bars[cls][0]['value'] = prob_percent
            bars[cls][1].config(text=f"{prob_percent:.1f}%")

    def predict(self):
        try:
            print("Memulai prediksi...")
            # Pake Smart Crop dari src/smart_crop.py
            processed_img = preprocess_sketch_smart_crop(self.image)
            
            # 1. DL Inference
            img_dl = processed_img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            transform = transforms.Compose([transforms.ToTensor()])
            tensor_dl = transform(img_dl).unsqueeze(0).to(DEVICE)
            
            # 2. ML Inference (Feature Engineering dari src/feature_engineering.py)
            img_ml = processed_img.resize((IMG_SIZE_ML, IMG_SIZE_ML), Image.Resampling.LANCZOS)
            arr_ml = np.array(img_ml).flatten()
            features_ml = extract_sketch_features(arr_ml, IMG_SIZE_ML) 

            arr_ml_rf = features_ml.reshape(1, -1)

            # Predict DL
            with torch.no_grad():
                for name, model in self.dl_models.items():
                    output = model(tensor_dl)
                    probs = torch.nn.functional.softmax(output, dim=1)[0]
                    self.update_bars(self.bars[name], probs.cpu().numpy())

            # Predict ML
            probs_rf = self.ml_models['Random Forest'].predict_proba(arr_ml_rf)[0]
            self.update_bars(self.bars['Random Forest'], probs_rf)
            print("Prediksi selesai!")
            
        except Exception as e:
            print(f"Error saat prediksi: {e}")
            import traceback
            traceback.print_exc()
