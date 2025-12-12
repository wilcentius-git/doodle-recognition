import os
import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

from src.config import DATASET_PATH, SELECTED_CLASSES, IMG_SIZE, BATCH_SIZE, IMG_SIZE_ML
from src.feature_engineering import extract_sketch_features

# Custom Transform
class InvertImage:
    def __call__(self, img):
        return ImageOps.invert(img)

def get_dataloaders():
    print(f"--- Membaca Dataset dari: {DATASET_PATH} ---")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Folder dataset tidak ditemukan di: {DATASET_PATH}")

    # TRANSFORMASI
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        InvertImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
    ])

    # Load Dataset
    full_dataset = ImageFolder(root=DATASET_PATH, transform=train_transform)
    
    # Filter Classes
    class_to_idx = {cls_name: i for i, cls_name in enumerate(SELECTED_CLASSES)}
    original_class_to_idx = full_dataset.class_to_idx
    valid_indices = [original_class_to_idx[cls] for cls in SELECTED_CLASSES if cls in original_class_to_idx]

    filtered_samples = []
    
    for path, target in full_dataset.samples:
        if target in valid_indices:
            cls_name = full_dataset.classes[target]
            new_target = class_to_idx[cls_name]
            filtered_samples.append((path, new_target))

    full_dataset.samples = filtered_samples
    full_dataset.classes = SELECTED_CLASSES
    full_dataset.class_to_idx = class_to_idx
    
    print(f"Total gambar: {len(full_dataset)}")

    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {train_size}, Validation: {val_size}")
    
    # Data ML
    print("Menyiapkan data ML dengan Feature Engineering...")
    X_ml, y_ml = [], []
    for path, target in filtered_samples:
        img = Image.open(path).convert('L')
        img = ImageOps.invert(img)
        img = img.resize((IMG_SIZE_ML, IMG_SIZE_ML))
        arr = np.array(img).flatten()
        features = extract_sketch_features(arr, IMG_SIZE_ML)
        X_ml.append(features)
        y_ml.append(target)
        

    return train_loader, val_loader, np.array(X_ml), np.array(y_ml)
