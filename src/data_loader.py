import os
import torch
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm

from src.config import DATASET_PATH, SELECTED_CLASSES, IMG_SIZE, BATCH_SIZE, IMG_SIZE_ML
from src.feature_engineering import extract_sketch_features

# Custom Transform
class InvertImage:
    def __call__(self, img):
        return ImageOps.invert(img)


# ML Feature Extraction (from file paths)
def extract_ml_data(samples, desc="ML feature extraction"):
    """
    samples: list of (path, target) tuples where path is a string file path.
    """
    X, y = [], []
    for path, target in tqdm(samples, desc=desc, total=len(samples)):
        img = Image.open(path).convert("L")
        img = ImageOps.invert(img)
        img = img.resize((IMG_SIZE_ML, IMG_SIZE_ML))

        arr = np.array(img).flatten()
        feats = extract_sketch_features(arr, IMG_SIZE_ML)

        X.append(feats)
        y.append(target)

    return np.array(X), np.array(y)

# Main Data Loader (3-way split)
def get_dataloaders(
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42,
    build_ml=True,
    ml_max_samples=None
):
    """
    Returns:
      - train_loader, val_loader, test_loader
      - X_train_ml, y_train_ml, X_val_ml, y_val_ml, X_test_ml, y_test_ml (or None if build_ml=False)
    """

    print(f"--- Membaca Dataset dari: {DATASET_PATH} ---")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATASET_PATH}")

    # Transformasi untuk DL (train-time augmentations)
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        InvertImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
    ])

    # Load dataset
    full_dataset = ImageFolder(root=DATASET_PATH, transform=train_transform)

    # Filter classes and remap labels to 0..C-1
    class_to_idx = {cls_name: i for i, cls_name in enumerate(SELECTED_CLASSES)}
    original_class_to_idx = full_dataset.class_to_idx

    missing = [c for c in SELECTED_CLASSES if c not in original_class_to_idx]
    if missing:
        raise ValueError(
            f"Selected classes not found in dataset folder names: {missing}. "
            f"Available classes (sample): {list(original_class_to_idx.keys())[:25]}"
        )

    valid_indices = [original_class_to_idx[cls] for cls in SELECTED_CLASSES]

    filtered_samples = []
    for path, target in full_dataset.samples:
        if target in valid_indices:
            cls_name = full_dataset.classes[target]
            new_target = class_to_idx[cls_name]
            filtered_samples.append((path, new_target))

    full_dataset.samples = filtered_samples
    full_dataset.classes = SELECTED_CLASSES
    full_dataset.class_to_idx = class_to_idx

    total_size = len(full_dataset)
    print(f"Total gambar setelah filtering: {total_size}")
    print(f"Classes: {SELECTED_CLASSES}")

    # 3-way split (Train / Val / Test)
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1) or (train_ratio + val_ratio) >= 1:
        raise ValueError("train_ratio and val_ratio must be in (0,1) and train_ratio+val_ratio < 1")

    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    g = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=g
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Split -> Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # ML data follows EXACT SAME split using indices (paths)
    if build_ml:
        print("Menyiapkan data ML dengan feature engineering (mengikuti split yang sama)...")

        train_samples = [full_dataset.samples[i] for i in train_dataset.indices]
        val_samples = [full_dataset.samples[i] for i in val_dataset.indices]
        test_samples = [full_dataset.samples[i] for i in test_dataset.indices]

        if ml_max_samples is not None:
            train_samples = train_samples[:ml_max_samples]
            val_samples = val_samples[:ml_max_samples]
            test_samples = test_samples[:ml_max_samples]
            print(f"[ML] Using max_samples per split: {ml_max_samples}")

        X_train_ml, y_train_ml = extract_ml_data(train_samples, desc="ML features (train)")
        X_val_ml, y_val_ml = extract_ml_data(val_samples, desc="ML features (val)")
        X_test_ml, y_test_ml = extract_ml_data(test_samples, desc="ML features (test)")
    else:
        X_train_ml = y_train_ml = None
        X_val_ml = y_val_ml = None
        X_test_ml = y_test_ml = None

    return (
        train_loader,
        val_loader,
        test_loader,
        X_train_ml, y_train_ml,
        X_val_ml, y_val_ml,
        X_test_ml, y_test_ml
    )

# Optional sanity-check runner
if __name__ == "__main__":
    print("=== DATA LOADER SANITY CHECK (DL only) ===")

    (
        train_loader,
        val_loader,
        test_loader,
        X_train_ml, y_train_ml,
        X_val_ml, y_val_ml,
        X_test_ml, y_test_ml
    ) = get_dataloaders(build_ml=False)

    print("\n[DL]")
    print(f"Train batches : {len(train_loader)}")
    print(f"Val batches   : {len(val_loader)}")
    print(f"Test batches  : {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Sample train batch images shape: {images.shape} | labels shape: {labels.shape}")

    print("\n[ML]")
    print("Skipped ML feature extraction for sanity check.")
    print("To quick-check ML extraction, run in Python/Notebook:")
    print("  from src.data_loader import get_dataloaders")
    print("  get_dataloaders(build_ml=True, ml_max_samples=200)")
    print("\n=== DATA LOADER CHECK SELESAI ===")
