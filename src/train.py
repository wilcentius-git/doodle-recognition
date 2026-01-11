import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    DEVICE, EPOCHS, LEARNING_RATE,
    SELECTED_CLASSES, MODEL_SAVE_PATH
)
from src.models_dl import TunedCNN, get_resnet18
from src.models_ml import get_rf_model
from src.data_loader import get_dataloaders
from src.utils import set_seed

# Helper: CSV Logger
def init_csv_logger(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_accuracy", "val_accuracy", "train_loss"])


def log_csv(path: str, row):
    with open(path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def _get_env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_int(name: str, default=None):
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default

# Training Function
def train_all_models():
    set_seed(42)

    # ML extraction controls (useful for debugging)
    # Default: build_ml=True (since RF training needs ML features)
    # Optional debug:
    #   set DOODLE_BUILD_ML=0   -> skip ML feature extraction (RF will be skipped)
    #   set DOODLE_ML_MAX_SAMPLES=200 -> limit samples per split for fast run
    build_ml = _get_env_bool("DOODLE_BUILD_ML", True)
    ml_max_samples = _get_env_int("DOODLE_ML_MAX_SAMPLES", default=None)

    print(f"[Config] DOODLE_BUILD_ML={build_ml}")
    print(f"[Config] DOODLE_ML_MAX_SAMPLES={ml_max_samples}")

    (
        train_loader,
        val_loader,
        test_loader,  # disiapkan untuk evaluasi akhir (evaluate.ipynb / evaluate.py)
        X_train_ml, y_train_ml,
        X_val_ml, y_val_ml,
        X_test_ml, y_test_ml
    ) = get_dataloaders(build_ml=build_ml, ml_max_samples=ml_max_samples)

    num_classes = len(SELECTED_CLASSES)

    # Deep Learning Models
    dl_models = {
        "CNN (BatchNorm)": TunedCNN(num_classes).to(DEVICE),
        "ResNet18": get_resnet18(num_classes).to(DEVICE)
    }

    criterion = nn.CrossEntropyLoss()
    saved_states = {"classes": SELECTED_CLASSES}

    print("\n=== TRAINING DEEP LEARNING MODELS ===")
    for name, model in dl_models.items():
        print(f"\n>>> Training {name}")

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Log file per model
        log_path = f"outputs/training_logs/{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        init_csv_logger(log_path)

        for epoch in range(EPOCHS):
            # -------- TRAIN --------
            model.train()
            train_correct, train_total, train_loss = 0, 0, 0.0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                train_total += labels.size(0)
                train_correct += (preds == labels).sum().item()

            train_acc = 100.0 * train_correct / max(train_total, 1)
            avg_loss = train_loss / max(len(train_loader), 1)

            # -------- VALIDATION --------
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    val_total += labels.size(0)
                    val_correct += (preds == labels).sum().item()

            val_acc = 100.0 * val_correct / max(val_total, 1)
            scheduler.step()

            # Save per-epoch metrics
            log_csv(log_path, [epoch + 1, train_acc, val_acc, avg_loss])

            if (epoch + 1) % 5 == 0 or (epoch + 1) == EPOCHS:
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | "
                    f"Train Acc: {train_acc:.1f}% | "
                    f"Val Acc: {val_acc:.1f}% | "
                    f"Loss: {avg_loss:.4f}"
                )

        saved_states[name] = model.state_dict()
        print(f">>> Saved weights for: {name}")
        print(f">>> Training log saved to: {log_path}")

    # Machine Learning Model
    print("\n=== TRAINING MACHINE LEARNING MODEL ===")

    if not build_ml or X_train_ml is None:
        print(">>> ML feature extraction is disabled. Skipping Random Forest training.")
    else:
        rf = get_rf_model()
        rf.fit(X_train_ml, y_train_ml)

        rf_val_acc = rf.score(X_val_ml, y_val_ml) if X_val_ml is not None else None
        if rf_val_acc is not None:
            print(f">>> Random Forest Validation Accuracy: {rf_val_acc*100:.1f}%")

        # Save RF model object inside saved_states (as before)
        saved_states["ml_models"] = {"Random Forest": rf}

    # Save everything
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    print(f"\n>>> Menyimpan model ke: {MODEL_SAVE_PATH}")
    torch.save(saved_states, MODEL_SAVE_PATH)
    print(">>> Training selesai.")


if __name__ == "__main__":
    train_all_models()
