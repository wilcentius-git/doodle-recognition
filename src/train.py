import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
# StandardScaler dihapus karena RF tidak butuh scaling

from src.config import DEVICE, EPOCHS, LEARNING_RATE, SELECTED_CLASSES, MODEL_SAVE_PATH
# Hanya import model yang diminta dosen
from src.models_dl import TunedCNN, get_resnet18
from src.models_ml import get_rf_model
from src.data_loader import get_dataloaders

def train_all_models():
    train_loader, val_loader, X_ml, y_ml = get_dataloaders()
    num_classes = len(SELECTED_CLASSES)
    
    # Hanya CNN dan ResNet18
    dl_models = {
        "CNN (BatchNorm)": TunedCNN(num_classes).to(DEVICE),
        "ResNet18": get_resnet18(num_classes).to(DEVICE)
    }
    
    criterion = nn.CrossEntropyLoss()
    saved_states = {'classes': SELECTED_CLASSES}
    
    print("\n=== TRAINING DEEP LEARNING ===")
    for name, model in dl_models.items():
        print(f">>> Training {name}...")
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        for epoch in range(EPOCHS):
            # TRAINING PHASE
            model.train()
            train_correct = 0
            train_total = 0
            train_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # VALIDATION PHASE
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            scheduler.step()
            
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_loss = train_loss / len(train_loader)
            
            if (epoch+1) % 5 == 0:
                print(f"   Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | Loss: {avg_loss:.4f}")
        
        saved_states[name] = model.state_dict()

    print("\n=== TRAINING MACHINE LEARNING ===")

    # Split train/test untuk ML
    X_train, X_test, y_train, y_test = train_test_split(
        X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml
    )

    # Random Forest (Tanpa Scaler)
    print(">>> Training Random Forest...")
    rf = get_rf_model()
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"   Random Forest Test Accuracy: {rf_acc*100:.1f}%")
    
    # Simpan hanya Random Forest
    saved_states['ml_models'] = {'Random Forest': rf}
    
    print(f"\n>>> Menyimpan Model ke {MODEL_SAVE_PATH}...")
    torch.save(saved_states, MODEL_SAVE_PATH)
    print(">>> Selesai!")
