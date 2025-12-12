import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DEVICE, EPOCHS, LEARNING_RATE, SELECTED_CLASSES, MODEL_SAVE_PATH
from src.models_dl import TunedMLP, TunedCNN, get_resnet18, get_mobilenet_v2
from src.models_ml import get_rf_model, get_svm_model
from src.data_loader import get_dataloaders

def train_all_models():
    train_loader, val_loader, X_ml, y_ml = get_dataloaders()
    num_classes = len(SELECTED_CLASSES)
    
    dl_models = {
        "MLP (Dropout)": TunedMLP(num_classes).to(DEVICE),
        "CNN (BatchNorm)": TunedCNN(num_classes).to(DEVICE),
        "ResNet18": get_resnet18(num_classes).to(DEVICE),
        "MobileNetV2": get_mobilenet_v2(num_classes).to(DEVICE)
    }
    
    criterion = nn.CrossEntropyLoss()
    saved_states = {'classes': SELECTED_CLASSES}
    
    print("\n=== TRAINING DEEP LEARNING ===")
    for name, model in dl_models.items():
        print(f">>> Training {name}...")
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        for epoch in range(EPOCHS):
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

    X_train, X_test, y_train, y_test = train_test_split(
        X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    print(">>> Training Random Forest...")
    rf = get_rf_model()
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"   Random Forest Test Accuracy: {rf_acc*100:.1f}%")

    # SVM
    print(">>> Training SVM...")
    svm = get_svm_model()
    svm.fit(X_train_scaled, y_train)
    svm_acc = svm.score(X_test_scaled, y_test)
    print(f"   SVM Test Accuracy: {svm_acc*100:.1f}%")
    
    saved_states['ml_models'] = {'Random Forest': rf, 'SVM': svm}
    saved_states['scaler'] = scaler
    
    print(f"\n>>> Menyimpan Model ke {MODEL_SAVE_PATH}...")
    torch.save(saved_states, MODEL_SAVE_PATH)

    print(">>> Selesai!")
